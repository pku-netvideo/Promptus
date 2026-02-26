import cv2 as cv
from scripts.demo.streamlit_helpers import *
from sgm.modules.diffusionmodules.sampling import EulerAncestralSampler
from lossbuilder import LossBuilder
from quantization import QParam, FakeQuantize
from diffusers import AutoencoderTiny
import argparse

VERSION2SPECS = {
    "SDXL-Turbo": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_xl_base.yaml",
        "ckpt": "checkpoints/sd_xl_turbo_1.0_fp16.safetensors",
    },
    "SD-Turbo": {
        "H": 512,
        "W": 512,
        "C": 4,
        "f": 8,
        "is_legacy": False,
        "config": "configs/inference/sd_2_1.yaml",
        "ckpt": "checkpoints/sd_turbo.safetensors",
    },
}


class SubstepSampler(EulerAncestralSampler):
    def __init__(self, n_sample_steps=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_sample_steps = n_sample_steps
        self.steps_subset = [0, 100, 200, 300, 1000]

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(
            self.num_steps if num_steps is None else num_steps, device=self.device
        )
        sigmas = sigmas[
            self.steps_subset[: self.n_sample_steps] + self.steps_subset[-1:]
            ]
        uc = cond
        x = x * torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)
        s_in = x.new_ones([x.shape[0]])
        return x, s_in, sigmas, num_sigmas, cond, uc


def seeded_randn(shape, seed):
    randn = np.random.RandomState(seed).randn(*shape)
    randn = torch.from_numpy(randn).to(device="cuda", dtype=torch.float32)
    return randn


class SeededNoise:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, x):
        self.seed = self.seed + 1
        return seeded_randn(x.shape, self.seed)


def inversion(
        model,
        sampler,
        decoder,
        rank,
        interval,
        frame_path,
        max_id,
        H=512,
        W=512,
        seed=0,
        filter=None
):
    F = 8
    C = 4
    shape = (1, C, H // F, W // F)

    if seed is None:
        seed = torch.seed()
    precision_scope = autocast
    with precision_scope("cuda"):
        def denoiser(input, sigma, c):
            return model.denoiser(
                model.model,
                input,
                sigma,
                c,
            )

        def load_img(path):
            img = cv.imread(path)
            img = img[:, :, ::-1]
            H, W, C = img.shape
            l, r = int(W / 2 - H / 2), int(W / 2 + H / 2)
            img = img[:, l:r, :]
            img = cv.resize(img, [512, 512])
            img = (img / 255) * 2 - 1
            img = torch.from_numpy(img)
            img = img.float()
            img = img.permute(2, 0, 1)
            img = img.unsqueeze(dim=0)
            img = img.cuda()
            return img

        uc = None
        rand_noise = seeded_randn(shape, seed)
        sigma = torch.Tensor([0.05]).float().cuda()

        # Set the loss functions.
        # Reconstruction loss.
        mse_loss = torch.nn.MSELoss()
        # Perceptual loss.
        builder = LossBuilder('cuda')
        content_layers = [('conv_1', 1), ('conv_2', 1), ('conv_3', 1), ('conv_4', 1),
                          ('conv_5', 1), ('conv_6', 1), ('conv_7', 1), ('conv_8', 1),
                          ('conv_9', 1), ('conv_10', 1), ('conv_11', 1), ('conv_12', 1),
                          ('conv_13', 1), ('conv_14', 1), ('conv_15', 1),
                          ('conv_16', 1)]
        vgg_model, lpips_nodes = builder.get_style_and_content_loss(dict(content_layers))

        # Inversion.
        for f_id in range(0, max_id, interval):
            # Initialize the low-rank factor U.
            U = torch.rand([77, rank]).float().cuda()
            U.requires_grad = True
            # Fake Quantizer for U
            Quant_Param_U = QParam(num_bits=8)
            # Initialize the low-rank factor V.
            V = torch.rand([rank, 1024]).float().cuda()
            V.requires_grad = True
            # Fake Quantizer for V
            Quant_Param_V = QParam(num_bits=8)

            # Initialize the learning rate and the optimizer.
            lr = 0.1
            optimizer = torch.optim.Adam([U, V], lr=lr)

            # for learning rate scheduler and logging
            min_loss = 1e9
            latest_min_loss = 1e9

            # logs and results path
            prompt_path = os.path.join(frame_path, 'results/rank{}_interval{}/'.format(rank, interval))
            log_path = os.path.join(prompt_path,'{:05d}'.format(f_id))
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            log_output = open(os.path.join(log_path,'log.txt'), 'a')

            if f_id > 0:
                ckpt_prev = torch.load(os.path.join(prompt_path,'{:05d}/ckpt.pth'.format(f_id - interval)))
                U_prev, V_prev = ckpt_prev["U"], ckpt_prev["V"]
                prev_frame = ckpt_prev["z"]
                # Subsequent frames require fewer iterations.
                # Reduce total_iterations to speed up inversion, but this may lower quality.
                total_iterations = 1500
                lr_schedule_cnt = 20
                step_list_base = [_ for _ in range(1, interval + 1)]
            else:
                # Initialization of the first frame.
                prev_frame = model.encode_first_stage(load_img(os.path.join(frame_path,'00000.png')))
                torch.save(prev_frame, os.path.join(prompt_path, 'init.pth'))
                # The first frame requires more iterations.
                total_iterations = 10000
                lr_schedule_cnt = 300
                step_list_base = [0]
            # add random noise to the previous frame
            randn = (prev_frame * sigma + rand_noise * (1 - sigma)).detach()
            step_list = step_list_base

            for iter in range(total_iterations):
                loss_list = {}
                for step in step_list:
                    # Fake Quantification
                    Quant_Param_U.update(U)
                    Q_U = FakeQuantize.apply(U, Quant_Param_U)
                    Quant_Param_V.update(V)
                    Q_V = FakeQuantize.apply(V, Quant_Param_V)

                    if f_id > 0:
                        # perform linear interpolation on keyframe prompts
                        # approximating the intermediate prompts.
                        factor = 1 / interval
                        u = (1 - step * factor) * U_prev + (step * factor) * Q_U
                        v = (1 - step * factor) * V_prev + (step * factor) * Q_V
                        # prompt composition
                        c = (u @ v / np.sqrt(rank)).unsqueeze(dim=0)
                        cur_id = f_id - interval + step
                    else:
                        # for the first frame
                        c = (Q_U @ Q_V / np.sqrt(rank)).unsqueeze(dim=0)
                        cur_id = 0
                    c = {'crossattn': c}

                    # generating a frame
                    samples_z = sampler(denoiser, randn, cond=c, uc=uc)
                    samples_x = decoder(samples_z)

                    # Calculating loss
                    gt = load_img(os.path.join(frame_path, '{:05d}.png'.format(cur_id)))
                    gt.requires_grad = True
                    # Perceptual loss.
                    vgg_model(torch.cat([gt, samples_x], dim=0))
                    lpips_loss = 0
                    for node in lpips_nodes:
                        lpips_loss += node.loss
                    lpips_loss = lpips_loss / (len(lpips_nodes) + 1e-9)
                    # Combine the perceptual loss and reconstruction loss.
                    loss = 0.2 * lpips_loss + 0.8 * mse_loss(samples_x, gt)

                    # regularization
                    loss_regu = torch.mean(torch.abs(c['crossattn']))
                    loss = loss + 0.1 * loss_regu

                    # logging
                    print('iter: {}, cur_id: {}, loss: {}, c_max: {}, c_mean: {}, c_std: {}'.format(iter, cur_id, loss, c['crossattn'].max(), c['crossattn'].mean(), c['crossattn'].std()))
                    log_output.write('iter: {}, cur_id: {}, loss: {}, c_max: {}, c_mean: {}, c_std: {}\n'.format(iter, cur_id, loss, c['crossattn'].max(), c['crossattn'].mean(), c['crossattn'].std()))
                    log_output.flush()

                    # saving the generated frames
                    if iter % 10 == 0:
                        img = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                        img = (
                            (255 * img)
                                .to(dtype=torch.uint8)
                                .permute(0, 2, 3, 1)
                                .detach()
                                .cpu()
                                .numpy()
                        )
                        img = img[0][:, :, ::-1]
                        cv2.imwrite(os.path.join(log_path,'iter_{:05d}_id_{:05d}.png'.format(iter, cur_id)), img)

                    # Optimization
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    model.model.zero_grad()

                    # for learning rate scheduler and logging
                    if cur_id not in loss_list.keys():
                        loss_list[cur_id] = loss.item()
                mean_loss = np.mean(list(loss_list.values()))
                print('iter: {}, mean loss: {}'.format(iter, mean_loss))
                log_output.write('iter: {}, mean loss: {}\n'.format(iter, mean_loss))
                log_output.flush()
                if mean_loss < min_loss:
                    # saving the ckpt
                    min_loss = mean_loss
                    ckpt = {
                        'U': Q_U,
                        'U_scale': Quant_Param_U.scale,
                        'U_zero_point': Quant_Param_U.zero_point,
                        'U_bits': Quant_Param_U.num_bits,
                        'V': Q_V,
                        'V_scale': Quant_Param_V.scale,
                        'V_zero_point': Quant_Param_V.zero_point,
                        'V_bits': Quant_Param_V.num_bits,
                        'z': samples_z,
                        'randn': randn,
                        'iter': iter,
                        'loss': mean_loss,
                    }
                    torch.save(ckpt, os.path.join(log_path, 'ckpt.pth'))
                    # saving the prompt
                    U_Byte = Quant_Param_U.quantize_tensor(Q_U).byte()
                    V_Byte = Quant_Param_V.quantize_tensor(Q_V).byte()
                    prompt = {
                        'U': U_Byte,
                        'V': V_Byte,
                        'U_scale': Quant_Param_U.scale,
                        'U_zero_point': Quant_Param_U.zero_point,
                        'V_scale': Quant_Param_V.scale,
                        'V_zero_point': Quant_Param_V.zero_point,
                    }
                    torch.save(prompt, os.path.join(prompt_path, 'frame_{:05d}.prompt'.format(f_id)))
                if f_id > 0:
                    # Dynamic training.
                    # Allocating more training to the frames with the worst performance.
                    worst_step = max(loss_list, key=loss_list.get) - f_id + interval
                    step_list = step_list_base + [worst_step] * 2
                    step_list = sorted(step_list)
                # Learning rate scheduler
                lr_schedule_cnt = lr_schedule_cnt - 1
                if lr_schedule_cnt == 0:
                    if min_loss == latest_min_loss:
                        # Reduce the learning rate by half.
                        lr = max(lr * 0.5, 0.001)
                        optimizer = torch.optim.Adam([U, V], lr=lr)
                        print('reduce lr to: {}'.format(optimizer.param_groups[0]['lr']))
                        log_output.write('reduce lr to: {}\n'.format(optimizer.param_groups[0]['lr']))
                        log_output.flush()
                    latest_min_loss = min_loss
                    lr_schedule_cnt = 20 if f_id > 0 else 300
            log_output.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-frame_path', type=str, default="data/sky")
    parser.add_argument('-max_id', type=int, default=140)
    parser.add_argument('-rank', type=int, default="8")
    parser.add_argument('-interval', type=int, default="10")
    args = parser.parse_args()

    # Set up and load the models.
    version_dict = VERSION2SPECS['SD-Turbo']
    state = init_st(version_dict, load_filter=True)
    if state["msg"]:
        st.info(state["msg"])
    model = state["model"]
    load_model(model)
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float32).cuda()
    sampler = SubstepSampler(
        n_sample_steps=1,
        num_steps=1000,
        eta=1.0,
        discretization_config=dict(
            target="sgm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization"
        ),
    )
    seed_ = 88
    sampler.noise_sampler = SeededNoise(seed=seed_)

    # Inversion: from video to prompts
    inversion(
       model, sampler, decoder=taesd.decoder, rank=args.rank, interval=args.interval, frame_path=args.frame_path, max_id=args.max_id, H=512, W=512, seed=seed_, filter=state.get("filter")
    )