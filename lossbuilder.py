import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class LossBuilder:
    def __init__(self, device):
        self.vgg_path = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
        self.vgg_used = False
        self.device = device

    # create a module to normalize input image so we can easily put it in a
    # nn.Sequential
    class VGGNormalization(nn.Module):
        def __init__(self, device):
            super().__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(device)
            self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(device)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std

    class PerceptualLoss(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight
        def forward(self, x):
            gt_pool, pred_pool = torch.chunk(x, 2, dim=0)
            self.loss = F.mse_loss(gt_pool, pred_pool) * self.weight
            return x

    def get_style_and_content_loss(self,
                                   content_layers):

        cnn = models.vgg19(pretrained=True).features.eval().cuda()
        for p in cnn.parameters():
            p.requires_grad = False

        # normalization module
        normalization = LossBuilder.VGGNormalization(self.device)

        # losses
        content_losses = []

        # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
        # to put in modules that are supposed to be activated sequentially
        model = nn.Sequential(normalization)

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ContentLoss
                # we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers.keys():
                # add content loss:
                weight = content_layers[name]
                content_loss = LossBuilder.PerceptualLoss(weight)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

        # now we trim off the layers after the last content loss
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], LossBuilder.PerceptualLoss):
                break

        model = model[:(i + 1)]

        return model, content_losses