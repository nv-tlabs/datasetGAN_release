"""
Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
Licensed under The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import torch
from PIL import Image
import numpy as np
from torch import nn

class Interpolate(nn.Module):
    def __init__(self, size, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners=False)
        return x



def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = acc * 100

    return acc


def oht_to_scalar(y_pred):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    return y_pred_tags

def latent_to_image(g_all, upsamplers, latents, return_upsampled_layers=False, use_style_latents=False,
                    style_latents=None, process_out=True, return_stylegan_latent=False, dim=512, return_only_im=False):
    '''Given a input latent code, generate corresponding image and concatenated feature maps'''

    # assert (len(latents) == 1)  # for GPU memory constraints
    if not use_style_latents:
        # generate style_latents from latents
        style_latents = g_all.module.truncation(g_all.module.g_mapping(latents))
        style_latents = style_latents.clone()  # make different layers non-alias

    else:
        style_latents = latents

        # style_latents = latents
    if return_stylegan_latent:

        return  style_latents
    img_list, affine_layers = g_all.module.g_synthesis(style_latents)

    if return_only_im:
        if process_out:
            if img_list.shape[-2] > 512:
                img_list = upsamplers[-1](img_list)

            img_list = img_list.cpu().detach().numpy()
            img_list = process_image(img_list)
            img_list = np.transpose(img_list, (0, 2, 3, 1)).astype(np.uint8)
        return img_list, style_latents

    number_feautre = 0

    for item in affine_layers:
        number_feautre += item.shape[1]


    affine_layers_upsamples = torch.FloatTensor(1, number_feautre, dim, dim).cuda()
    if return_upsampled_layers:

        start_channel_index = 0
        for i in range(len(affine_layers)):
            len_channel = affine_layers[i].shape[1]
            affine_layers_upsamples[:, start_channel_index:start_channel_index + len_channel] = upsamplers[i](
                affine_layers[i])
            start_channel_index += len_channel

    if img_list.shape[-2] != 512:
        img_list = upsamplers[-1](img_list)

    if process_out:
        img_list = img_list.cpu().detach().numpy()
        img_list = process_image(img_list)
        img_list = np.transpose(img_list, (0, 2, 3, 1)).astype(np.uint8)
        # print('start_channel_index',start_channel_index)


    return img_list, affine_layers_upsamples


def process_image(images):
    drange = [-1, 1]
    scale = 255 / (drange[1] - drange[0])
    images = images * scale + (0.5 - drange[0] * scale)

    images = images.astype(int)
    images[images > 255] = 255
    images[images < 0] = 0

    return images.astype(int)

def colorize_mask(mask, palette):
    # mask: numpy array of the mask

    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return np.array(new_mask.convert('RGB'))


def get_label_stas(data_loader):
    count_dict = {}
    for i in range(data_loader.__len__()):
        x, y = data_loader.__getitem__(i)
        if int(y.item()) not in count_dict:
            count_dict[int(y.item())] = 1
        else:
            count_dict[int(y.item())] += 1

    return count_dict
