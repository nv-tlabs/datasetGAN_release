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

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

import sys
sys.path.append('..')

import torch
import torch.nn as nn
torch.manual_seed(0)
import json
from collections import OrderedDict
import numpy as np
import os
device_ids = [0]
from PIL import Image

from models.stylegan1 import G_mapping,Truncation,G_synthesis
import copy
from numpy.random import choice
from utils.utils import latent_to_image, Interpolate
import argparse
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def prepare_stylegan(args):

    if args['stylegan_ver'] == "1":
        if args['category'] == "car":
            resolution = 512
            max_layer = 8
        elif  args['category'] == "face":
            resolution = 1024
            max_layer = 8
        elif args['category'] == "bedroom":
            resolution = 256
            max_layer = 7
        elif args['category'] == "cat":
            resolution = 256
            max_layer = 7
        else:
            assert "Not implementated!"

        if args['average_latent'] != "":
            avg_latent = np.load(args['average_latent'])
            avg_latent = torch.from_numpy(avg_latent).type(torch.FloatTensor).to(device)
        else:
            avg_latent = None
        g_all = nn.Sequential(OrderedDict([
            ('g_mapping', G_mapping()),
            ('truncation', Truncation(avg_latent,max_layer=max_layer, device=device, threshold=0.7)),
            ('g_synthesis', G_synthesis( resolution=resolution))
        ]))

        g_all.load_state_dict(torch.load(args['stylegan_checkpoint'], map_location=device))
        g_all.eval()
        g_all = nn.DataParallel(g_all, device_ids=device_ids).cuda()

        if args['average_latent'] == '':
            avg_latent = g_all.module.g_mapping.make_mean_latent(8000)
            g_all.module.truncation.avg_latent = avg_latent



    else:
        assert "Not implementated error"

    res  = args['dim'][1]
    mode = args['upsample_mode']
    upsamplers = [nn.Upsample(scale_factor=res / 4, mode=mode),
                  nn.Upsample(scale_factor=res / 4, mode=mode),
                  nn.Upsample(scale_factor=res / 8, mode=mode),
                  nn.Upsample(scale_factor=res / 8, mode=mode),
                  nn.Upsample(scale_factor=res / 16, mode=mode),
                  nn.Upsample(scale_factor=res / 16, mode=mode),
                  nn.Upsample(scale_factor=res / 32, mode=mode),
                  nn.Upsample(scale_factor=res / 32, mode=mode),
                  nn.Upsample(scale_factor=res / 64, mode=mode),
                  nn.Upsample(scale_factor=res / 64, mode=mode),
                  nn.Upsample(scale_factor=res / 128, mode=mode),
                  nn.Upsample(scale_factor=res / 128, mode=mode),
                  nn.Upsample(scale_factor=res / 256, mode=mode),
                  nn.Upsample(scale_factor=res / 256, mode=mode)
                  ]

    if resolution > 256:
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode))
        upsamplers.append(nn.Upsample(scale_factor=res / 512, mode=mode))

    if resolution > 512:

        upsamplers.append(Interpolate(res, 'bilinear'))
        upsamplers.append(Interpolate(res, 'bilinear'))

    return g_all, avg_latent, upsamplers


def generate_data(args, num_sample, sv_path):
    # use face_palette because it has most classes
    from utils.data_util import face_palette as palette



    if os.path.exists(sv_path):
        pass
    else:
        os.system('mkdir -p %s' % (sv_path))
        print('Experiment folder created at: %s' % (sv_path))


    g_all, avg_latent, upsamplers = prepare_stylegan(args)

    # dump avg_latent for reproducibility
    mean_latent_sv_path = os.path.join(sv_path, "avg_latent_stylegan1.npy")
    np.save(mean_latent_sv_path, avg_latent[0].detach().cpu().numpy())


    with torch.no_grad():
        latent_cache = []

        results = []
        np.random.seed(1111)


        print( "num_sample: ", num_sample)


        for i in range(num_sample):
            if i % 10 == 0:
                print("Genearte", i, "Out of:", num_sample)

            if i == 0:

                latent = avg_latent.to(device)
                img, _ = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1],
                                         return_upsampled_layers=False, use_style_latents=True)
            else:
                latent = np.random.randn(1, 512)
                latent_cache.append(copy.deepcopy(latent))


                latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device)

                img, _ = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1],
                                                         return_upsampled_layers=False)

            if args['dim'][0] != args['dim'][1]:
                img = img[:, 64:448][0]
            else:
                img = img[0]


            img = Image.fromarray(img)

            image_name =  os.path.join(sv_path, "image_%d.jpg" % i)
            img.save(image_name)

        latent_cache = np.concatenate(latent_cache, 0)
        latent_sv_path = os.path.join(sv_path, "latent_stylegan1.npy")
        np.save(latent_sv_path, latent_cache)

        for latent in latent_cache:

            latent = torch.from_numpy(latent).type(torch.FloatTensor).to(device).unsqueeze(0)
            img, _ = latent_to_image(g_all, upsamplers, latent, dim=args['dim'][1],
                                     return_upsampled_layers=False)

            if args['dim'][0] != args['dim'][1]:
                img = img[:, 64:448][0]
            else:
                img = img[0]


            img = Image.fromarray(img)

            image_name =  os.path.join(sv_path, 'reconstruct', "image_%d.jpg" % i)
            img.save(image_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str)
    parser.add_argument('--num_sample', type=int,  default=100)
    parser.add_argument('--sv_path', type=str)

    args = parser.parse_args()

    opts = json.load(open(args.exp, 'r'))
    print("Opt", opts)


    path =opts['exp_dir']
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))

    os.system('cp %s %s' % (args.exp, opts['exp_dir']))



    generate_data(opts, args.num_sample, args.sv_path)
