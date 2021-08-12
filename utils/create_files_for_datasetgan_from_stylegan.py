"""
Generate the files that are needed to run DatasetGAN from a PyTorch
StyleGAN checkpoint:

    1. the average w latent code,
    2. images generated by StyleGAN and
    3. the latent code from StyleGAN that is used to generate each image.

In the configuration files of NVIDIA's DatasetGAN repo,
https://github.com/nv-tlabs/datasetGAN_release/tree/master/datasetGAN/experiments,
there are two fields, "average_latent" and "annotation_image_latent_path".

    * "average_latent" is the path to a NumPy binary of the average w latent
      code. For NVIDIA's cat example, the shape is (18, 512).

    * "annotation_image_latent_path" is the path to a NumPy binary. The binary
      contains the latent code from StyleGAN that is used to generate each of
      the images in your DatasetGAN training (and testing?) set. For NVIDIA's
      cat example, the shape is (30, 512).

Images will be generated with the names "image_{i}.jpg". The matching mask
must be manually created and should get the name "image_mask{i}.jpg".

DatasetGAN repo:
https://github.com/nv-tlabs/datasetGAN_release

NVIDIA's PyTorch checkpoints, which this function uses, can be downloaded
here:
https://drive.google.com/drive/folders/1Hhu8aGxbnUtK-yHRD9BXZ3gn3bNNmsGi
"""
import argparse
import logging
import logging.config
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import yaml
from matplotlib import pyplot
from PIL import Image

from utils import Interpolate, latent_to_image

DIR_MODELS = "../models"
sys.path.append(DIR_MODELS)
from stylegan1 import G_mapping, G_synthesis, Truncation

DIR_DATASETGAN = "../datasetGAN"
sys.path.append(DIR_DATASETGAN)
from train_interpreter import prepare_stylegan


logger = logging.getLogger(__name__)
with open("logging.yml", "r") as f:
    config = yaml.safe_load(f.read())
logging.config.dictConfig(config)


DEFAULT_RESOLUTION = 256
DEFAULT_MAX_LAYERS = 7
DEFAULT_NUM_IMAGES = 30
DEFAULT_UPSAMPLE_MODE = "bilinear"
DEFAULT_PATH_AVG_LATENT = Path("avg_latent.npy")
DEFAULT_PATH_LATENT_USED_TO_GENERATE_IMAGES = "latent_stylegan1.npy"


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.manual_seed(0)
device_ids = [0]


def load_stylegan_checkpoint(
    stylegan_checkpoint_path,
    average_w_latent_code: Optional[np.ndarray] = None,  # or torch.Tensor
    which_layers_to_load: str = "all",
    resolution: int = DEFAULT_RESOLUTION,
    device: str = "cuda",
):
    """
    Load all of StyleGAN or load pieces of the network.

    NVIDIA's PyTorch checkpoints, which this function uses, can be downloaded
    here:
    https://drive.google.com/drive/folders/1Hhu8aGxbnUtK-yHRD9BXZ3gn3bNNmsGi
    """
    assert which_layers_to_load in ["all", "mapping", "mapping and truncation"]
    assert type(average_w_latent_code) in (type(None), np.ndarray, torch.Tensor)

    if type(average_w_latent_code) == np.ndarray:
        average_w_latent_code = torch.from_numpy(average_w_latent_code)
        if device != "cpu":
            average_w_latent_code = average_w_latent_code.to(device)

    state_dictionary = torch.load(stylegan_checkpoint_path)

    if which_layers_to_load == "all":
        model = nn.Sequential(
            OrderedDict(
                [
                    ("g_mapping", G_mapping()),
                    ("truncation", Truncation(average_w_latent_code, device=device,),),
                    ("g_synthesis", G_synthesis(resolution=resolution)),
                ]
            )
        )
        model.load_state_dict(state_dictionary)

    elif which_layers_to_load == "mapping and truncation":
        mapping_and_truncation_state_dictionary = OrderedDict(
            list(state_dictionary.items())[:16]
        )  # remove everything besides the mapping and truncation network
        model = nn.Sequential(
            OrderedDict(
                [
                    ("g_mapping", G_mapping()),
                    ("truncation", Truncation(average_w_latent_code, device=device),),
                ]
            )
        )
        model.load_state_dict(mapping_and_truncation_state_dictionary)

    elif which_layers_to_load == "mapping":
        mapping_state_dictionary = OrderedDict(
            list(state_dictionary.items())[:16]
        )  # remove everything besides mapping network
        model = nn.Sequential(OrderedDict([("g_mapping", G_mapping())]))
        model.load_state_dict(mapping_state_dictionary)

    model.eval()
    model.to(device)
    return model


def generate_random_z_latent_code(n: int = 1, device: str = "cuda"):
    return torch.randn(n, 512, device=device)


def generate_average_w_latent_code(
    model,
    path_to_save: Path = DEFAULT_PATH_AVG_LATENT,
    num_w_latent_codes_to_generate: int = 8000,
    device: str = "cuda",
):
    """
    Generate the average w latent code. This is a pre-requisite for DatasetGAN.

    Loops through generating random inputs for StyleGAN, gives the inputs to
    StyleGAN, collects the outputs, calculates the element-wise average and
    then save the average as a NumPy binary.
    """
    w_latent_code_list = []
    for i in range(0, num_w_latent_codes_to_generate):
        random_tensor = generate_random_z_latent_code(device=device)
        w_latent_code = model.g_mapping(random_tensor)
        [np_arr] = w_latent_code.cpu().detach().numpy()
        w_latent_code_list.append(np_arr)
    average_w_latent_code = np.mean(w_latent_code_list, axis=0)
    if path_to_save is not None:
        np.save(path_to_save, average_w_latent_code)
        logger.info(f"saved average of w latent space: {path_to_save}")
    return average_w_latent_code


def generate_latent_space_and_image_pairs_using_nvidia_code(
    model,
    upsamplers,
    output_image_dir: Path,
    num_images: int,
    resolution: int,
    path_latent_used_to_generate_images: Path = DEFAULT_PATH_LATENT_USED_TO_GENERATE_IMAGES,
    device: str = "cuda",
):
    """
    Generate image and latent space pairs.
    """
    logger.debug(f"num images to generate: {num_images}")
    w_latent_code_list = []
    with torch.no_grad():
        for i in range(0, num_images):
            torch.cuda.empty_cache()
            random_tensor = generate_random_z_latent_code(
                n=1, device=device
            )  # nvidia asserts that length of tensor == 1
            [numpy_permuted_img], affine_layer_upsamples = latent_to_image(
                model, upsamplers, random_tensor, dim=resolution,
            )
            pyplot.imsave(Path(output_image_dir, f"image_{i}.jpg"), numpy_permuted_img)
            [random_tensor_in_format_needed] = random_tensor.cpu().detach().numpy()
            w_latent_code_list.append(random_tensor_in_format_needed)
    np.save(path_latent_used_to_generate_images, w_latent_code_list)
    logger.info(f"generated {num_images} images: {output_image_dir}")
    logger.info(f"saved latent code: {path_latent_used_to_generate_images}")


def main(
    stylegan_checkpoint_path: Path,
    output_image_dir: Path,
    num_images: int,
    path_average_w_latent_code: Path = DEFAULT_PATH_AVG_LATENT,
    path_latent_used_to_generate_images: Path = DEFAULT_PATH_LATENT_USED_TO_GENERATE_IMAGES,
    resolution: int = DEFAULT_RESOLUTION,
    max_layers: int = DEFAULT_MAX_LAYERS,
    device: str = "cuda",
):
    if output_image_dir.exists():
        logger.debug(f"already exists: {output_image_dir}")
    else:
        output_image_dir.mkdir()
        logger.debug(f"mkdir: {output_image_dir}")

    # 1. generate average latent code
    mapping_network = load_stylegan_checkpoint(stylegan_checkpoint_path, device=device)
    average_w_latent_code = generate_average_w_latent_code(
        mapping_network,
        path_to_save=path_average_w_latent_code,
        device=device,
    )

    # 2. load the network with average latent code
    args_for_nvidia = {
        "stylegan_ver": "1",
        "category": "cat",
        "average_latent": path_average_w_latent_code,
        "stylegan_checkpoint": stylegan_checkpoint_path,
        "dim": [
            resolution,
            resolution,
            4992,
        ],  # TODO: what does the 4992 mean? taken from nvidia code
        "upsample_mode": "bilinear",
    }
    entire_network, other_average_w_latent_code, upsamplers = prepare_stylegan(
        args_for_nvidia
    )

    # 3. generate images and save associated z latent code
    generate_latent_space_and_image_pairs_using_nvidia_code(
        entire_network,
        upsamplers,
        output_image_dir,
        num_images,
        resolution=resolution,
        path_latent_used_to_generate_images=path_latent_used_to_generate_images,
        device=device,
    )
main.__doc__ = __doc__


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_image_dir", type=Path, metavar="output-image-dir")
    parser.add_argument(
        "num_images", type=int, metavar="num-images", default=DEFAULT_NUM_IMAGES
    )
    parser.add_argument(
        "--path-average-w-latent-code", type=Path, default=DEFAULT_PATH_AVG_LATENT
    )
    parser.add_argument(
        "--path-latent-used-to-generate-images",
        type=Path,
        default=DEFAULT_PATH_LATENT_USED_TO_GENERATE_IMAGES,
    )
    parser.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION)
    parser.add_argument("--max-layers", type=int, default=DEFAULT_MAX_LAYERS)
    parser.add_argument("--disable-cuda", action="store_true")
    opts = parser.parse_args()

    if opts.disable_cuda:
        device = "cpu"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logger.debug(f"pytorch device: {device}")

    main(
        opts.checkpoint,
        opts.output_image_dir,
        opts.num_images,
        path_average_w_latent_code=opts.path_average_w_latent_code,
        path_latent_used_to_generate_images=opts.path_latent_used_to_generate_images,
        resolution=opts.resolution,
        device=device,
    )
