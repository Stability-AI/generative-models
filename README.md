# Generative Models by Stability AI

![sample1](assets/000.jpg)

## News

**June 22, 2023**

- We are releasing two new diffusion models:
  - `SD-XL 0.9-base`: The base model was trained on a variety of aspect ratios on images with resolution 1024^2. The base model uses [OpenCLIP-ViT/G](https://github.com/mlfoundations/open_clip) and [CLIP-ViT/L](https://github.com/openai/CLIP/tree/main) for text encoding whereas the refiner model only uses the OpenCLIP model.
  - `SD-XL 0.9-refiner`: The refiner has been trained to denoise small noise levels of high quality data and as such is not expected to work as a text-to-image model; instead, it should only be used as an image-to-image model.

## The codebase

### General Philosophy

Modularity is king. This repo implements a config-driven approach where we build and combine submodules by calling `instantiate_from_config()` on objects defined in yaml configs. See `configs/` for many examples.

### Changelog from the old `ldm` codebase

For training, we use [pytorch-lightning](https://www.pytorchlightning.ai/index.html), but it should be easy to use other training wrappers around the base modules. The core diffusion model class (formerly `LatentDiffusion`, now `DiffusionEngine`) has been cleaned up:

- No more extensive subclassing! We now handle all types of conditioning inputs (vectors, sequences and spatial conditionings, and all combinations thereof) in a single class: `GeneralConditioner`, see `sgm/modules/encoders/modules.py`.
- We separate guiders (such as classifier-free guidance, see `sgm/modules/diffusionmodules/guiders.py`) from the
  samplers (`sgm/modules/diffusionmodules/sampling.py`), and the samplers are independent of the model.
- We adopt the ["denoiser framework"](https://arxiv.org/abs/2206.00364) for both training and inference (most notable change is probably now the option to train continuous time models):
    * Discrete times models (denoisers) are simply a special case of continuous time models (denoisers); see `sgm/modules/diffusionmodules/denoiser.py`.
    * The following features are now independent: weighting of the diffusion loss function (`sgm/modules/diffusionmodules/denoiser_weighting.py`), preconditioning of the network (`sgm/modules/diffusionmodules/denoiser_scaling.py`), and sampling of noise levels during training (`sgm/modules/diffusionmodules/sigma_sampling.py`).
- Autoencoding models have also been cleaned up.

## Installation:
<a name="installation"></a>

#### 1. Clone the repo

```shell
git clone git@github.com:Stability-AI/generative-models.git
cd generative-models
```

#### 2. Setting up the virtualenv

This is assuming you have navigated to the `generative-models` root after cloning it. 

**NOTE:** This is tested under `python3.8` and `python3.10`. For other python versions, you might encounter version conflicts.


**PyTorch 1.13** 

```shell
# install required packages from pypi
python3 -m venv .pt1
source .pt1/bin/activate
pip3 install wheel
pip3 install -r requirements_pt13.txt
```

**PyTorch 2.0** 


```shell
# install required packages from pypi
python3 -m venv .pt2
source .pt2/bin/activate
pip3 install wheel
pip3 install -r requirements_pt2.txt
```

#### 3. Installation of the required packages from pypi

```shell
pip3 install .
```

## Inference:

We provide a streamlit demo for text-to-image and image-to-image sampling in `scripts/demo/sampling.py`. The following models are currently supported:
- [SD-XL 0.9-base](https://huggingface.co/stabilityai/stable-diffusion-xl-base-0.9)
- [SD-XL 0.9-refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-0.9)
- [SD 2.1-512](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.safetensors)
- [SD 2.1-768](https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.safetensors)

Please download the checkpoints from hugginface and place them into `checkpoints/`. Afterwards, you can start the demo using

```
streamlit run scripts/streamlit/sampling.py --server.port <your_port>
```

## Training:

We are providing example training configs in `configs/example_training`. To launch a training, run

```
python main.py --base configs/<config1.yaml> configs/<config2.yaml>
```

where configs are merged from left to right (later configs overwrite the same values).
This can be used to combine model, training and data configs. However, all of them can also be
defined in a single config. For example, to run a class-conditional pixel-based diffusion model training on MNIST,
run

```bash
python main.py --base configs/example_training/toy/mnist_cond.yaml
```

**NOTE 1:** Using the non-toy-dataset configs `configs/example_training/imagenet-f8_cond.yaml`, `configs/example_training/txt2img-clipl.yaml` and `configs/example_training/txt2img-clipl-legacy-ucg-training.yaml` for training will require edits depdending on the used dataset (which is expected to stored in tar-file in the [webdataset-format](https://github.com/webdataset/webdataset)). To find the parts which have to be adapted, search for comments conaining `USER:` in the respective config.

**NOTE 2:** This repository supports both `pytorch1.13` and `pytorch2`for training generative models. However for autoencoder training as e.g. in `configs/example_training/autoencoder/kl-f4/imagenet-attnfree-logvar.yaml`, only `pytorch1.13` is supported.

### Building New Diffusion Models

#### Conditioner

The `GeneralConditioner` is configured through the `conditioner_config`. Its only attribute is `emb_models`, a list of
different embedders (all inherited from `AbstractEmbModel`) that are used to condition the generative model.
All embedders should define whether or not they are trainable (`is_trainable`, default `False`), a classifier-free
guidance dropout rate is used (`ucg_rate`, default `0`), and an input key (`input_key`), for example, `txt` for text-conditioning or `cls` for class-conditioning.
When computing conditionings, the embedder will get `batch[input_key]` as input.
We currently support two to four dimensional conditionings and conditionings of different embedders are concatenated
appropriately.
Note that the order of the embedders in the `conditioner_config` is important.

#### Network

The neural network is set through the `network_config`. This used to be called `unet_config`, which is not general
enough as we plan to experiment with transformer-based diffusion backbones.

#### Loss

The loss is configured through `loss_config`. For standard diffusion model training, you will have to set `sigma_sampler_config`.

#### Sampler config

As discussed above, the sampler is independent of the model. In the `sampler_config`, we set the type of numerical
solver, number of steps, type of discretization, as well as, for example, guidance wrappers for classifier-free
guidance.

### Dataset Handling


For large scale training we recommend using the datapipelines from our [datapipelines](https://github.com/Stability-AI/datapipelines) project. The project is contained in the requirement and automatically included when following the steps from the [Installation section](#installation).
Small map-style datasets should be defined here in the repository (e.g., MNIST, CIFAR-10, ...), and return a dict of
data keys/values,
e.g.,

```python
example = {"jpg": x,  # this is a tensor -1...1 chw
           "txt": "a beautiful image"}
```

where we expect images in -1...1, channel-first format.

