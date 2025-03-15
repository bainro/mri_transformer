<img src="./images/vit.gif" width="500px"></img>

## Table of Contents

- [Vision Transformer - Pytorch](#vision-transformer---pytorch)
- [Install](#install)
- [Usage](#usage)
- [Parameters](#parameters)
- [Simple ViT](#simple-vit)
- [3D Vit](#3d-vit)
- [Accessing Attention](#accessing-attention)

## Vision Transformer - Pytorch

Implementation of <a href="https://openreview.net/pdf?id=YicbFdNTTy">Vision Transformer</a>, a simple way to achieve SOTA in vision classification with only a single transformer encoder, in Pytorch. Significance is further explained in <a href="https://www.youtube.com/watch?v=TrdevFK_am4">Yannic Kilcher's</a> video. There's really not much to code here, but may as well lay it out for everyone so we expedite the attention revolution.

For a Pytorch implementation with pretrained models, please see Ross Wightman's repository <a href="https://github.com/rwightman/pytorch-image-models">here</a>.

The official Jax repository is <a href="https://github.com/google-research/vision_transformer">here</a>.

A tensorflow2 translation also exists <a href="https://github.com/taki0112/vit-tensorflow">here</a>, created by research scientist <a href="https://github.com/taki0112">Junho Kim</a>! 🙏

<a href="https://github.com/conceptofmind/vit-flax">Flax translation</a> by <a href="https://github.com/conceptofmind">Enrico Shippole</a>!

## Install

```bash
$ pip install vit-pytorch
```

## Usage

```python
import torch
from vit_pytorch import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)
```

## Parameters

- `image_size`: int.  
Image size. If you have rectangular images, make sure your image size is the maximum of the width and height
- `patch_size`: int.  
Size of patches. `image_size` must be divisible by `patch_size`.  
The number of patches is: ` n = (image_size // patch_size) ** 2` and `n` **must be greater than 16**.
- `num_classes`: int.  
Number of classes to classify.
- `dim`: int.  
Last dimension of output tensor after linear transformation `nn.Linear(..., dim)`.
- `depth`: int.  
Number of Transformer blocks.
- `heads`: int.  
Number of heads in Multi-head Attention layer. 
- `mlp_dim`: int.  
Dimension of the MLP (FeedForward) layer. 
- `channels`: int, default `3`.  
Number of image's channels. 
- `dropout`: float between `[0, 1]`, default `0.`.  
Dropout rate. 
- `emb_dropout`: float between `[0, 1]`, default `0`.  
Embedding dropout rate.
- `pool`: string, either `cls` token pooling or `mean` pooling


## Simple ViT

<a href="https://arxiv.org/abs/2205.01580">An update</a> from some of the same authors of the original paper proposes simplifications to `ViT` that allows it to train faster and better.

Among these simplifications include 2d sinusoidal positional embedding, global average pooling (no CLS token), no dropout, batch sizes of 1024 rather than 4096, and use of RandAugment and MixUp augmentations. They also show that a simple linear at the end is not significantly worse than the original MLP head

You can use it by importing the `SimpleViT` as shown below

```python
import torch
from vit_pytorch import SimpleViT

v = SimpleViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)
```

## 3D ViT

By popular request, I will start extending a few of the architectures in this repository to 3D ViTs, for use with video, medical imaging, etc.

You will need to pass in two additional hyperparameters: (1) the number of frames `frames` and (2) patch size along the frame dimension `frame_patch_size`

For starters, 3D ViT

```python
import torch
from vit_pytorch.vit_3d import ViT

v = ViT(
    image_size = 128,          # image size
    frames = 16,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 2,      # frame patch size
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

video = torch.randn(4, 3, 16, 128, 128) # (batch, channels, frames, height, width)

preds = v(video) # (4, 1000)
```

3D Simple ViT

```python
import torch
from vit_pytorch.simple_vit_3d import SimpleViT

v = SimpleViT(
    image_size = 128,          # image size
    frames = 16,               # number of frames
    image_patch_size = 16,     # image patch size
    frame_patch_size = 2,      # frame patch size
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

video = torch.randn(4, 3, 16, 128, 128) # (batch, channels, frames, height, width)

preds = v(video) # (4, 1000)
```

3D version of <a href="https://github.com/lucidrains/vit-pytorch#cct">CCT</a>

```python
import torch
from vit_pytorch.cct_3d import CCT

cct = CCT(
    img_size = 224,
    num_frames = 8,
    embedding_dim = 384,
    n_conv_layers = 2,
    frame_kernel_size = 3,
    kernel_size = 7,
    stride = 2,
    padding = 3,
    pooling_kernel_size = 3,
    pooling_stride = 2,
    pooling_padding = 1,
    num_layers = 14,
    num_heads = 6,
    mlp_ratio = 3.,
    num_classes = 1000,
    positional_embedding = 'learnable'
)

video = torch.randn(1, 3, 8, 224, 224) # (batch, channels, frames, height, width)
pred = cct(video)
```

## Accessing Attention

If you would like to visualize the attention weights (post-softmax) for your research, just follow the procedure below

```python
import torch
from vit_pytorch.vit import ViT

v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

# import Recorder and wrap the ViT

from vit_pytorch.recorder import Recorder
v = Recorder(v)

# forward pass now returns predictions and the attention maps

img = torch.randn(1, 3, 256, 256)
preds, attns = v(img)

# there is one extra patch due to the CLS token

attns # (1, 6, 16, 65, 65) - (batch x layers x heads x patch x patch)
```

to cleanup the class and the hooks once you have collected enough data

```python
v = v.eject()  # wrapper is discarded and original ViT instance is returned
```
