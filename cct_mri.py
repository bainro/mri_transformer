import torch
import numpy as np
from vit_pytorch.cct_3d import CCT

cct = CCT(
    img_size = 128,
    num_frames = 128,
    embedding_dim = 384,
    n_conv_layers = 2, # increase after things are working!
    n_input_channels=1,
    frame_kernel_size = 7, # decrease after things are working
    kernel_size = 7, # decrease after things are working
    stride = 2,
    frame_stride = 2,
    padding = 3,
    frame_padding= 3,
    pooling_kernel_size = 3,
    frame_pooling_kernel_size = 3,
    pooling_stride = 2,
    frame_pooling_stride = 2,
    pooling_padding = 1,
    frame_pooling_padding = 1,
    num_layers = 10, # default was 14
    num_heads = 6, # parallel attn fxs. Same input can go thru different W_k & W_q's.
    mlp_ratio = 2., # how many neurons in a Transformer block's FC layers. Bigger = more neurons. 
    num_classes = 1, # we're going to regress (e.g. white matter) & use MSE loss
    positional_embedding = 'sine'
)

# shape: (1, 128, 128, 128)
t1 = np.load("./subjects/0000_T1.npy")
# add the batch dimension: (1, 1, 128, 128, 128)
t1 = np.expand_dims(t1, axis=0)
# convert from np array to torch tensor of dtype float32
t1 = torch.from_numpy(t1).to(torch.float32)

pred = cct(t1)