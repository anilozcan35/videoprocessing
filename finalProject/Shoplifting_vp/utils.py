import os
import numpy as np
import torch
import math

## ------------------------ 3D CNN module ---------------------- ##
def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape


def maxpool3d_output_size(input_shape, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0)):
  
    # Calculates the output shape of a max pooling 3D layer.

    D, H, W = input_shape
    kernel_depth, kernel_height, kernel_width = kernel_size
    pad_depth, pad_height, pad_width = padding
    stride_depth, stride_height, stride_width = stride

    out_depth = int(np.floor((D - kernel_depth + 2 * pad_depth) / stride_depth)) + 1
    out_height = int(np.floor((H - kernel_height + 2 * pad_height) / stride_height)) + 1
    out_width = int(np.floor((W - kernel_width + 2 * pad_width) / stride_width)) + 1

    return (out_depth, out_height, out_width)

def pooling_output_size(input_size, kernel_size, stride, padding=(0, 0, 0)):
    """
    Calculates the output size of a 3D convolution operation.

    Args:
    - input_size (list or tuple): Input size in the format [D, H, W].
    - kernel_size (list or tuple): Kernel size in the format [D, H, W].
    - stride (list or tuple): Stride in the format [D, H, W].
    - padding (list or tuple): Padding in the format [D, H, W].
    - dilation (list or tuple): Dilation in the format [D, H, W].

    Returns:
    - output_size (list): Output size in the format [D_out, H_out, W_out].
    """
    # Unpack input_size, kernel_size, stride, padding, and dilation
    D_in, H_in, W_in = input_size
    D_kernel, H_kernel, W_kernel = kernel_size
    D_stride, H_stride, W_stride = stride
    D_padding, H_padding, W_padding = padding
    # D_dilation, H_dilation, W_dilation = dilation accepeted as 1

    # Calculate output size using the provided formulas
    D_out = math.floor((D_in + 2 * D_padding - (D_kernel - 1) - 1) / D_stride + 1)
    H_out = math.floor((H_in + 2 * H_padding - (H_kernel - 1) - 1) / H_stride + 1)
    W_out = math.floor((W_in + 2 * W_padding - (W_kernel - 1) - 1) / W_stride + 1)

    return (D_out, H_out, W_out)