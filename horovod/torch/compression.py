# Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Gradient compression algorithms."""

import torch


class Compressor(object):
    """Interface for compressing and decompressing a given tensor."""
    @staticmethod
    def compress(tensor):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        pass

    @staticmethod
    def decompress(tensor, ctx):
        """Decompress the tensor with the given context."""
        pass


class NoneCompressor(Compressor):
    """Default no-op compression."""
    @staticmethod
    def compress(tensor):
        """Returns the tensor unmodified."""
        return tensor, None

    @staticmethod
    def decompress(tensor, ctx):
        """Returns the tensor unmodified."""
        return tensor


class FP16Compressor(Compressor):
    """Compress all floating point gradients to 16-bit."""
    @staticmethod
    def compress(tensor):
        """Downcasts the tensor to 16-bit."""
        tensor_compressed = tensor
        if tensor.dtype.is_floating_point:
            # Only allow compression from other floating point types
            tensor_compressed = tensor.type(torch.float16)
        return tensor_compressed, tensor.dtype

    @staticmethod
    def decompress(tensor, ctx):
        """Upcasts the tensor to the initialization dtype."""
        tensor_decompressed = tensor
        dtype = ctx
        if dtype.is_floating_point:
            tensor_decompressed = tensor.type(dtype)
        return tensor_decompressed

class SparCompressor(Compressor):
    """Compress only 30% numbers of gradients."""
    @staticmethod
    def compress(tensor):
        """Copy the device number and element type of input tensor."""
        device_number = tensor.device
        element_type = tensor.dtype
        """Pseudo-random number generator."""
        random_seed = 19
        torch.manual_seed(random_seed)
        """Form 30% selected the indices(mask)."""
        rand = torch.rand(tensor.shape, device=device_number, dtype=element_type)
        threshold = 0.3
        bool_mask = rand < torch.Tensor(threshold, device=device_number, dtype=element_type)
        """New tensor with inherit values."""
        tensor_compressed = torch.masked_select(tensor, bool_mask)
        """ctx for info in original shape and select indcies."""
        # ctx = bool_mask
        return tensor_compressed, bool_mask
    
    @staticmethod
    def decompress(tensor, ctx):
        """decompress tensor to input shape."""

        mask = ctx.view(-1)
        tensor_decompressed = torch.zeros_like(ctx).type(tensor.dtype).to(tensor.device)
        tensor_decompressed.view(-1)[mask] = tensor
        return tensor_decompressed

class Compression(object):
    """Optional gradient compression algorithm used during allreduce."""

    """Do not compress the gradients. This is the default."""
    none = NoneCompressor

    """Compress all floating point gradients to 16-bit."""
    fp16 = FP16Compressor
    
    """Compress only 30% numbers of gradients."""
    spar = SparCompressor
