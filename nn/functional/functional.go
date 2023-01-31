// Go bindings for torch::nn::functional.
//
// Copyright (c) 2023 Christian Kauten
// Copyright (c) 2022 Sensory, Inc.
// Copyright (c) 2020 GoTorch Authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXTERNRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

package nn_functional

// #cgo CFLAGS: -I ${SRCDIR}/../..
// #cgo LDFLAGS: -L ${SRCDIR}/../../build -Wl,-rpath ${SRCDIR}/../../build -lcgotorch
// #include <stdio.h>
// #include <stdlib.h>
// #include "cgotorch/cgotorch.h"
import "C"
import (
    "unsafe"
    "github.com/Kautenja/gotorch"
    internal "github.com/Kautenja/gotorch/internal"
)

// MARK: torch::nn::functional::adaptive_avg_pool1d
// MARK: torch::nn::functional::adaptive_avg_pool2d
// MARK: torch::nn::functional::adaptive_avg_pool3d
// MARK: torch::nn::functional::adaptive_max_pool1d
// MARK: torch::nn::functional::adaptive_max_pool2d
// MARK: torch::nn::functional::adaptive_max_pool2d_with_indices
// MARK: torch::nn::functional::adaptive_max_pool3d
// MARK: torch::nn::functional::adaptive_max_pool3d_with_indices
// MARK: torch::nn::functional::affine_grid
// MARK: torch::nn::functional::alpha_dropout
// MARK: torch::nn::functional::avg_pool1d
// MARK: torch::nn::functional::avg_pool2d
// MARK: torch::nn::functional::avg_pool3d
// MARK: torch::nn::functional::batch_norm
// MARK: torch::nn::functional::bilinear
// MARK: torch::nn::functional::binary_cross_entropy
// MARK: torch::nn::functional::binary_cross_entropy_with_logits
// MARK: torch::nn::functional::celu
// MARK: torch::nn::functional::conv1d
// MARK: torch::nn::functional::conv2d
// MARK: torch::nn::functional::conv3d
// MARK: torch::nn::functional::conv_transpose1d
// MARK: torch::nn::functional::conv_transpose2d
// MARK: torch::nn::functional::conv_transpose3d
// MARK: torch::nn::functional::cosine_embedding_loss
// MARK: torch::nn::functional::cosine_similarity
// MARK: torch::nn::functional::cross_entropy
// MARK: torch::nn::functional::ctc_loss
// MARK: torch::nn::functional::dropout
// MARK: torch::nn::functional::dropout2d
// MARK: torch::nn::functional::dropout3d
// MARK: torch::nn::functional::elu
// MARK: torch::nn::functional::embedding
// MARK: torch::nn::functional::embedding_bag
// MARK: torch::nn::functional::feature_alpha_dropout
// MARK: torch::nn::functional::fold
// MARK: torch::nn::functional::fractional_max_pool2d
// MARK: torch::nn::functional::fractional_max_pool2d_with_indices
// MARK: torch::nn::functional::fractional_max_pool3d
// MARK: torch::nn::functional::fractional_max_pool3d_with_indices
// MARK: torch::nn::functional::gelu
// MARK: torch::nn::functional::glu
// MARK: torch::nn::functional::grid_sample
// MARK: torch::nn::functional::group_norm
// MARK: torch::nn::functional::gumbel_softmax
// MARK: torch::nn::functional::hardshrink
// MARK: torch::nn::functional::hardtanh
// MARK: torch::nn::functional::hinge_embedding_loss
// MARK: torch::nn::functional::huber_loss
// MARK: torch::nn::functional::instance_norm

// MARK: torch::nn::functional::interpolate

// Interpolation algorithms implemented by libtorch.
type InterpolateMode int64
const (
    InterpolateNearest InterpolateMode = iota
    InterpolateLinear
    InterpolateBilinear
    InterpolateBicubic
    InterpolateTrilinear
    InterpolateArea
    InterpolateNearestExact
)

// Interpolate a tensor by size.
func InterpolateSize(
    input torch.Tensor,
    size []int64,
    mode InterpolateMode,
    alignCorners bool,
    antialias bool,
) torch.Tensor {
    var output C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_NN_Functional_InterpolateSize(&output,
        C.Tensor(*input.T),
        (*C.int64_t)(unsafe.Pointer(&size[0])),
        C.int64_t(len(size)),
        C.int64_t(mode),
        C.bool(alignCorners),
        C.bool(antialias),
    )))
    return torch.NewTorchTensor((*unsafe.Pointer)(&output))
}

// Interpolate a tensor by scale.
func InterpolateScale(
    input torch.Tensor,
    scale []float64,
    mode InterpolateMode,
    alignCorners bool,
    antialias bool,
) torch.Tensor {
    var output C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_NN_Functional_InterpolateScale(&output,
        C.Tensor(*input.T),
        (*C.double)(unsafe.Pointer(&scale[0])),
        C.int64_t(len(scale)),
        C.int64_t(mode),
        C.bool(alignCorners),
        C.bool(antialias),
    )))
    return torch.NewTorchTensor((*unsafe.Pointer)(&output))
}

// MARK: torch::nn::functional::kl_div
// MARK: torch::nn::functional::l1_loss
// MARK: torch::nn::functional::layer_norm

// func LeakyRelu(tensor torch.Tensor, negativeSlope float64) torch.Tensor {
//     var output C.Tensor
//     internal.PanicOnCException(unsafe.Pointer(C.LeakyRelu(C.Tensor(*tensor.T), C.double(negativeSlope), &output)))
//     return torch.NewTorchTensor((*unsafe.Pointer)(&output))
// }

// MARK: torch::nn::functional::linear
// MARK: torch::nn::functional::local_response_norm

// func LogSoftmax(tensor torch.Tensor, dim int64) torch.Tensor {
//     var output C.Tensor
//     internal.PanicOnCException(unsafe.Pointer(C.LogSoftmax(C.Tensor(*tensor.T), C.int64_t(dim), &output)))
//     return torch.NewTorchTensor((*unsafe.Pointer)(&output))
// }

// MARK: torch::nn::functional::logsigmoid
// MARK: torch::nn::functional::lp_pool1d
// MARK: torch::nn::functional::lp_pool2d
// MARK: torch::nn::functional::margin_ranking_loss
// MARK: torch::nn::functional::max_pool1d
// MARK: torch::nn::functional::max_pool1d_with_indices
// MARK: torch::nn::functional::max_pool2d
// MARK: torch::nn::functional::max_pool2d_with_indices
// MARK: torch::nn::functional::max_pool3d
// MARK: torch::nn::functional::max_pool3d_with_indices
// MARK: torch::nn::functional::max_unpool1d
// MARK: torch::nn::functional::max_unpool2d
// MARK: torch::nn::functional::max_unpool3d
// MARK: torch::nn::functional::mish
// MARK: torch::nn::functional::mse_loss
// MARK: torch::nn::functional::multi_head_attention_forward
// MARK: torch::nn::functional::multi_margin_loss
// MARK: torch::nn::functional::multilabel_margin_loss
// MARK: torch::nn::functional::multilabel_soft_margin_loss
// MARK: torch::nn::functional::nll_loss

// MARK: torch::nn::functional::normalize

func Normalize(input torch.Tensor, p float64, dim int, eps float64) torch.Tensor {
    var output C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_NN_Functional_Normalize(
        &output,
        C.Tensor(*input.T),
        C.double(p),
        C.int64_t(dim),
        C.double(eps))))
    // runtime.KeepAlive(input.T)
    return torch.NewTorchTensor((*unsafe.Pointer)(&output))
}

// MARK: torch::nn::functional::one_hot
// MARK: torch::nn::functional::pad

// Padding algorithms implemented by libtorch.
type PadMode int64
const (
    PadConstant PadMode = iota
    PadReflect
    PadReplicate
    PadCircular
)

// Pad a tensor.
func Pad(
    input torch.Tensor,
    padding []int64,
    mode PadMode,
    value ...float64,
) torch.Tensor {
    var output C.Tensor
    var value_ float64 = 0
    if len(value) == 1 {
        value_ = value[0]
    } else if len(value) > 1 {
        panic("value should contain 0 or 1 values")
    }
    internal.PanicOnCException(unsafe.Pointer(C.Torch_NN_Functional_Pad(&output,
        C.Tensor(*input.T),
        (*C.int64_t)(unsafe.Pointer(&padding[0])),
        C.int64_t(len(padding)),
        C.int64_t(mode),
        C.double(value_),
    )))
    return torch.NewTorchTensor((*unsafe.Pointer)(&output))
}

// MARK: torch::nn::functional::pairwise_distance
// MARK: torch::nn::functional::pdist
// MARK: torch::nn::functional::pixel_shuffle
// MARK: torch::nn::functional::pixel_unshuffle
// MARK: torch::nn::functional::poisson_nll_loss
// MARK: torch::nn::functional::prelu

func Relu(tensor torch.Tensor, inplace bool) torch.Tensor {
    var output C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_NN_Functional_Relu(
        &output,
        C.Tensor(*tensor.T),
        C.bool(inplace),
    )))
    return torch.NewTorchTensor((*unsafe.Pointer)(&output))
}

// MARK: torch::nn::functional::relu6
// MARK: torch::nn::functional::rrelu
// MARK: torch::nn::functional::selu
// MARK: torch::nn::functional::silu
// MARK: torch::nn::functional::smooth_l1_loss
// MARK: torch::nn::functional::soft_margin_loss

func Softmax(tensor torch.Tensor, dim int64) torch.Tensor {
    var output C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_NN_Functional_Softmax(
        &output,
        C.Tensor(*tensor.T),
        C.int64_t(dim),
    )))
    return torch.NewTorchTensor((*unsafe.Pointer)(&output))
}

// MARK: torch::nn::functional::softmin
// MARK: torch::nn::functional::softplus
// MARK: torch::nn::functional::softshrink
// MARK: torch::nn::functional::softsign
// MARK: torch::nn::functional::tanhshrink
// MARK: torch::nn::functional::threshold
// MARK: torch::nn::functional::triplet_margin_loss
// MARK: torch::nn::functional::triplet_margin_with_distance_loss
// MARK: torch::nn::functional::unfold
