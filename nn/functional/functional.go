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

// #cgo CPPFLAGS: -I/usr/local/include -I/usr/local/include/cgotorch
// #cgo LDFLAGS: -L/usr/local/lib -lc10 -ltorch_cpu -ltorch -lcgotorch
// #include <stdio.h>
// #include <stdlib.h>
// #include "cgotorch/cgotorch.h"
import "C"
import (
	"unsafe"
	"runtime"
	"github.com/Kautenja/gotorch"
	"github.com/Kautenja/gotorch/internal"
)

// Free the C-allocated heap memory associated with the given tensor.
func freeTensor(tensor *torch.Tensor) {
	if tensor.Pointer == nil {
		panic("Attempting to free a tensor that has already been freed!")
	}
	C.Torch_Tensor_Close((C.Tensor)(tensor.Pointer))
	tensor.Pointer = nil
}

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

// Down/up sample the input to either the given size. The algorithm used for
// interpolation is determined by mode. Currently temporal, spatial and
// volumetric sampling are supported, i.e. expected inputs are 3-D, 4-D or 5-D
// in shape. The input dimensions are interpreted in the form:
// mini-batch x channels x [optional depth] x [optional height] x width.
// The modes available for resizing are: nearest, linear (3D-only), bilinear,
// bicubic (4D-only), trilinear (5D-only), area, nearest-exact
func InterpolateSize(
	input *torch.Tensor,
	size []int64,
	mode InterpolateMode,
	alignCorners bool,
	antialias bool,
) (output *torch.Tensor) {
	output = &torch.Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_NN_Functional_InterpolateSize(
		(*C.Tensor)(&output.Pointer),
		(C.Tensor)(input.Pointer),
		(*C.int64_t)(unsafe.Pointer(&size[0])),
		C.int64_t(len(size)),
		C.int64_t(mode),
		C.bool(alignCorners),
		C.bool(antialias),
	)))
	runtime.KeepAlive(input)
	runtime.SetFinalizer(output, freeTensor)
	return
}

// Down/up sample the input to either the given scale factor. The algorithm
// used for interpolation is determined by mode. Currently temporal, spatial
// and volumetric sampling are supported, i.e. expected inputs are 3-D, 4-D or
// 5-D in shape. The input dimensions are interpreted in the form:
// mini-batch x channels x [optional depth] x [optional height] x width.
// The modes available for resizing are: nearest, linear (3D-only), bilinear,
// bicubic (4D-only), trilinear (5D-only), area, nearest-exact
func InterpolateScale(
	input *torch.Tensor,
	scale []float64,
	mode InterpolateMode,
	alignCorners bool,
	antialias bool,
) (output *torch.Tensor) {
	output = &torch.Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_NN_Functional_InterpolateScale(
		(*C.Tensor)(&output.Pointer),
		(C.Tensor)(input.Pointer),
		(*C.double)(unsafe.Pointer(&scale[0])),
		C.int64_t(len(scale)),
		C.int64_t(mode),
		C.bool(alignCorners),
		C.bool(antialias),
	)))
	runtime.KeepAlive(input)
	runtime.SetFinalizer(output, freeTensor)
	return
}

// MARK: torch::nn::functional::kl_div
// MARK: torch::nn::functional::l1_loss
// MARK: torch::nn::functional::layer_norm

// func LeakyRelu(tensor *torch.Tensor, negativeSlope float64) (output *torch.Tensor) {
// 	output = &torch.Tensor{}
// 	internal.PanicOnCException(unsafe.Pointer(C.LeakyRelu((C.Tensor)(tensor.Pointer), C.double(negativeSlope), (*C.Tensor)(&output.Pointer))))
// 	runtime.SetFinalizer(output, freeTensor)
// 	return
// }

// MARK: torch::nn::functional::linear
// MARK: torch::nn::functional::local_response_norm

// func LogSoftmax(tensor *torch.Tensor, dim int64) (output *torch.Tensor) {
//     output = &torch.Tensor{}
//     internal.PanicOnCException(unsafe.Pointer(C.LogSoftmax((C.Tensor)(tensor.Pointer), C.int64_t(dim), (*C.Tensor)(&output.Pointer))))
//     return
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

// Perform L_p normalization of inputs over specified dimension.
func Normalize(
	input *torch.Tensor,
	p float64,
	dim int,
	eps float64,
) (output *torch.Tensor) {
	output = &torch.Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_NN_Functional_Normalize(
		(*C.Tensor)(&output.Pointer),
		(C.Tensor)(input.Pointer),
		C.double(p),
		C.int64_t(dim),
		C.double(eps),
	)))
	runtime.KeepAlive(input)
	runtime.SetFinalizer(output, freeTensor)
	return
}

// MARK: torch::nn::functional::one_hot

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
	input *torch.Tensor,
	padding []int64,
	mode PadMode,
	value ...float64,
) (output *torch.Tensor) {
	output = &torch.Tensor{}
	var value_ float64 = 0
	if len(value) == 1 {
		value_ = value[0]
	} else if len(value) > 1 {
		panic("value should contain 0 or 1 values")
	}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_NN_Functional_Pad(
		(*C.Tensor)(&output.Pointer),
		(C.Tensor)(input.Pointer),
		(*C.int64_t)(unsafe.Pointer(&padding[0])),
		C.int64_t(len(padding)),
		C.int64_t(mode),
		C.double(value_),
	)))
	runtime.KeepAlive(input)
	runtime.SetFinalizer(output, freeTensor)
	return
}

// MARK: torch::nn::functional::pairwise_distance
// MARK: torch::nn::functional::pdist
// MARK: torch::nn::functional::pixel_shuffle
// MARK: torch::nn::functional::pixel_unshuffle
// MARK: torch::nn::functional::poisson_nll_loss
// MARK: torch::nn::functional::prelu

// Apply the rectified linear unit function element-wise.
func Relu(input *torch.Tensor, inplace bool) (output *torch.Tensor) {
	output = &torch.Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_NN_Functional_Relu(
		(*C.Tensor)(&output.Pointer),
		(C.Tensor)(input.Pointer),
		C.bool(inplace),
	)))
	runtime.KeepAlive(input)
	runtime.SetFinalizer(output, freeTensor)
	return
}

// MARK: torch::nn::functional::relu_
// MARK: torch::nn::functional::relu6
// MARK: torch::nn::functional::rrelu
// MARK: torch::nn::functional::selu
// MARK: torch::nn::functional::silu
// MARK: torch::nn::functional::smooth_l1_loss
// MARK: torch::nn::functional::soft_margin_loss

// Apply a softmax function.
func Softmax(input *torch.Tensor, dim int64) (output *torch.Tensor) {
	output = &torch.Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_NN_Functional_Softmax(
		(*C.Tensor)(&output.Pointer),
		(C.Tensor)(input.Pointer),
		C.int64_t(dim),
	)))
	runtime.KeepAlive(input)
	runtime.SetFinalizer(output, freeTensor)
	return
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
