// test cases for functional.go
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

package nn_functional_test

import (
	"unsafe"
	"testing"
	"github.com/stretchr/testify/assert"
	"github.com/Kautenja/gotorch"
	F "github.com/Kautenja/gotorch/nn/functional"
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

func TestInterpolateSize(t *testing.T) {
	tensor := torch.Zeros([]int64{1, 3, 256, 256}, torch.NewTensorOptions())
	output := F.InterpolateSize(tensor, []int64{128, 64}, F.InterpolateNearest, false, false)
	assert.NotNil(t, output.T)
	assert.Equal(t, []int64{1, 3, 128, 64}, output.Shape())
	// The input tensor should not be changed
	assert.Equal(t, []int64{1, 3, 256, 256}, tensor.Shape())
}

func TestInterpolateScale(t *testing.T) {
	tensor := torch.Zeros([]int64{1, 3, 256, 256}, torch.NewTensorOptions())
	output := F.InterpolateScale(tensor, []float64{0.5, 0.25}, F.InterpolateNearest, false, false)
	assert.NotNil(t, output.T)
	assert.Equal(t, []int64{1, 3, 128, 64}, output.Shape())
	// The input tensor should not be changed
	assert.Equal(t, []int64{1, 3, 256, 256}, tensor.Shape())
}

// MARK: torch::nn::functional::kl_div
// MARK: torch::nn::functional::l1_loss
// MARK: torch::nn::functional::layer_norm

// >>> torch.nn.functional.leaky_relu(torch.tensor([[-0.5, -1.], [1., 0.5]]))
// tensor([[-0.0050, -0.0100],
//         [ 1.0000,  0.5000]])
// func TestLeakyRelu(t *testing.T) {
//  tensor := torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}})
//  r := torch.LeakyRelu(tensor, 0.01)
//  g := "-0.0050 -0.0100\n 1.0000  0.5000\n[ CPUFloatType{2,2} ]"
//  assert.Equal(t, g, r.String())
// }

// MARK: torch::nn::functional::linear
// MARK: torch::nn::functional::local_response_norm

// >>> torch.nn.functional.log_softmax(torch.tensor([[-0.5, -1.], [1., 0.5]]), dim=1)
// tensor([[-0.4741, -0.9741],
//         [-0.4741, -0.9741]])
// func TestLogSoftmax(t *testing.T) {
//  tensor := torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}})
//  r := torch.LogSoftmax(tensor, 1)
//  g := "-0.4741 -0.9741\n-0.4741 -0.9741\n[ CPUFloatType{2,2} ]"
//  assert.Equal(t, g, r.String())
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

func TestNormalize(t *testing.T) {
	data := [2][3]float32{{1.0, 1.1, 1.2}, {2, 3, 4}}
	tensor := torch.TensorFromBlob(unsafe.Pointer(&data), torch.Float, []int64{2, 3})
	output := F.Normalize(tensor, 2.0, 1, 1e-12)
	assert.InEpsilon(t, 2.0, output.Square().Sum().Item().(float32), 1e-6)
}

// MARK: torch::nn::functional::one_hot

// MARK: torch::nn::functional::pad

// >>> a = torch.tensor([[0, 1], [2, 3]])
// >>> torch.nn.functional.pad(a, (0, 1, 0, 1))
// tensor([[0, 1, 0],
//         [2, 3, 0],
//         [0, 0, 0]])
func TestPad(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 1}, {2, 3}})
	output := F.Pad(tensor, []int64{0, 1, 0, 1}, F.PadConstant)
	assert.NotNil(t, output.T)
	expected := torch.NewTensor([][]int64{{0, 1, 0}, {2, 3, 0}, {0, 0, 0}})
	assert.True(t, torch.AllClose(output, expected, 1e-8, 1e-3), "Got %v, expected %v", output, expected)
}

// >>> a = torch.tensor([[0, 1], [2, 3]])
// >>> torch.nn.functional.pad(a, (0, 1, 0, 1), value=1)
// tensor([[0, 1, 1],
//         [2, 3, 1],
//         [1, 1, 1]])
func TestPadWithConstantValue(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 1}, {2, 3}})
	output := F.Pad(tensor, []int64{0, 1, 0, 1}, F.PadConstant, 1)
	assert.NotNil(t, output.T)
	expected := torch.NewTensor([][]int64{{0, 1, 1}, {2, 3, 1}, {1, 1, 1}})
	assert.True(t, torch.AllClose(output, expected, 1e-8, 1e-3), "Got %v, expected %v", output, expected)
}

func TestPadPanicsOnPaddingValueLengthGreaterThan1(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 1}, {2, 3}})
	assert.PanicsWithValue(t, "value should contain 0 or 1 values", func() {
		F.Pad(tensor, []int64{0, 0, 0, 0}, F.PadConstant, 1, 1)
	})
}

// >>> a = torch.tensor([[0, 1], [2, 3]])
// >>> torch.nn.functional.pad(a, (0, 1, 0, 1, 1))
// Traceback (most recent call last):
//   File "<stdin>", line 1, in <module>
// RuntimeError: Padding length must be divisible by 2
func TestPadPanicsOnPaddingLengthNotDivisbleBy2(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 1}, {2, 3}})
	assert.PanicsWithValue(t, "Padding length must be divisible by 2", func() {
		F.Pad(tensor, []int64{0, 1, 0, 1, 1}, F.PadConstant)
	})
}

// >>> a = torch.tensor([[0, 1], [2, 3]])
// >>> torch.nn.functional.pad(a, (0, 1, 0, 1, 1, 1))
// Traceback (most recent call last):
//   File "<stdin>", line 1, in <module>
// RuntimeError: Padding length too large
func TestPadPanicsOnPaddingLengthTooLarge(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 1}, {2, 3}})
	assert.PanicsWithValue(t, "Padding length too large", func() {
		F.Pad(tensor, []int64{0, 1, 0, 1, 1, 1}, F.PadConstant)
	})
}

// MARK: torch::nn::functional::pairwise_distance
// MARK: torch::nn::functional::pdist
// MARK: torch::nn::functional::pixel_shuffle
// MARK: torch::nn::functional::pixel_unshuffle
// MARK: torch::nn::functional::poisson_nll_loss
// MARK: torch::nn::functional::prelu

// >>> torch.nn.functional.relu(torch.tensor([[-0.5, -1.], [1., 0.5]]))
// tensor([[0.0000, 0.0000],
//         [1.0000, 0.5000]])
func TestRelu(t *testing.T) {
	tensor := torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}})
	r := F.Relu(tensor, false)
	g := " 0.0000  0.0000\n 1.0000  0.5000\n[ CPUFloatType{2,2} ]"
	assert.Equal(t, g, r.String())
}

// MARK: torch::nn::functional::relu6
// MARK: torch::nn::functional::rrelu
// MARK: torch::nn::functional::selu
// MARK: torch::nn::functional::silu
// MARK: torch::nn::functional::smooth_l1_loss
// MARK: torch::nn::functional::soft_margin_loss

// >>> torch.nn.functional.softmax(torch.eye(2).float(), -1)
// tensor([[0.7311, 0.2689],
//         [0.2689, 0.7311]])
func TestSoftmax(t *testing.T) {
	tensor := torch.NewTensor([][]float32{{1.0, 0.0}, {0.0, 1.0}})
	output := F.Softmax(tensor, -1)
	expected := torch.NewTensor([][]float32{{0.7311, 0.2689}, {0.2689, 0.7311}})
	assert.True(t, torch.AllClose(output, expected, 1e-8, 1e-3), "Got %v, expected %v", output, expected)
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
