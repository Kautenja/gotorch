// test cases for clip_boxes_to_image.go
//
// Copyright (c) 2023 Christian Kauten
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

package vision_ops_test

import (
	"testing"
	"github.com/stretchr/testify/assert"
	"github.com/Kautenja/gotorch"
	ops "github.com/Kautenja/gotorch/vision/ops"
)

// ---------------------------------------------------------------------------
// MARK: ClipBoxesToImage_
// ---------------------------------------------------------------------------

func TestClipBoxesToImage_PanicsOn1DInputs(t *testing.T) {
	tensor := torch.NewTensor([]int64{0, 0, 10, 10})
	message := "Expected inputs to be in (N, 4, ...) format, but received tensor with shape [4]"
	assert.PanicsWithValue(t, message, func() {
		ops.ClipBoxesToImage_(tensor, 10, 10)
	})
}

func TestClipBoxesToImage_PanicsOnNonBBoxInputs(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 0, 0}})
	message := "Expected inputs to be in (N, 4, ...) format, but received tensor with shape [1 3]"
	assert.PanicsWithValue(t, message, func() {
		ops.ClipBoxesToImage_(tensor, 10, 10)
	})
}

func TestClipBoxesToImage_IdentityCase(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 0, 10, 10}})
	output := ops.ClipBoxesToImage_(tensor, 100, 100)
	assert.True(t, output.Equal(torch.NewTensor([][]int64{{0, 0, 10, 10}})))
	assert.True(t, output.Equal(tensor))
}

func TestClipBoxesToImage_ClipsBox(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{-10, -20, 200, 200}})
	output := ops.ClipBoxesToImage_(tensor, 50, 100)
	assert.True(t, output.Equal(torch.NewTensor([][]int64{{0, 0, 100, 50}})))
	assert.True(t, output.Equal(tensor))
}

func TestClipBoxesToImage_ClipsBoxes(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 0, 10, 10}, {0, 0, 200, 200}})
	output := ops.ClipBoxesToImage_(tensor, 50, 100)
	expected := torch.NewTensor([][]int64{{0, 0, 10, 10}, {0, 0, 100, 50}})
	assert.True(t, output.Equal(expected))
	assert.True(t, output.Equal(tensor))
}

func TestClipBoxesToImage_ClipsBoxesInSpatialLayout(t *testing.T) {
	a := torch.NewTensor([][]int64{{ 0,  0,  10,  10}})
	b := torch.NewTensor([][]int64{{ 0,  0, 200, 200}})
	c := torch.NewTensor([][]int64{{ 0, -5, 150,  25}})
	d := torch.NewTensor([][]int64{{-5,  0,  25,  75}})
	boxes := torch.Stack([]*torch.Tensor{
		torch.Stack([]*torch.Tensor{a, b}, -1),
		torch.Stack([]*torch.Tensor{c, d}, -1),
	}, -1)
	output := ops.ClipBoxesToImage_(boxes, 50, 100)
	// Create the expected output tensor.
	a = torch.NewTensor([][]int64{{0, 0,  10, 10}})
	b = torch.NewTensor([][]int64{{0, 0, 100, 50}})
	c = torch.NewTensor([][]int64{{0, 0, 100, 25}})
	d = torch.NewTensor([][]int64{{0, 0,  25, 50}})
	expected := torch.Stack([]*torch.Tensor{
		torch.Stack([]*torch.Tensor{a, b}, -1),
		torch.Stack([]*torch.Tensor{c, d}, -1),
	}, -1)
	assert.True(t, output.Equal(expected))
	assert.True(t, output.Equal(boxes))
}

// ---------------------------------------------------------------------------
// MARK: ClipBoxesToImage
// ---------------------------------------------------------------------------

func TestClipBoxesToImagePanicsOn1DInputs(t *testing.T) {
	tensor := torch.NewTensor([]int64{0, 0, 10, 10})
	message := "Expected inputs to be in (N, 4, ...) format, but received tensor with shape [4]"
	assert.PanicsWithValue(t, message, func() {
		ops.ClipBoxesToImage(tensor, 10, 10)
	})
}

func TestClipBoxesToImagePanicsOnNonBBoxInputs(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 0, 0}})
	message := "Expected inputs to be in (N, 4, ...) format, but received tensor with shape [1 3]"
	assert.PanicsWithValue(t, message, func() {
		ops.ClipBoxesToImage(tensor, 10, 10)
	})
}

func TestClipBoxesToImageIdentityCase(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 0, 10, 10}})
	output := ops.ClipBoxesToImage(tensor, 100, 100)
	assert.True(t, output.Equal(torch.NewTensor([][]int64{{0, 0, 10, 10}})))
	assert.True(t, tensor.Equal(torch.NewTensor([][]int64{{0, 0, 10, 10}})))
}

func TestClipBoxesToImageClipsBox(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 0, 200, 200}})
	output := ops.ClipBoxesToImage(tensor, 50, 100)
	assert.True(t, output.Equal(torch.NewTensor([][]int64{{0, 0, 100, 50}})))
	assert.True(t, tensor.Equal(torch.NewTensor([][]int64{{0, 0, 200, 200}})))
}

func TestClipBoxesToImageClipsBoxes(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 0, 10, 10}, {0, 0, 200, 200}})
	output := ops.ClipBoxesToImage(tensor, 50, 100)
	expected := torch.NewTensor([][]int64{{0, 0, 10, 10}, {0, 0, 100, 50}})
	assert.True(t, output.Equal(expected))
	inputs := torch.NewTensor([][]int64{{0, 0, 10, 10}, {0, 0, 200, 200}})
	assert.True(t, tensor.Equal(inputs))
}

func TestClipBoxesToImageClipsBoxesInSpatialLayout(t *testing.T) {
	a := torch.NewTensor([][]int64{{ 0,  0,  10,  10}})
	b := torch.NewTensor([][]int64{{ 0,  0, 200, 200}})
	c := torch.NewTensor([][]int64{{ 0, -5, 150,  25}})
	d := torch.NewTensor([][]int64{{-5,  0,  25,  75}})
	boxes := torch.Stack([]*torch.Tensor{
		torch.Stack([]*torch.Tensor{a, b}, -1),
		torch.Stack([]*torch.Tensor{c, d}, -1),
	}, -1)
	output := ops.ClipBoxesToImage(boxes, 50, 100)
	t.Log(output)
	// Create the expected output tensor.
	a = torch.NewTensor([][]int64{{0, 0,  10, 10}})
	b = torch.NewTensor([][]int64{{0, 0, 100, 50}})
	c = torch.NewTensor([][]int64{{0, 0, 100, 25}})
	d = torch.NewTensor([][]int64{{0, 0,  25, 50}})
	expected := torch.Stack([]*torch.Tensor{
		torch.Stack([]*torch.Tensor{a, b}, -1),
		torch.Stack([]*torch.Tensor{c, d}, -1),
	}, -1)
	t.Log(expected)
	assert.True(t, output.Equal(expected))
	assert.False(t, output.Equal(boxes))
}
