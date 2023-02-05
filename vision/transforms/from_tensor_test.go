// test cases for from_tensor.go
//
// Copyright (c) 2023 Christian Kauten
// Copyright (c) 2022 Sensory, Inc.
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

package vision_transforms_test

import (
	"testing"
	"github.com/stretchr/testify/assert"
	"github.com/Kautenja/gotorch"
	"github.com/Kautenja/gotorch/vision/transforms"
)

func TestFromTensorPanicsOn1DShape(t *testing.T) {
	tensor := torch.Zeros([]int64{128}, torch.NewTensorOptions())
	assert.PanicsWithValue(t, "Expected tensor to be 3-dimensional but tensor has shape [128]", func() {
		vision_transforms.FromTensor().Forward(tensor)
	})
}

func TestFromTensorPanicsOn2DShape(t *testing.T) {
	tensor := torch.Zeros([]int64{128, 64}, torch.NewTensorOptions())
	assert.PanicsWithValue(t, "Expected tensor to be 3-dimensional but tensor has shape [128 64]", func() {
		vision_transforms.FromTensor().Forward(tensor)
	})
}

func TestFromTensorPanicsOn4DShape(t *testing.T) {
	tensor := torch.Zeros([]int64{1, 3, 128, 64}, torch.NewTensorOptions())
	assert.PanicsWithValue(t, "Expected tensor to be 3-dimensional but tensor has shape [1 3 128 64]", func() {
		vision_transforms.FromTensor().Forward(tensor)
	})
}

func TestFromTensorPanicsOn2Channels(t *testing.T) {
	tensor := torch.Zeros([]int64{2, 128, 64}, torch.NewTensorOptions())
	assert.PanicsWithValue(t, "Expected tensor to have 1, 3, or 4 channels, but found 2", func() {
		vision_transforms.FromTensor().Forward(tensor)
	})
}

func TestFromTensorPanicsOn5Channels(t *testing.T) {
	tensor := torch.Zeros([]int64{5, 128, 64}, torch.NewTensorOptions())
	assert.PanicsWithValue(t, "Expected tensor to have 1, 3, or 4 channels, but found 5", func() {
		vision_transforms.FromTensor().Forward(tensor)
	})
}

func TestFromTensorConvertsTensorWith4ChannelsToImage(t *testing.T) {
	height := int64(4)
	width := int64(5)
	tensor := torch.Rand([]int64{4, height, width}, torch.NewTensorOptions())
	image := vision_transforms.FromTensor().Forward(tensor)
	// Check that the bounds match the tensor shape.
	window := image.Bounds()
	assert.Equal(t, 0, window.Min.X)
	assert.Equal(t, 0, window.Min.Y)
	assert.Equal(t, int(width), window.Max.X)
	assert.Equal(t, int(height), window.Max.Y)
	// Convert the image back to a tensor
	retensor := vision_transforms.ToTensor().Forward(image)
	// Check for approximate equivalence with a single decimal point of
	// precision. The operation is lossy so a single decimal point is necessary.
	assert.True(t, torch.AllClose(tensor.Slice(0, 0, 3, 1), retensor, 1e-8, 1e-1),
		"Got %v, expected %v", retensor, tensor.Slice(0, 0, 3, 1))
}

func TestFromTensorConvertsTensorWith3ChannelsToImage(t *testing.T) {
	height := int64(4)
	width := int64(5)
	tensor := torch.Rand([]int64{3, height, width}, torch.NewTensorOptions())
	image := vision_transforms.FromTensor().Forward(tensor)
	// Check that the bounds match the tensor shape.
	window := image.Bounds()
	assert.Equal(t, 0, window.Min.X)
	assert.Equal(t, 0, window.Min.Y)
	assert.Equal(t, int(width), window.Max.X)
	assert.Equal(t, int(height), window.Max.Y)
	// Convert the image back to a tensor
	retensor := vision_transforms.ToTensor().Forward(image)
	// Check for approximate equivalence with a single decimal point of
	// precision. The operation is lossy so a single decimal point is necessary.
	assert.True(t, torch.AllClose(tensor, retensor, 1e-8, 1e-1),
		"Got %v, expected %v", retensor, tensor)
}

func TestFromTensorConvertsTensorWith1ChannelToImage(t *testing.T) {
	height := int64(4)
	width := int64(5)
	tensor := torch.Rand([]int64{1, height, width}, torch.NewTensorOptions())
	image := vision_transforms.FromTensor().Forward(tensor)
	// Check that the bounds match the tensor shape.
	window := image.Bounds()
	assert.Equal(t, 0, window.Min.X)
	assert.Equal(t, 0, window.Min.Y)
	assert.Equal(t, int(width), window.Max.X)
	assert.Equal(t, int(height), window.Max.Y)
	// Convert the image back to a tensor
	retensor := vision_transforms.ToTensor().Forward(image)
	// Check for approximate equivalence with a single decimal point of
	// precision. The operation is lossy so a single decimal point is necessary.
	assert.True(t, torch.AllClose(tensor, retensor.Slice(0, 0, 1, 1), 1e-8, 1e-1),
		"[Red] Got %v, expected %v", retensor.Slice(0, 0, 1, 1), tensor)
	assert.True(t, torch.AllClose(tensor, retensor.Slice(0, 1, 2, 1), 1e-8, 1e-1),
		"[Green] Got %v, expected %v", retensor.Slice(0, 1, 2, 1), tensor)
	assert.True(t, torch.AllClose(tensor, retensor.Slice(0, 2, 3, 1), 1e-8, 1e-1),
		"[Blue] Got %v, expected %v", retensor.Slice(0, 2, 3, 1), tensor)
}
