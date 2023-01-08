// test cases for to_tensor.go
//
// Copyright (c) 2022 Christian Kauten
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

package vision_transforms_functional_test

import (
    "testing"
    "image"
    "image/color"
    "github.com/stretchr/testify/assert"
    "github.com/Kautenja/gotorch"
    "github.com/Kautenja/gotorch/vision/transforms/functional"
)

func TestToTensorRGBAImage(t *testing.T) {
    height := int(3)
    width := int(4)
    window := image.Rect(0, 0, width, height)
    // Create the image and assign data to the corners for checking the tensor.
    image := image.NewRGBA(window)
    image.Set(width - 1, 0,          color.RGBA{127, 127, 127, 255})
    image.Set(width - 1, height - 1, color.RGBA{255, 255, 255, 255})
    // Convert the image to a tensor.
    tensor := vision_transforms_functional.ToTensor(image)
    assert.Equal(t, tensor.Dim(), int64(3))
    assert.Equal(t, tensor.Shape(), []int64{3, int64(height), int64(width)})
    expected := torch.NewTensor([][][]float32{
        {
            {0, 0, 0, 0.498},
            {0, 0, 0, 0},
            {0, 0, 0, 1.0},
        },
        {
            {0, 0, 0, 0.498},
            {0, 0, 0, 0},
            {0, 0, 0, 1.0},
        },
        {
            {0, 0, 0, 0.498},
            {0, 0, 0, 0},
            {0, 0, 0, 1.0},
        },
    })
    assert.True(t, torch.AllClose(expected, tensor, 1e-5, 1e-3), "Got %v, expected %v", tensor, expected)
}

func TestToTensorNRGBAImage(t *testing.T) {
    height := int(3)
    width := int(4)
    window := image.Rect(0, 0, width, height)
    // Create the image and assign data to the corners for checking the tensor.
    image := image.NewNRGBA(window)
    image.Set(width - 1, 0,          color.RGBA{127, 127, 127, 255})
    image.Set(width - 1, height - 1, color.RGBA{255, 255, 255, 255})
    // Convert the image to a tensor.
    tensor := vision_transforms_functional.ToTensor(image)
    assert.Equal(t, tensor.Dim(), int64(3))
    assert.Equal(t, tensor.Shape(), []int64{3, int64(height), int64(width)})
    expected := torch.NewTensor([][][]float32{
        {
            {0, 0, 0, 0.498},
            {0, 0, 0, 0},
            {0, 0, 0, 1.0},
        },
        {
            {0, 0, 0, 0.498},
            {0, 0, 0, 0},
            {0, 0, 0, 1.0},
        },
        {
            {0, 0, 0, 0.498},
            {0, 0, 0, 0},
            {0, 0, 0, 1.0},
        },
    })
    assert.True(t, torch.AllClose(expected, tensor, 1e-5, 1e-3), "Got %v, expected %v", tensor, expected)
}

func TestToTensorGrayImage(t *testing.T) {
    height := int(3)
    width := int(4)
    window := image.Rect(0, 0, width, height)
    // Create the image and assign data to the corners for checking the tensor.
    image := image.NewGray(window)
    image.Set(width - 1, 0,          color.RGBA{127, 127, 127, 255})
    image.Set(width - 1, height - 1, color.RGBA{255, 255, 255, 255})
    // Convert the image to a tensor.
    tensor := vision_transforms_functional.ToTensor(image)
    assert.Equal(t, tensor.Dim(), int64(3))
    assert.Equal(t, tensor.Shape(), []int64{3, int64(height), int64(width)})
    expected := torch.NewTensor([][][]float32{
        {
            {0, 0, 0, 0.498},
            {0, 0, 0, 0},
            {0, 0, 0, 1.0},
        },
        {
            {0, 0, 0, 0.498},
            {0, 0, 0, 0},
            {0, 0, 0, 1.0},
        },
        {
            {0, 0, 0, 0.498},
            {0, 0, 0, 0},
            {0, 0, 0, 1.0},
        },
    })
    assert.True(t, torch.AllClose(expected, tensor, 1e-5, 1e-3), "Got %v, expected %v", tensor, expected)
}

func TestToTensorRaisesErrorOnUniformImage(t *testing.T) {
    image := image.NewUniform(color.RGBA{1, 1, 1, 255})
    assert.PanicsWithValue(t, "ToTensor not implemented for image of type Uniform", func() {
        _ = vision_transforms_functional.ToTensor(image)
    })
}
