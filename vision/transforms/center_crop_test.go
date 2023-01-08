// test cases for center_crop.go
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

package vision_transforms_test

import (
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/Kautenja/gotorch"
    "github.com/Kautenja/gotorch/vision/transforms"
)

func TestCenterCrop(t *testing.T) {
    tensor := torch.NewTensor([][]float32{
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 8, 7},
        {6, 5, 4, 3},
    })
    expected := torch.NewTensor([][]float32{
        {5, 6},
        {9, 8},
    })
    transform := vision_transforms.CenterCrop(2, 2)
    assert.True(t, transform.Forward(tensor).Equal(expected))
}

func TestCenterCropTransformPanicsOnZeroHeight(t *testing.T) {
    assert.PanicsWithValue(t, "height should be greater than 0", func() {
        vision_transforms.CenterCrop(0, 224)
    })
}

func TestCenterCropTransformPanicsOnNegativeHeight(t *testing.T) {
    assert.PanicsWithValue(t, "height should be greater than 0", func() {
        vision_transforms.CenterCrop(-1, 224)
    })
}

func TestCenterCropTransformPanicsOnZeroWidth(t *testing.T) {
    assert.PanicsWithValue(t, "width should be greater than 0", func() {
        vision_transforms.CenterCrop(224, 0)
    })
}

func TestCenterCropTransformPanicsOnNegativeWidth(t *testing.T) {
    assert.PanicsWithValue(t, "width should be greater than 0", func() {
        vision_transforms.CenterCrop(224, -1)
    })
}

func TestCenterCropTransformPanicsOnOneDimensionalInputs(t *testing.T) {
    assert.PanicsWithValue(t, "CenterCrop only supports tensors with 2 or more dimensions", func() {
        vision_transforms.CenterCrop(224, 224).Forward(torch.Zeros([]int64{5}, torch.NewTensorOptions()))
    })
}
