// test cases for resize.go
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
    F "github.com/Kautenja/gotorch/nn/functional"
    "github.com/Kautenja/gotorch/vision/transforms"
)

// >>> a = torch.tensor([[0, 1, 2, 3],[4, 5, 6, 7],[8, 9, 8, 7],[6, 5, 4, 3]]).float()
// >>> F.interpolate(a.view(1, 1, 4, 4), (2, 2), mode='bilinear', align_corners=False)
// tensor([[[[2.5000, 4.5000],
//           [7.0000, 5.5000]]]])
func TestResizeWithoutAlignCorners(t *testing.T) {
    tensor := torch.NewTensor([][]float32{
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 8, 7},
        {6, 5, 4, 3},
    })
    expected := torch.NewTensor([][]float32{
        {2.5000, 4.5000},
        {7.0000, 5.5000},
    })
    output := vision_transforms.Resize(2, 2, F.InterpolateBilinear, false, false).Forward(tensor.View(1, 1, 4, 4))
    assert.True(t, output.Equal(expected.View(1, 1, 2, 2)), "Got %v, expected %v", output, expected)
}

// >>> a = torch.tensor([[0, 1, 2, 3],[4, 5, 6, 7],[8, 9, 8, 7],[6, 5, 4, 3]]).float()
// >>> F.interpolate(a.view(1, 1, 4, 4), (2, 2), mode='bilinear', align_corners=True)
// tensor([[[[0., 3.],
//           [6., 3.]]]])
func TestResizeWithAlignedCorners(t *testing.T) {
    tensor := torch.NewTensor([][]float32{
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 8, 7},
        {6, 5, 4, 3},
    })
    expected := torch.NewTensor([][]float32{
        {0, 3},
        {6, 3},
    })
    output := vision_transforms.Resize(2, 2, F.InterpolateBilinear, true, false).Forward(tensor.View(1, 1, 4, 4))
    assert.True(t, output.Equal(expected.View(1, 1, 2, 2)), "Got %v, expected %v", output, expected)
}

// >>> a = torch.tensor([[0, 1, 2, 3],[4, 5, 6, 7],[8, 9, 8, 7],[6, 5, 4, 3]]).float()
// >>> F.interpolate(a.view(1, 1, 4, 4), (2, 2), mode='bilinear', align_corners=False, antialias=True)
// tensor([[[[3.5306, 4.7755],
//           [6.5510, 5.7959]]]])
func TestResizeWithAntialiasing(t *testing.T) {
    tensor := torch.NewTensor([][]float32{
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 8, 7},
        {6, 5, 4, 3},
    })
    expected := torch.NewTensor([][]float32{
        {3.5306, 4.7755},
        {6.5510, 5.7959},
    })
    output := vision_transforms.Resize(2, 2, F.InterpolateBilinear, false, true).Forward(tensor.View(1, 1, 4, 4))
    assert.True(t, output.AllClose(expected.View(1, 1, 2, 2), 1e-5, 1e-8), "Got %v, expected %v", output, expected)
}

func TestResizeTransformPanicsOnZeroHeight(t *testing.T) {
    assert.PanicsWithValue(t, "height should be greater than 0", func() {
        vision_transforms.Resize(0, 224, F.InterpolateNearest, false, false)
    })
}

func TestResizeTransformPanicsOnNegativeHeight(t *testing.T) {
    assert.PanicsWithValue(t, "height should be greater than 0", func() {
        vision_transforms.Resize(-1, 224, F.InterpolateNearest, false, false)
    })
}

func TestResizeTransformPanicsOnZeroWidth(t *testing.T) {
    assert.PanicsWithValue(t, "width should be greater than 0", func() {
        vision_transforms.Resize(224, 0, F.InterpolateNearest, false, false)
    })
}

func TestResizeTransformPanicsOnNegativeWidth(t *testing.T) {
    assert.PanicsWithValue(t, "width should be greater than 0", func() {
        vision_transforms.Resize(224, -1, F.InterpolateNearest, false, false)
    })
}

func TestResizeTransformPanicsOnOneDimensionalInputs(t *testing.T) {
    transform := vision_transforms.Resize(224, 224, F.InterpolateNearest, false, false)
    assert.Panics(t, func() { _  = transform.Forward(torch.Zeros([]int64{5}, torch.NewTensorOptions())) })
}
