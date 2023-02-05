// test cases for crop.go
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

package vision_transforms_functional_test

import (
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/Kautenja/gotorch"
    "github.com/Kautenja/gotorch/vision/transforms/functional"
)

func TestCropIdentity(t *testing.T) {
    tensor := torch.NewTensor([][]float32{
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 8, 7},
        {6, 5, 4, 3},
    })
    output := vision_transforms_functional.Crop(tensor, 0, 0, 4, 4)
    assert.True(t, output.Equal(tensor))
}

func TestCropExpandsYBounds(t *testing.T) {
    tensor := torch.NewTensor([][]float32{
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 8, 7},
        {6, 5, 4, 3},
    })
    output := vision_transforms_functional.Crop(tensor, 0, 0, 4, 5)
    expected := torch.NewTensor([][]float32{
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 8, 7},
        {6, 5, 4, 3},
        {0, 0, 0, 0},
    })
    assert.True(t, expected.Equal(output))
}

func TestCropExpandsXBounds(t *testing.T) {
    tensor := torch.NewTensor([][]float32{
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 8, 7},
        {6, 5, 4, 3},
    })
    output := vision_transforms_functional.Crop(tensor, 0, 0, 5, 4)
    expected := torch.NewTensor([][]float32{
        {0, 1, 2, 3, 0},
        {4, 5, 6, 7, 0},
        {8, 9, 8, 7, 0},
        {6, 5, 4, 3, 0},
    })
    assert.True(t, expected.Equal(output))
}

func TestCropExpandsBounds(t *testing.T) {
    tensor := torch.NewTensor([][]float32{
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 8, 7},
        {6, 5, 4, 3},
    })
    output := vision_transforms_functional.Crop(tensor, 0, 0, 5, 5)
    expected := torch.NewTensor([][]float32{
        {0, 1, 2, 3, 0},
        {4, 5, 6, 7, 0},
        {8, 9, 8, 7, 0},
        {6, 5, 4, 3, 0},
        {0, 0, 0, 0, 0},
    })
    assert.True(t, expected.Equal(output))
}

func TestCropShiftsYOrigin(t *testing.T) {
    tensor := torch.NewTensor([][]float32{
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 8, 7},
        {6, 5, 4, 3},
    })
    output := vision_transforms_functional.Crop(tensor, 0, -1, 4, 4)
    expected := torch.NewTensor([][]float32{
        {0, 0, 0, 0},
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 8, 7},
        {6, 5, 4, 3},
    })
    assert.True(t, expected.Equal(output))
}

func TestCropShiftsXOrigin(t *testing.T) {
    tensor := torch.NewTensor([][]float32{
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 8, 7},
        {6, 5, 4, 3},
    })
    output := vision_transforms_functional.Crop(tensor, -1, 0, 4, 4)
    expected := torch.NewTensor([][]float32{
        {0, 0, 1, 2, 3},
        {0, 4, 5, 6, 7},
        {0, 8, 9, 8, 7},
        {0, 6, 5, 4, 3},
    })
    assert.True(t, expected.Equal(output))
}

func TestCropShiftsOrigin(t *testing.T) {
    tensor := torch.NewTensor([][]float32{
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 8, 7},
        {6, 5, 4, 3},
    })
    output := vision_transforms_functional.Crop(tensor, -1, -1, 4, 4)
    expected := torch.NewTensor([][]float32{
        {0, 0, 0, 0, 0},
        {0, 0, 1, 2, 3},
        {0, 4, 5, 6, 7},
        {0, 8, 9, 8, 7},
        {0, 6, 5, 4, 3},
    })
    assert.True(t, expected.Equal(output))
}

func TestCropShiftsOriginAndExpandsBounds(t *testing.T) {
    tensor := torch.NewTensor([][]float32{
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 8, 7},
        {6, 5, 4, 3},
    })
    output := vision_transforms_functional.Crop(tensor, -1, -1, 5, 5)
    expected := torch.NewTensor([][]float32{
        {0, 0, 0, 0, 0, 0},
        {0, 0, 1, 2, 3, 0},
        {0, 4, 5, 6, 7, 0},
        {0, 8, 9, 8, 7, 0},
        {0, 6, 5, 4, 3, 0},
        {0, 0, 0, 0, 0, 0},
    })
    assert.True(t, expected.Equal(output))
}

func TestCropFromOriginToSafeBounds(t *testing.T) {
    tensor := torch.NewTensor([][]float32{
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 8, 7},
        {6, 5, 4, 3},
    })
    expected := torch.NewTensor([][]float32{
        {0, 1},
        {4, 5},
        {8, 9},
    })
    output := vision_transforms_functional.Crop(tensor, 0, 0, 2, 3)
    assert.True(t, output.Equal(expected), "got %v, expected %v", output, expected)
}

func TestCropFromSafePointToBounds(t *testing.T) {
    tensor := torch.NewTensor([][]float32{
        {0, 1, 2, 3},
        {4, 5, 6, 7},
        {8, 9, 8, 7},
        {6, 5, 4, 3},
    })
    expected := torch.NewTensor([][]float32{
        {6, 7},
        {8, 7},
        {4, 3},
    })
    output := vision_transforms_functional.Crop(tensor, 2, 1, 4, 4)
    assert.True(t, output.Equal(expected), "got %v, expected %v", output, expected)
}

func TestCropOfArbitrarySafeWindow(t *testing.T) {
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
    output := vision_transforms_functional.Crop(tensor, 1, 1, 3, 3)
    assert.True(t, output.Equal(expected), "got %v, expected %v", output, expected)
}

func TestCropTransformPanicsOn1DimensionalInput(t *testing.T) {
    tensor := torch.NewTensor([]float32{0, 1, 2, 3})
    assert.PanicsWithValue(t, "Crop requires inputs with 2 or more dimensions", func() {
        vision_transforms_functional.Crop(tensor, 0, 0, 1, 1)
    })
}
