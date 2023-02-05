// test cases for longest_max_size.go
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
	F "github.com/Kautenja/gotorch/nn/functional"
	T "github.com/Kautenja/gotorch/vision/transforms"
)


func TestPadIfNeededTransformPanicsOnZeroSize(t *testing.T) {
	assert.PanicsWithValue(t, "size should be greater than 0", func() {
		T.LongestMaxSize(0, F.InterpolateBilinear, false, false)
	})
}

func TestPadIfNeededTransformPanicsOnNegativeSize(t *testing.T) {
	assert.PanicsWithValue(t, "size should be greater than 0", func() {
		T.LongestMaxSize(-1, F.InterpolateBilinear, false, false)
	})
}

func TestLongestMaxSizeTransformPanicsOn0DimensionalInput(t *testing.T) {
	tensor := torch.NewTensor([]float32{})
	assert.PanicsWithValue(t, "LongestMaxSize requires tensor with 3 or more dimensions", func() {
		T.LongestMaxSize(2, F.InterpolateBilinear, false, false).Forward(tensor)
	})
}

func TestLongestMaxSizeTransformPanicsOn1DimensionalInput(t *testing.T) {
	tensor := torch.NewTensor([]float32{0, 1, 2, 3})
	assert.PanicsWithValue(t, "LongestMaxSize requires tensor with 3 or more dimensions", func() {
		T.LongestMaxSize(2, F.InterpolateBilinear, false, false).Forward(tensor)
	})
}

func TestLongestMaxSizeTransformPanicsOn2DimensionalInput(t *testing.T) {
	tensor := torch.NewTensor([][]float32{{0, 1, 2, 3}, {4, 5, 6, 7}})
	assert.PanicsWithValue(t, "LongestMaxSize requires tensor with 3 or more dimensions", func() {
		T.LongestMaxSize(2, F.InterpolateBilinear, false, false).Forward(tensor)
	})
}

func TestLongestMaxSizeIdentity(t *testing.T) {
	tensor := torch.NewTensor([][]float32{
		{0, 1, 2, 3},
		{4, 5, 6, 7},
		{8, 9, 8, 7},
	}).View(1, 1, 3, 4)
	output := T.LongestMaxSize(4, F.InterpolateBilinear, false, false).Forward(tensor)
	assert.True(t, output.Equal(tensor))
}

func TestLongestMaxSizeShrink(t *testing.T) {
	tensor := torch.NewTensor([][]float32{
		{0, 1, 2, 3},
		{4, 5, 6, 7},
		{8, 9, 8, 7},
	}).View(1, 1, 3, 4)
	outputs := T.LongestMaxSize(3, F.InterpolateBilinear, true, false).Forward(tensor)
	expected := torch.NewTensor([][]float32{
		{0.0, 1.5, 3.0},
		{8.0, 8.5, 7.0},
	}).View(1, 1, 2, 3)
	assert.True(t, torch.AllClose(expected, outputs, 1e-5, 1e-3), "Got %v expected %v", outputs, expected)
}

func TestLongestMaxSizeGrow(t *testing.T) {
	tensor := torch.NewTensor([][]float32{
		{0, 1, 2, 3},
		{4, 5, 6, 7},
		{8, 9, 8, 7},
	}).View(1, 1, 3, 4)
	outputs := T.LongestMaxSize(5, F.InterpolateBilinear, true, false).Forward(tensor)
	expected := torch.NewTensor([][]float32{
		{0.0, 0.75, 1.5, 2.25, 3.0},
		{4.0, 4.75, 5.5, 6.25, 7.0},
		{8.0, 8.75, 8.5, 7.75, 7.0},
	}).View(1, 1, 3, 5)
	assert.True(t, torch.AllClose(expected, outputs, 1e-5, 1e-3), "Got %v expected %v", outputs, expected)
}
