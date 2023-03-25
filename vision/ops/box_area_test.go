// test cases for box_area.go
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

func TestBoxAreaPanicsOn1DInputs(t *testing.T) {
	tensor := torch.NewTensor([]int64{0, 0, 10, 10})
	message := "Expected inputs to be in (N, 4, ...) format, but received tensor with shape [4]"
	assert.PanicsWithValue(t, message, func() {
		ops.BoxArea(tensor)
	})
}

func TestBoxAreaPanicsOnNonBBoxInputs(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 0, 0}})
	message := "Expected inputs to be in (N, 4, ...) format, but received tensor with shape [1 3]"
	assert.PanicsWithValue(t, message, func() {
		ops.BoxArea(tensor)
	})
}

func TestBoxAreaComputesAreaOfBox(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 0, 10, 10}})
	area := ops.BoxArea(tensor)
	assert.Equal(t, float32(100.0), area.Item())
}

func TestBoxAreaComputesAreaOfBoxes(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 0, 10, 10}, {5, 5, 10, 10}})
	area := ops.BoxArea(tensor).ToSlice().([]float32)
	assert.Equal(t, float32(100.0), area[0])
	assert.Equal(t, float32(25.0), area[1])
}

func TestBoxAreaComputesAreaOfBoxesInSpatialLayout(t *testing.T) {
	a := torch.NewTensor([][]int64{{0, 0, 10, 10}})
	b := torch.NewTensor([][]int64{{5, 5, 10, 10}})
	c := torch.NewTensor([][]int64{{10, 20, 12, 25}})
	d := torch.NewTensor([][]int64{{3, 7, 8, 10}})
	boxes := torch.Stack([]*torch.Tensor{
		torch.Stack([]*torch.Tensor{a, b}, -1),
		torch.Stack([]*torch.Tensor{c, d}, -1),
	}, -1)
	area := ops.BoxArea(boxes)
	assert.Equal(t, []int64{1, 1, 2, 2}, area.Shape())
	assert.Equal(t, []float32{100, 10, 25, 15}, area.ToSlice())
}
