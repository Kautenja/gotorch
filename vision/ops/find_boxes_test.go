// test cases for find_boxes.go
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
// MARK: FindBoxesInSizeRange
// ---------------------------------------------------------------------------

func TestFindBoxesInSizeRangePanicsOn1DInputs(t *testing.T) {
	tensor := torch.NewTensor([]int64{0, 0, 10, 10})
	message := "Expected inputs to be in (N, 4, ...) format, but received tensor with shape [4]"
	assert.PanicsWithValue(t, message, func() {
		ops.FindBoxesInSizeRange(tensor, 0, 0, 100, 100)
	})
}

func TestFindBoxesInSizeRangePanicsOnNonBBoxInputs(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 0, 0}})
	message := "Expected inputs to be in (N, 4, ...) format, but received tensor with shape [1 3]"
	assert.PanicsWithValue(t, message, func() {
		ops.FindBoxesInSizeRange(tensor, 0, 0, 100, 100)
	})
}

func TestFindBoxesInSizeRangeIdentityCase(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 0, 10, 10}})
	output := ops.FindBoxesInSizeRange(tensor, 0, 0, 100, 100)
	expected := torch.NewTensor([][]bool{{true}})
	assert.True(t, output.Equal(expected))
}

func TestFindBoxesInSizeRangeBoxes(t *testing.T) {
	tensor := torch.NewTensor([][]int64{
		{ 10,  20,  20,  30},  // [ 10 x  10] on inclusive lower bound
		{  0,   0, 100, 100},  // [100 x 100] on inclusive upper bound
		{ 25,  25,  75,  75},  // [ 50 x  50] within bounds
		{  0,   0,  10,   9},  // [  9 x  10] 1 below minimum height
		{  0,   0,   9,  10},  // [ 10 x   9] 1 below minimum width
		{  0,   0, 100, 101},  // [101 x 100] 1 above maximum height
		{  0,   0, 101, 100},  // [100 x 101] 1 above maximum width
		{100, 100, 105, 125},  // [ 25 x   5] width too small
		{ 50,  25, 250,  50},  // [ 25 x 200] width too large
		{100, 100, 125, 105},  // [  5 x  25] height too small
		{ 25,  50,  50, 250},  // [200 x  25] height too large
	})
	output := ops.FindBoxesInSizeRange(tensor, 10, 10, 100, 100)
	expected := torch.NewTensor([][]bool{
		{true},
		{true},
		{true},
		{false},
		{false},
		{false},
		{false},
		{false},
		{false},
		{false},
		{false},
	})
	assert.True(t, output.Equal(expected))
}

func TestFindBoxesInSizeRangeSpatialLayout(t *testing.T) {
	a := torch.NewTensor([][]int64{{0, 0,  10,  10}})
	b := torch.NewTensor([][]int64{{0, 0, 100, 100}})
	c := torch.NewTensor([][]int64{{0, 0, 200, 200}})
	d := torch.NewTensor([][]int64{{0, 0,   5,   5}})
	boxes := torch.Stack([]torch.Tensor{
		torch.Stack([]torch.Tensor{a, b}, -1),
		torch.Stack([]torch.Tensor{c, d}, -1),
	}, -1)
	output := ops.FindBoxesInSizeRange(boxes, 10, 10, 100, 100)
	// Create the expected output tensor.
	a = torch.NewTensor([]bool{true})
	b = torch.NewTensor([]bool{true})
	c = torch.NewTensor([]bool{false})
	d = torch.NewTensor([]bool{false})
	expected := torch.Stack([]torch.Tensor{
		torch.Stack([]torch.Tensor{a, b}, -1),
		torch.Stack([]torch.Tensor{c, d}, -1),
	}, -1).Unsqueeze(0)
	assert.True(t, output.Equal(expected))
}

// ---------------------------------------------------------------------------
// MARK: FindLargeBoxes
// ---------------------------------------------------------------------------

func TestFindLargeBoxesPanicsOn1DInputs(t *testing.T) {
	tensor := torch.NewTensor([]int64{0, 0, 10, 10})
	message := "Expected inputs to be in (N, 4, ...) format, but received tensor with shape [4]"
	assert.PanicsWithValue(t, message, func() {
		ops.FindLargeBoxes(tensor, 0, 0)
	})
}

func TestFindLargeBoxesPanicsOnNonBBoxInputs(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 0, 0}})
	message := "Expected inputs to be in (N, 4, ...) format, but received tensor with shape [1 3]"
	assert.PanicsWithValue(t, message, func() {
		ops.FindLargeBoxes(tensor, 0, 0)
	})
}

func TestFindLargeBoxesIdentityCase(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 0, 10, 10}})
	output := ops.FindLargeBoxes(tensor, 0, 0)
	expected := torch.NewTensor([][]bool{{true}})
	assert.True(t, output.Equal(expected))
}

func TestFindLargeBoxes(t *testing.T) {
	tensor := torch.NewTensor([][]int64{
		{ 10,  20,  20,  30},  // [ 10 x  10] on inclusive lower bound
		{ 25,  25,  75,  75},  // [ 50 x  50] above bounds
		{  0,   0,  10,   9},  // [  9 x  10] 1 below minimum height
		{  0,   0,   9,  10},  // [ 10 x   9] 1 below minimum width
		{100, 100, 105, 125},  // [ 25 x   5] width too small
		{100, 100, 125, 105},  // [  5 x  25] height too small
	})
	output := ops.FindLargeBoxes(tensor, 10, 10)
	expected := torch.NewTensor([][]bool{
		{true},
		{true},
		{false},
		{false},
		{false},
		{false},
	})
	assert.True(t, output.Equal(expected))
}

func TestFindLargeBoxesSpatialLayout(t *testing.T) {
	a := torch.NewTensor([][]int64{{0, 0,  10,  10}})
	b := torch.NewTensor([][]int64{{0, 0, 100, 100}})
	c := torch.NewTensor([][]int64{{0, 0, 200, 200}})
	d := torch.NewTensor([][]int64{{0, 0,   5,   5}})
	boxes := torch.Stack([]torch.Tensor{
		torch.Stack([]torch.Tensor{a, b}, -1),
		torch.Stack([]torch.Tensor{c, d}, -1),
	}, -1)
	output := ops.FindLargeBoxes(boxes, 10, 10)
	// Create the expected output tensor.
	a = torch.NewTensor([]bool{true})
	b = torch.NewTensor([]bool{true})
	c = torch.NewTensor([]bool{true})
	d = torch.NewTensor([]bool{false})
	expected := torch.Stack([]torch.Tensor{
		torch.Stack([]torch.Tensor{a, b}, -1),
		torch.Stack([]torch.Tensor{c, d}, -1),
	}, -1).Unsqueeze(0)
	assert.True(t, output.Equal(expected))
}

// ---------------------------------------------------------------------------
// MARK: FindSmallBoxes
// ---------------------------------------------------------------------------

func TestFindSmallBoxesPanicsOn1DInputs(t *testing.T) {
	tensor := torch.NewTensor([]int64{0, 0, 10, 10})
	message := "Expected inputs to be in (N, 4, ...) format, but received tensor with shape [4]"
	assert.PanicsWithValue(t, message, func() {
		ops.FindSmallBoxes(tensor, 100, 100)
	})
}

func TestFindSmallBoxesPanicsOnNonBBoxInputs(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 0, 0}})
	message := "Expected inputs to be in (N, 4, ...) format, but received tensor with shape [1 3]"
	assert.PanicsWithValue(t, message, func() {
		ops.FindSmallBoxes(tensor, 100, 100)
	})
}

func TestFindSmallBoxesIdentityCase(t *testing.T) {
	tensor := torch.NewTensor([][]int64{{0, 0, 10, 10}})
	output := ops.FindSmallBoxes(tensor, 100, 100)
	expected := torch.NewTensor([][]bool{{true}})
	assert.True(t, output.Equal(expected))
}

func TestFindSmallBoxes(t *testing.T) {
	tensor := torch.NewTensor([][]int64{
		{  0,   0, 100, 100},  // [100 x 100] on inclusive upper bound
		{ 25,  25,  75,  75},  // [ 50 x  50] within bounds
		{  0,   0, 100, 101},  // [101 x 100] 1 above maximum height
		{  0,   0, 101, 100},  // [100 x 101] 1 above maximum width
		{ 50,  25, 250,  50},  // [ 25 x 200] width too large
		{ 25,  50,  50, 250},  // [200 x  25] height too large
	})
	output := ops.FindSmallBoxes(tensor, 100, 100)
	expected := torch.NewTensor([][]bool{
		{true},
		{true},
		{false},
		{false},
		{false},
		{false},
	})
	assert.True(t, output.Equal(expected))
}

func TestFindSmallBoxesSpatialLayout(t *testing.T) {
	a := torch.NewTensor([][]int64{{0, 0,  10,  10}})
	b := torch.NewTensor([][]int64{{0, 0, 100, 100}})
	c := torch.NewTensor([][]int64{{0, 0, 200, 200}})
	d := torch.NewTensor([][]int64{{0, 0,   5,   5}})
	boxes := torch.Stack([]torch.Tensor{
		torch.Stack([]torch.Tensor{a, b}, -1),
		torch.Stack([]torch.Tensor{c, d}, -1),
	}, -1)
	output := ops.FindSmallBoxes(boxes, 100, 100)
	// Create the expected output tensor.
	a = torch.NewTensor([]bool{true})
	b = torch.NewTensor([]bool{true})
	c = torch.NewTensor([]bool{false})
	d = torch.NewTensor([]bool{true})
	expected := torch.Stack([]torch.Tensor{
		torch.Stack([]torch.Tensor{a, b}, -1),
		torch.Stack([]torch.Tensor{c, d}, -1),
	}, -1).Unsqueeze(0)
	assert.True(t, output.Equal(expected))
}
