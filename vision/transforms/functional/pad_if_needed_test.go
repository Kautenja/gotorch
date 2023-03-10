// test cases for pad_if_needed.go
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
	F "github.com/Kautenja/gotorch/nn/functional"
	T "github.com/Kautenja/gotorch/vision/transforms/functional"
)

// ---------------------------------------------------------------------------
// MARK: Panics
// ---------------------------------------------------------------------------

func TestPadIfNeededTransformPanicsOn0DimensionalInput(t *testing.T) {
	tensor := torch.NewTensor([]float32{})
	assert.PanicsWithValue(t, "PadIfNeeded requires tensor with 2 or more dimensions", func() {
		T.PadIfNeeded(tensor, 2, 2, F.PadConstant, 0)
	})
}

func TestPadIfNeededTransformPanicsOn1DimensionalInput(t *testing.T) {
	tensor := torch.NewTensor([]float32{0, 1, 2, 3})
	assert.PanicsWithValue(t, "PadIfNeeded requires tensor with 2 or more dimensions", func() {
		T.PadIfNeeded(tensor, 2, 2, F.PadConstant, 0)
	})
}

// ---------------------------------------------------------------------------
// MARK: Identity
// ---------------------------------------------------------------------------

func TestPadIfNeededIdentity2D(t *testing.T) {
	tensor := torch.NewTensor([][]float32{
		{0, 1, 2, 3},
		{4, 5, 6, 7},
		{8, 9, 8, 7},
	})
	outputs, padding := T.PadIfNeeded(tensor, 3, 4, F.PadConstant, 0)
	assert.True(t, outputs.Equal(tensor))
	assert.Equal(t, []int64{0, 0, 0, 0}, padding)
}

func TestPadIfNeededIdentity3D(t *testing.T) {
	tensor := torch.NewTensor([][]float32{
		{0, 1, 2, 3},
		{4, 5, 6, 7},
		{8, 9, 8, 7},
	}).View(1, 3, 4)
	outputs, padding := T.PadIfNeeded(tensor, 3, 4, F.PadConstant, 0)
	assert.True(t, outputs.Equal(tensor))
	assert.Equal(t, []int64{0, 0, 0, 0}, padding)
}

func TestPadIfNeededIdentity4D(t *testing.T) {
	tensor := torch.NewTensor([][]float32{
		{0, 1, 2, 3},
		{4, 5, 6, 7},
		{8, 9, 8, 7},
	}).View(1, 1, 3, 4)
	outputs, padding := T.PadIfNeeded(tensor, 3, 4, F.PadConstant, 0)
	assert.True(t, outputs.Equal(tensor))
	assert.Equal(t, []int64{0, 0, 0, 0}, padding)
}

func TestPadIfNeededIdentity5D(t *testing.T) {
	tensor := torch.NewTensor([][]float32{
		{0, 1, 2, 3},
		{4, 5, 6, 7},
		{8, 9, 8, 7},
	}).View(1, 1, 1, 3, 4)
	outputs, padding := T.PadIfNeeded(tensor, 3, 4, F.PadConstant, 0)
	assert.True(t, outputs.Equal(tensor))
	assert.Equal(t, []int64{0, 0, 0, 0}, padding)
}

// ---------------------------------------------------------------------------
// MARK: NoOp
// ---------------------------------------------------------------------------

func TestPadIfNeededNoOp2D(t *testing.T) {
	tensor := torch.NewTensor([][]float32{
		{0, 1, 2, 3, 0},
		{4, 5, 6, 7, 0},
		{8, 9, 8, 7, 0},
		{4, 5, 6, 7, 0},
		{0, 1, 2, 3, 0},
	})
	outputs, padding := T.PadIfNeeded(tensor, 3, 4, F.PadConstant, 0)
	assert.True(t, outputs.Equal(tensor))
	assert.Equal(t, []int64{0, 0, 0, 0}, padding)
}

func TestPadIfNeededNoOp3D(t *testing.T) {
	tensor := torch.NewTensor([][]float32{
		{0, 1, 2, 3, 0},
		{4, 5, 6, 7, 0},
		{8, 9, 8, 7, 0},
		{4, 5, 6, 7, 0},
		{0, 1, 2, 3, 0},
	}).Unsqueeze(0)
	outputs, padding := T.PadIfNeeded(tensor, 3, 4, F.PadConstant, 0)
	assert.True(t, outputs.Equal(tensor))
	assert.Equal(t, []int64{0, 0, 0, 0}, padding)
}

func TestPadIfNeededNoOp4D(t *testing.T) {
	tensor := torch.NewTensor([][]float32{
		{0, 1, 2, 3, 0},
		{4, 5, 6, 7, 0},
		{8, 9, 8, 7, 0},
		{4, 5, 6, 7, 0},
		{0, 1, 2, 3, 0},
	}).Unsqueeze(0).Unsqueeze(0)
	outputs, padding := T.PadIfNeeded(tensor, 3, 4, F.PadConstant, 0)
	assert.True(t, outputs.Equal(tensor))
	assert.Equal(t, []int64{0, 0, 0, 0}, padding)
}

func TestPadIfNeededNoOp5D(t *testing.T) {
	tensor := torch.NewTensor([][]float32{
		{0, 1, 2, 3, 0},
		{4, 5, 6, 7, 0},
		{8, 9, 8, 7, 0},
		{4, 5, 6, 7, 0},
		{0, 1, 2, 3, 0},
	}).Unsqueeze(0).Unsqueeze(0).Unsqueeze(0)
	outputs, padding := T.PadIfNeeded(tensor, 3, 4, F.PadConstant, 0)
	assert.True(t, outputs.Equal(tensor))
	assert.Equal(t, []int64{0, 0, 0, 0}, padding)
}

// ---------------------------------------------------------------------------
// MARK: Transformation
// ---------------------------------------------------------------------------

func TestPadIfNeededHeight(t *testing.T) {
	tensor := torch.NewTensor([][]float32{
		{0, 1, 2, 3},
		{4, 5, 6, 7},
		{8, 9, 8, 7},
	}).Unsqueeze(0)
	outputs, padding := T.PadIfNeeded(tensor, 4, 4, F.PadConstant, 0)
	expected := torch.NewTensor([][]float32{
		{0, 1, 2, 3},
		{4, 5, 6, 7},
		{8, 9, 8, 7},
		{0, 0, 0, 0},
	}).Unsqueeze(0)
	assert.True(t, outputs.Equal(expected), "Got %v expected %v", outputs, expected)
	assert.Equal(t, []int64{0, 0, 0, 1}, padding)
}

func TestPadIfNeededWidth(t *testing.T) {
	tensor := torch.NewTensor([][]float32{
		{0, 1, 2},
		{4, 5, 6},
		{8, 9, 8},
		{8, 9, 8},
	}).Unsqueeze(0)
	outputs, padding := T.PadIfNeeded(tensor, 4, 4, F.PadConstant, 0)
	expected := torch.NewTensor([][]float32{
		{0, 1, 2, 0},
		{4, 5, 6, 0},
		{8, 9, 8, 0},
		{8, 9, 8, 0},
	}).Unsqueeze(0)
	assert.True(t, outputs.Equal(expected), "Got %v expected %v", outputs, expected)
	assert.Equal(t, []int64{0, 1, 0, 0}, padding)
}

func TestPadIfNeeded(t *testing.T) {
	tensor := torch.NewTensor([][]float32{
		{0, 1, 2, 3},
		{4, 5, 6, 7},
		{8, 9, 8, 7},
	}).Unsqueeze(0)
	outputs, padding := T.PadIfNeeded(tensor, 5, 5, F.PadConstant, 0)
	expected := torch.NewTensor([][]float32{
		{0, 0, 0, 0, 0},
		{0, 1, 2, 3, 0},
		{4, 5, 6, 7, 0},
		{8, 9, 8, 7, 0},
		{0, 0, 0, 0, 0},
	}).Unsqueeze(0)
	assert.True(t, outputs.Equal(expected), "Got %v expected %v", outputs, expected)
	assert.Equal(t, []int64{0, 1, 1, 1}, padding)
}
