// Test cases for functions.go
//
// Copyright (c) 2023 Christian Kauten
// Copyright (c) 2022 Sensory, Inc.
// Copyright (c) 2020 GoTorch Authors
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

package torch_test

import (
	"math"
	"testing"
	"fmt"
	"github.com/stretchr/testify/assert"
	"github.com/Kautenja/gotorch"
)

// ---------------------------------------------------------------------------
// MARK: Tensor Metadata
// ---------------------------------------------------------------------------

func TestNumel(t *testing.T) {
	assert.Equal(t, int64(0), torch.Zeros([]int64{0}, torch.NewTensorOptions()).Numel())
	assert.Equal(t, int64(1), torch.Zeros([]int64{1}, torch.NewTensorOptions()).Numel())
	assert.Equal(t, int64(4), torch.Zeros([]int64{2, 2}, torch.NewTensorOptions()).Numel())
}

func TestIsComplex(t *testing.T) {
	assert.False(t, torch.Zeros([]int64{0}, torch.NewTensorOptions()).IsComplex())
	assert.True(t, torch.Zeros([]int64{0}, torch.NewTensorOptions().Dtype(torch.ComplexFloat)).IsComplex())
}

func TestIsConj(t *testing.T) {
	assert.False(t, torch.Zeros([]int64{0}, torch.NewTensorOptions().Dtype(torch.ComplexFloat)).IsConj())
	// TODO: Implement `conj`
	// assert.True(t, torch.Zeros([]int64{0}, torch.NewTensorOptions().Dtype(torch.ComplexFloat)).Conj().IsConj())
}

func TestIsFloatingPoint(t *testing.T) {
	assert.False(t, torch.Zeros([]int64{0}, torch.NewTensorOptions().Dtype(torch.Long)).IsFloatingPoint())
	assert.True(t, torch.Zeros([]int64{0}, torch.NewTensorOptions().Dtype(torch.Float)).IsFloatingPoint())
}

func TestIsNonzero(t *testing.T) {
	assert.False(t, torch.Zeros([]int64{1}, torch.NewTensorOptions()).IsNonzero())
	assert.True(t, torch.Ones([]int64{1}, torch.NewTensorOptions()).IsNonzero())
}

// -----------------------------------------------------------------------------
// MARK: Zeros
// -----------------------------------------------------------------------------

func TestZerosCreatesEmptyTensor(t *testing.T) {
	tensor := torch.Zeros([]int64{0}, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([]float32{}).Equal(tensor))
}

func TestZerosCreatesItem(t *testing.T) {
	tensor := torch.Zeros([]int64{1}, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([]float32{0}).Equal(tensor))
}

func TestZerosCreatesMatrix(t *testing.T) {
	tensor := torch.Zeros([]int64{3, 3}, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([][]float32{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}).Equal(tensor))
}

func TestZerosUsesTensorOptions(t *testing.T) {
	tensor := torch.Zeros([]int64{1}, torch.NewTensorOptions().RequiresGrad(true))
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, tensor.RequiresGrad())
}

func TestZerosPanicsOnEmptySize(t *testing.T) {
	assert.PanicsWithValue(t, "size is empty", func() {
		torch.Zeros([]int64{}, torch.NewTensorOptions())
	})
}

func TestZerosPanicsOnInvalidSize(t *testing.T) {
	assert.PanicsWithValue(t, "Trying to create tensor with negative dimension -1: [-1]", func() {
		torch.Zeros([]int64{-1}, torch.NewTensorOptions())
	})
}

// -----------------------------------------------------------------------------
// MARK: ZerosLike
// -----------------------------------------------------------------------------

func TestZerosLike(t *testing.T) {
	reference := torch.Ones([]int64{3, 3}, torch.NewTensorOptions())
	tensor := torch.ZerosLike(reference)
	assert.NotNil(t, tensor.Pointer)
	assert.Equal(t, []int64{3, 3}, tensor.Shape())
	expected := torch.Zeros([]int64{3, 3}, torch.NewTensorOptions())
	assert.True(t, torch.Equal(expected, tensor))
}

func TestZerosLikePanicsOnUninitializedInput(t *testing.T) {
	tensor := torch.Tensor{}
	assert.PanicsWithValue(t, "input tensor is nil", func() {
		torch.ZerosLike(tensor)
	})
}

// -----------------------------------------------------------------------------
// MARK: Ones
// -----------------------------------------------------------------------------

func TestOnesCreatesEmptyTensor(t *testing.T) {
	tensor := torch.Ones([]int64{0}, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([]float32{}).Equal(tensor))
}

func TestOnesCreatesItem(t *testing.T) {
	tensor := torch.Ones([]int64{1}, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([]float32{1}).Equal(tensor))
}

func TestOnesCreatesMatrix(t *testing.T) {
	tensor := torch.Ones([]int64{3, 3}, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([][]float32{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}).Equal(tensor))
}

func TestOnesUsesTensorOptions(t *testing.T) {
	tensor := torch.Ones([]int64{1}, torch.NewTensorOptions().RequiresGrad(true))
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, tensor.RequiresGrad())
}

func TestOnesPanicsOnEmptySize(t *testing.T) {
	assert.PanicsWithValue(t, "size is empty", func() {
		torch.Ones([]int64{}, torch.NewTensorOptions())
	})
}

func TestOnesPanicsOnInvalidSize(t *testing.T) {
	assert.PanicsWithValue(t, "Trying to create tensor with negative dimension -1: [-1]", func() {
		torch.Zeros([]int64{-1}, torch.NewTensorOptions())
	})
}

// -----------------------------------------------------------------------------
// MARK: OnesLike
// -----------------------------------------------------------------------------

func TestOnesLikeWithoutGrad(t *testing.T) {
	reference := torch.Zeros([]int64{3, 3}, torch.NewTensorOptions())
	tensor := torch.OnesLike(reference)
	assert.NotNil(t, tensor.Pointer)
	assert.Equal(t, reference.Shape(), tensor.Shape())
	expected := torch.Ones([]int64{3, 3}, torch.NewTensorOptions())
	assert.True(t, torch.Equal(expected, tensor))
}

func TestOnesLikePanicsOnUninitializedInput(t *testing.T) {
	tensor := torch.Tensor{}
	assert.PanicsWithValue(t, "input tensor is nil", func() {
		torch.OnesLike(tensor)
	})
}

// -----------------------------------------------------------------------------
// MARK: Arange
// -----------------------------------------------------------------------------

func TestArangeValuesStepSize1(t *testing.T) {
	tensor := torch.Arange(0, 5, 1, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	expected := torch.NewTensor([]float32{0, 1, 2, 3, 4})
	assert.True(t, torch.Equal(expected, tensor))
}

func TestArangeValuesStepSize2(t *testing.T) {
	tensor := torch.Arange(0, 5, 2, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	expected := torch.NewTensor([]float32{0, 2, 4})
	assert.True(t, torch.Equal(expected, tensor))
}

func TestArangeValuesStepSize3(t *testing.T) {
	tensor := torch.Arange(0, 5, 3, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	expected := torch.NewTensor([]float32{0, 3})
	assert.True(t, torch.Equal(expected, tensor))
}

func TestArangeValuesStepSize5(t *testing.T) {
	tensor := torch.Arange(0, 5, 5, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	expected := torch.NewTensor([]float32{0})
	assert.True(t, torch.Equal(expected, tensor))
}

func TestArangeUsesTensorOptions(t *testing.T) {
	tensor := torch.Arange(0, 1, 1, torch.NewTensorOptions().RequiresGrad(true))
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, tensor.RequiresGrad())
}

func TestArangeThrowsErrorOnInvalidStepSize(t *testing.T) {
	assert.PanicsWithValue(t, "upper bound and larger bound inconsistent with step sign", func() {
		torch.Arange(0, 5, -1, torch.NewTensorOptions())
	})
}

func TestArangeThrowsErrorOnInvalidLargerBound(t *testing.T) {
	assert.PanicsWithValue(t, "upper bound and larger bound inconsistent with step sign", func() {
		torch.Arange(0, -5, 1, torch.NewTensorOptions())
	})
}

// -----------------------------------------------------------------------------
// MARK: Range
// -----------------------------------------------------------------------------

func TestRangeValuesStepSize1(t *testing.T) {
	tensor := torch.Range(0, 5, 1, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	expected := torch.NewTensor([]float32{0, 1, 2, 3, 4, 5})
	assert.True(t, torch.Equal(expected, tensor))
}

func TestRangeValuesStepSize2(t *testing.T) {
	tensor := torch.Range(0, 5, 2, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	expected := torch.NewTensor([]float32{0, 2, 4})
	assert.True(t, torch.Equal(expected, tensor))
}

func TestRangeValuesStepSize3(t *testing.T) {
	tensor := torch.Range(0, 5, 3, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	expected := torch.NewTensor([]float32{0, 3})
	assert.True(t, torch.Equal(expected, tensor))
}

func TestRangeValuesStepSize5(t *testing.T) {
	tensor := torch.Range(0, 5, 5, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	expected := torch.NewTensor([]float32{0, 5})
	assert.True(t, torch.Equal(expected, tensor))
}

func TestRangeUsesTensorOptions(t *testing.T) {
	tensor := torch.Range(0, 1, 1, torch.NewTensorOptions().RequiresGrad(true))
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, tensor.RequiresGrad())
}

func TestRangeThrowsErrorOnInvalidStepSize(t *testing.T) {
	assert.PanicsWithValue(t, "upper bound and larger bound inconsistent with step sign", func() {
		torch.Range(0, 5, -1, torch.NewTensorOptions())
	})
}

func TestRangeThrowsErrorOnInvalidLargerBound(t *testing.T) {
	assert.PanicsWithValue(t, "upper bound and larger bound inconsistent with step sign", func() {
		torch.Range(0, -5, 1, torch.NewTensorOptions())
	})
}

// -----------------------------------------------------------------------------
// MARK: Linspace
// -----------------------------------------------------------------------------

func TestLinspaceValues(t *testing.T) {
	tensor := torch.Linspace(0, 5, 6, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([]float32{0, 1, 2, 3, 4, 5}).Equal(tensor))

	tensor = torch.Linspace(0, 5, 3, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([]float32{0, 2.5, 5}).Equal(tensor))

	tensor = torch.Linspace(0, 5, 1, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([]float32{0}).Equal(tensor))
}

func TestLinspaceUsesTensorOptions(t *testing.T) {
	tensor := torch.Linspace(0, 1, 1, torch.NewTensorOptions().RequiresGrad(true))
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, tensor.RequiresGrad())
}

func TestLinspacePanicsOnInvalidStepSize(t *testing.T) {
	assert.PanicsWithValue(t, "number of steps must be non-negative", func() {
		torch.Linspace(0, 5, -1, torch.NewTensorOptions())
	})
}

// -----------------------------------------------------------------------------
// MARK: Logspace
// -----------------------------------------------------------------------------

func TestLogspaceValues(t *testing.T) {
	tensor := torch.Logspace(0, 5, 6, 10, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([]float32{1, 10, 100, 1000, 10000, 100000}).Equal(tensor))

	tensor = torch.Logspace(0, 5, 3, 10, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.AllClose(torch.NewTensor([]float32{1.0000e+00, 3.1623e+02, 1.0000e+05}), tensor, 1e-5, 1e-3))

	tensor = torch.Logspace(0, 5, 1, 10, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([]float32{1}).Equal(tensor))

	tensor = torch.Logspace(0, 5, 6, 2, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([]float32{1, 2, 4, 8, 16, 32}).Equal(tensor))
}

func TestLogspaceUsesTensorOptions(t *testing.T) {
	tensor := torch.Logspace(0, 5, 6, 10, torch.NewTensorOptions().RequiresGrad(true))
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, tensor.RequiresGrad())
}

func TestLogspacePanicsErrorOnInvalidSize(t *testing.T) {
	assert.PanicsWithValue(t, "number of steps must be non-negative", func() {
		torch.Logspace(0, 5, -6, 2, torch.NewTensorOptions())
	})
}

// -----------------------------------------------------------------------------
// MARK: Eye
// -----------------------------------------------------------------------------

func TestEyeValues(t *testing.T) {
	tensor := torch.Eye(3, 3, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([][]float32{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}).Equal(tensor))

	tensor = torch.Eye(3, 1, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([][]float32{{1}, {0}, {0}}).Equal(tensor))

	tensor = torch.Eye(1, 3, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([][]float32{{1, 0, 0}}).Equal(tensor))

	tensor = torch.Eye(3, 2, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([][]float32{{1, 0}, {0, 1}, {0, 0}}).Equal(tensor))
}

func TestEyeUsesTensorOptions(t *testing.T) {
	tensor := torch.Eye(3, 3, torch.NewTensorOptions().RequiresGrad(true))
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, tensor.RequiresGrad())
}

func TestEyeThrowsErrorOnInvalidN(t *testing.T) {
	assert.PanicsWithValue(t, "n must be greater or equal to 0, got -1", func() {
		torch.Eye(-1, 3, torch.NewTensorOptions())
	})
}

// -----------------------------------------------------------------------------
// MARK: Empty
// -----------------------------------------------------------------------------

func TestEmptyCreatesEmptyTensor(t *testing.T) {
	tensor := torch.Empty([]int64{0}, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([]float32{}).Equal(tensor))
}

func TestEmptyCreatesItem(t *testing.T) {
	tensor := torch.Empty([]int64{1}, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.Equal(t, []int64{1}, tensor.Shape())
}

func TestEmptyCreatesMatrix(t *testing.T) {
	tensor := torch.Empty([]int64{3, 3}, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.Equal(t, []int64{3, 3}, tensor.Shape())
}

func TestEmptyUsesTensorOptions(t *testing.T) {
	tensor := torch.Empty([]int64{1}, torch.NewTensorOptions().RequiresGrad(true))
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, tensor.RequiresGrad())
}

func TestEmptyPanicsOnEmptySize(t *testing.T) {
	assert.PanicsWithValue(t, "size is empty", func() {
		torch.Empty([]int64{}, torch.NewTensorOptions())
	})
}

func TestEmptyPanicsOnInvalidSize(t *testing.T) {
	assert.PanicsWithValue(t, "Trying to create tensor with negative dimension -1: [-1]", func() {
		torch.Empty([]int64{-1}, torch.NewTensorOptions())
	})
}

// -----------------------------------------------------------------------------
// MARK: EmptyLike
// -----------------------------------------------------------------------------

func TestEmptyLike(t *testing.T) {
	reference := torch.Rand([]int64{3, 3}, torch.NewTensorOptions())
	tensor := torch.EmptyLike(reference)
	assert.NotNil(t, tensor.Pointer)
	assert.Equal(t, reference.Shape(), tensor.Shape())
	assert.False(t, torch.Equal(reference, tensor))
}

func TestEmptyLikePanicsOnUninitializedInput(t *testing.T) {
	tensor := torch.Tensor{}
	assert.PanicsWithValue(t, "input tensor is nil", func() {
		torch.EmptyLike(tensor)
	})
}

// -----------------------------------------------------------------------------
// MARK: Full
// -----------------------------------------------------------------------------

func TestFullCreatesEmptyTensor(t *testing.T) {
	tensor := torch.Full([]int64{0}, 1, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([]float32{}).Equal(tensor))
}

func TestFullCreatesItem(t *testing.T) {
	tensor := torch.Full([]int64{1}, 1, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.Equal(t, []int64{1}, tensor.Shape())
	expected := torch.Ones([]int64{1}, torch.NewTensorOptions())
	assert.True(t, torch.Equal(expected, tensor))
}

func TestFullCreatesMatrix(t *testing.T) {
	tensor := torch.Full([]int64{3, 3}, 1, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.Equal(t, []int64{3, 3}, tensor.Shape())
	expected := torch.Ones([]int64{3, 3}, torch.NewTensorOptions())
	assert.True(t, torch.Equal(expected, tensor))
}

func TestFullUsesTensorOptions(t *testing.T) {
	tensor := torch.Full([]int64{1}, 1, torch.NewTensorOptions().RequiresGrad(true))
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, tensor.RequiresGrad())
}

func TestFullPanicsOnEmptySize(t *testing.T) {
	assert.PanicsWithValue(t, "size is empty", func() {
		torch.Full([]int64{}, 1, torch.NewTensorOptions())
	})
}

func TestFullPanicsOnInvalidSize(t *testing.T) {
	assert.PanicsWithValue(t, "Trying to create tensor with negative dimension -1: [-1]", func() {
		torch.Full([]int64{-1}, 1, torch.NewTensorOptions())
	})
}

// -----------------------------------------------------------------------------
// MARK: FullLike
// -----------------------------------------------------------------------------

func TestFullLike(t *testing.T) {
	reference := torch.Rand([]int64{3, 3}, torch.NewTensorOptions())
	tensor := torch.FullLike(reference, 1)
	assert.NotNil(t, tensor.Pointer)
	assert.Equal(t, []int64{3, 3}, tensor.Shape())
	expected := torch.Ones([]int64{3, 3}, torch.NewTensorOptions())
	assert.True(t, torch.Equal(expected, tensor))
}

func TestFullLikePanicsOnUninitializedInput(t *testing.T) {
	tensor := torch.Tensor{}
	assert.PanicsWithValue(t, "input tensor is nil", func() {
		torch.FullLike(tensor, 1)
	})
}

// -----------------------------------------------------------------------------
// MARK: Rand
// -----------------------------------------------------------------------------

func TestRandCreatesEmptyTensor(t *testing.T) {
	tensor := torch.Rand([]int64{0}, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([]float32{}).Equal(tensor))
}

func TestRandValues(t *testing.T) {
	tensor := torch.Rand([]int64{1, 3, 1024, 1024}, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.Equal(t, []int64{1, 3, 1024, 1024}, tensor.Shape())
	// python -c "import torch; print(torch.rand(1, 3, 2**10, 2**10).mean())"
	mean := float64(tensor.Mean().Item().(float32))
	if math.Abs(mean - 0.5) > 1e-2 {
		t.Log(fmt.Sprintf("mean %f is not close to 0.5!", mean))
		t.Fail()
	}
	// python -c "import torch; print(torch.rand(1, 3, 2**10, 2**10).std())"
	std := float64(tensor.Std().Item().(float32))
	if math.Abs(std - 0.2886) > 1e-2 {
		t.Log(fmt.Sprintf("std %f is not close to 0.2886!", std))
		t.Fail()
	}
}

func TestRandUsesTensorOptions(t *testing.T) {
	tensor := torch.Rand([]int64{1}, torch.NewTensorOptions().RequiresGrad(true))
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, tensor.RequiresGrad())
}

func TestRandPanicsOnEmptySize(t *testing.T) {
	assert.PanicsWithValue(t, "size is empty", func() {
		torch.Rand([]int64{}, torch.NewTensorOptions())
	})
}

func TestRandPanicsOnInvalidSize(t *testing.T) {
	assert.PanicsWithValue(t, "Trying to create tensor with negative dimension -1: [-1]", func() {
		torch.Rand([]int64{-1}, torch.NewTensorOptions())
	})
}

// -----------------------------------------------------------------------------
// MARK: RandLike
// -----------------------------------------------------------------------------

func TestRandLike(t *testing.T) {
	reference := torch.Rand([]int64{3, 3}, torch.NewTensorOptions())
	tensor := torch.RandLike(reference)
	assert.NotNil(t, tensor.Pointer)
	assert.Equal(t, reference.Shape(), tensor.Shape())
	assert.False(t, torch.Equal(reference, tensor))
}

func TestRandLikePanicsOnUninitializedInput(t *testing.T) {
	tensor := torch.Tensor{}
	assert.PanicsWithValue(t, "input tensor is nil", func() {
		torch.RandLike(tensor)
	})
}

// -----------------------------------------------------------------------------
// MARK: RantInt
// -----------------------------------------------------------------------------

func TestRandIntCreatesEmptyTensor(t *testing.T) {
	tensor := torch.RandInt([]int64{0}, 0, 1, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([]float32{}).Equal(tensor))
}

func TestRandIntValues(t *testing.T) {
	tensor := torch.RandInt([]int64{1, 3, 1024, 1024}, 0, 11, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.Equal(t, []int64{1, 3, 1024, 1024}, tensor.Shape())
	// python -c "import torch; print(torch.randint(0, 11, size=(1, 3, 2**10, 2**10)).float().mean())"
	mean := float64(tensor.Mean().Item().(float32))
	if math.Abs(mean - 5) > 1e-2 {
		t.Log(fmt.Sprintf("mean %f is not close to 5!", mean))
		t.Fail()
	}
	// python -c "import torch; print(torch.randint(0, 11, size=(1, 3, 2**10, 2**10)).float().std())"
	std := float64(tensor.Std().Item().(float32))
	if math.Abs(std - 3.1630) > 1e-2 {
		t.Log(fmt.Sprintf("std %f is not close to 3.1630!", std))
		t.Fail()
	}
}

func TestRandIntUsesTensorOptions(t *testing.T) {
	tensor := torch.RandInt([]int64{1}, 0, 1, torch.NewTensorOptions().RequiresGrad(true))
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, tensor.RequiresGrad())
}

func TestRandIntPanicsOnEmptySize(t *testing.T) {
	assert.PanicsWithValue(t, "size is empty", func() {
		torch.RandInt([]int64{}, 0, 1, torch.NewTensorOptions())
	})
}

func TestRandIntPanicsOnInvalidSize(t *testing.T) {
	assert.PanicsWithValue(t, "Trying to create tensor with negative dimension -1: [-1]", func() {
		torch.RandInt([]int64{-1}, 0, 1, torch.NewTensorOptions())
	})
}

// -----------------------------------------------------------------------------
// MARK: RandIntLike
// -----------------------------------------------------------------------------

func TestRandIntLike(t *testing.T) {
	reference := torch.Rand([]int64{3, 3}, torch.NewTensorOptions())
	tensor := torch.RandIntLike(reference, 10, 20)
	assert.NotNil(t, tensor.Pointer)
	assert.Equal(t, reference.Shape(), tensor.Shape())
	assert.False(t, torch.Equal(reference, tensor))
}

func TestRandIntLikePanicsOnUninitializedInput(t *testing.T) {
	tensor := torch.Tensor{}
	assert.PanicsWithValue(t, "input tensor is nil", func() {
		torch.RandIntLike(tensor, 0, 1)
	})
}

// -----------------------------------------------------------------------------
// MARK: RandN
// -----------------------------------------------------------------------------

func TestRandNCreatesEmptyTensor(t *testing.T) {
	tensor := torch.RandN([]int64{0}, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, torch.NewTensor([]float32{}).Equal(tensor))
}

func TestRandNValues(t *testing.T) {
	tensor := torch.RandN([]int64{1, 3, 1024, 1024}, torch.NewTensorOptions())
	assert.NotNil(t, tensor.Pointer)
	assert.Equal(t, []int64{1, 3, 1024, 1024}, tensor.Shape())
	// python -c "import torch; print(torch.randn(1, 3, 2**10, 2**10).float().mean())"
	mean := float64(tensor.Mean().Item().(float32))
	if math.Abs(mean - 0) > 1e-2 {
		t.Log(fmt.Sprintf("mean %f is not close to 0!", mean))
		t.Fail()
	}
	// python -c "import torch; print(torch.randn(1, 3, 2**10, 2**10).float().mean())"
	std := float64(tensor.Std().Item().(float32))
	if math.Abs(std - 1) > 1e-2 {
		t.Log(fmt.Sprintf("std %f is not close to 1!", std))
		t.Fail()
	}
}

func TestRandNUsesTensorOptions(t *testing.T) {
	tensor := torch.RandN([]int64{1}, torch.NewTensorOptions().RequiresGrad(true))
	assert.NotNil(t, tensor.Pointer)
	assert.True(t, tensor.RequiresGrad())
}

func TestRandNPanicsOnEmptySize(t *testing.T) {
	assert.PanicsWithValue(t, "size is empty", func() {
		torch.RandN([]int64{}, torch.NewTensorOptions())
	})
}

func TestRandNPanicsOnInvalidSize(t *testing.T) {
	assert.PanicsWithValue(t, "Trying to create tensor with negative dimension -1: [-1]", func() {
		torch.RandN([]int64{-1}, torch.NewTensorOptions())
	})
}

// -----------------------------------------------------------------------------
// MARK: RandNLike
// -----------------------------------------------------------------------------

func TestRandNLike(t *testing.T) {
	reference := torch.Rand([]int64{3, 3}, torch.NewTensorOptions())
	tensor := torch.RandNLike(reference)
	assert.NotNil(t, tensor.Pointer)
	assert.Equal(t, reference.Shape(), tensor.Shape())
	assert.False(t, torch.Equal(reference, tensor))
}

func TestRandNLikePanicsOnUninitializedInput(t *testing.T) {
	tensor := torch.Tensor{}
	assert.PanicsWithValue(t, "input tensor is nil", func() {
		torch.RandNLike(tensor)
	})
}

// -----------------------------------------------------------------------------
// MARK: ToSlice
// -----------------------------------------------------------------------------

func TestToSliceUInt8(t *testing.T) {
	if slice, ok := torch.NewTensor([]uint8{2}).ToSlice().([]uint8); ok {
		assert.Equal(t, []uint8{2}, slice)
	} else {
		t.Log("Expected slice to be of type uint8")
		t.Fail()
	}
}

func TestToSliceInt8(t *testing.T) {
	if slice, ok := torch.NewTensor([]int8{2}).ToSlice().([]int8); ok {
		assert.Equal(t, []int8{2}, slice)
	} else {
		t.Log("Expected slice to be of type int8")
		t.Fail()
	}
}

func TestToSliceInt16(t *testing.T) {
	if slice, ok := torch.NewTensor([]int16{2}).ToSlice().([]int16); ok {
		assert.Equal(t, []int16{2}, slice)
	} else {
		t.Log("Expected slice to be of type int16")
		t.Fail()
	}
}

func TestToSliceInt32(t *testing.T) {
	if slice, ok := torch.NewTensor([]int32{2}).ToSlice().([]int32); ok {
		assert.Equal(t, []int32{2}, slice)
	} else {
		t.Log("Expected slice to be of type int32")
		t.Fail()
	}
}

func TestToSliceInt64(t *testing.T) {
	if slice, ok := torch.NewTensor([]int64{2}).ToSlice().([]int64); ok {
		assert.Equal(t, []int64{2}, slice)
	} else {
		t.Log("Expected slice to be of type int64")
		t.Fail()
	}
}

func TestToSliceHalf(t *testing.T) {
	if slice, ok := torch.NewTensor([]uint16{2}).ToSlice().([]uint16); ok {
		assert.Equal(t, []uint16{2}, slice)
	} else {
		t.Log("Expected slice to be of type uint16")
		t.Fail()
	}
}

func TestToSliceFloat(t *testing.T) {
	if slice, ok := torch.NewTensor([]float32{2}).ToSlice().([]float32); ok {
		assert.Equal(t, []float32{2}, slice)
	} else {
		t.Log("Expected slice to be of type float32")
		t.Fail()
	}
}

func TestToSliceDouble(t *testing.T) {
	if slice, ok := torch.NewTensor([]float64{2}).ToSlice().([]float64); ok {
		assert.Equal(t, []float64{2}, slice)
	} else {
		t.Log("Expected slice to be of type float64")
		t.Fail()
	}
}

// func TestToSliceComplexHalf(t *testing.T) {
//     if slice, ok := torch.NewTensor([]uint32{2}).ToSlice().([]uint32); ok {
//         assert.Equal(t, []uint32{2}, slice)
//     } else {
//         t.Log("Expected slice to be of type uint32")
//         t.Fail()
//     }
// }

func TestToSliceComplexFloat(t *testing.T) {
	if slice, ok := torch.NewTensor([]complex64{complex(1, 0.5)}).ToSlice().([]complex64); ok {
		assert.Equal(t, []complex64{complex(1, 0.5)}, slice)
	} else {
		t.Log("Expected slice to be of type complex64")
		t.Fail()
	}
}

func TestToSliceComplexDouble(t *testing.T) {
	if slice, ok := torch.NewTensor([]complex128{complex(1, 0.5)}).ToSlice().([]complex128); ok {
		assert.Equal(t, []complex128{complex(1, 0.5)}, slice)
	} else {
		t.Log("Expected slice to be of type complex128")
		t.Fail()
	}
}

func TestToSliceBool(t *testing.T) {
	if slice, ok := torch.NewTensor([]bool{true, false}).ToSlice().([]bool); ok {
		assert.Equal(t, []bool{true, false}, slice)
	} else {
		t.Log("Expected slice to be of type bool")
		t.Fail()
	}
}

// -----------------------------------------------------------------------------
// MARK: Maths
// -----------------------------------------------------------------------------

// >>> t = torch.tensor([[-0.5, -1.], [1., 0.5]])
// >>> s = torch.tensor([[-0.5, -1.], [1., 0.5]])
// >>> t+s
// tensor([[-1., -2.],
//         [ 2.,  1.]])
func TestArith(t *testing.T) {
	a := assert.New(t)
	r := torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}})
	s := torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}})
	q := r.Add(s, 1)
	expected := torch.NewTensor([][]float32{{-1, -2}, {2, 1}})
	a.True(torch.Equal(expected, q))

	q = r.Sub(s, 1)
	expected = torch.NewTensor([][]float32{{0, 0}, {0, 0}})
	a.True(torch.Equal(expected, q))

	q = r.Mul(s)
	expected = torch.NewTensor([][]float32{{0.25, 1}, {1, 0.25}})
	a.True(torch.Equal(expected, q))

	q = r.Div(s)
	expected = torch.NewTensor([][]float32{{1.0, 1.0}, {1.0, 1.0}})
	a.True(torch.Equal(expected, q))
}

func TestArithI(t *testing.T) {
	a := assert.New(t)

	x := torch.Rand([]int64{2, 3}, torch.NewTensorOptions())
	y := torch.Rand([]int64{2, 3}, torch.NewTensorOptions())
	z := torch.Add(x, y, 1)
	x.Add_(y, 1)
	a.True(torch.Equal(x, z))

	z = torch.Sub(x, y, 1)
	x.Sub_(y, 1)
	a.True(torch.Equal(x, z))

	z = torch.Mul(x, y)
	x.Mul_(y)
	a.True(torch.Equal(x, z))

	z = torch.Div(x, y)
	x.Div_(y)
	a.True(torch.Equal(x, z))
}

func TestAbs(t *testing.T) {
	inputs := torch.NewTensor([][]float32{{-0.5, -1}, {2, -3}})
	expected := torch.NewTensor([][]float32{{0.5, 1}, {2, 3}})
	assert.True(t, torch.Equal(expected, inputs.Abs()))
	assert.False(t, torch.Equal(inputs, inputs.Abs()))
}

func TestAbs_(t *testing.T) {
	inputs := torch.NewTensor([][]float32{{-0.5, -1}, {2, -3}})
	inputs_clone := inputs.Clone()
	inputs.Abs_()
	expected := torch.NewTensor([][]float32{{0.5, 1}, {2, 3}})
	assert.True(t, torch.Equal(expected, inputs))
	assert.False(t, torch.Equal(inputs, inputs_clone))
}

func TestSquare(t *testing.T) {
	inputs := torch.NewTensor([][]float32{{-0.5, -1}, {2, -3}})
	expected := torch.NewTensor([][]float32{{0.25, 1}, {4, 9}})
	assert.True(t, torch.Equal(expected, inputs.Square()))
	assert.False(t, torch.Equal(inputs, inputs.Square()))
}

func TestSquare_(t *testing.T) {
	inputs := torch.NewTensor([][]float32{{-0.5, -1}, {2, -3}})
	inputs_clone := inputs.Clone()
	inputs.Square_()
	expected := torch.NewTensor([][]float32{{0.25, 1}, {4, 9}})
	assert.True(t, torch.Equal(expected, inputs))
	assert.False(t, torch.Equal(inputs, inputs_clone))
}

func TestSqrt(t *testing.T) {
	inputs := torch.NewTensor([][]float32{{0.25, 1}, {4, 9}})
	expected := torch.NewTensor([][]float32{{0.5, 1}, {2, 3}})
	assert.True(t, torch.Equal(expected, inputs.Sqrt()))
}

func TestSqrt_(t *testing.T) {
	inputs := torch.NewTensor([][]float32{{0.25, 1}, {4, 9}})
	inputs_clone := inputs.Clone()
	inputs.Sqrt_()
	expected := torch.NewTensor([][]float32{{0.5, 1}, {2, 3}})
	assert.True(t, torch.Equal(expected, inputs))
	assert.False(t, torch.Equal(inputs, inputs_clone))
}

func TestPow2(t *testing.T) {
	inputs := torch.NewTensor([][]float32{{0.5, 1}, {2, 3}})
	expected := torch.NewTensor([][]float32{{0.25, 1}, {4, 9}})
	assert.True(t, torch.Equal(expected, inputs.Pow(2)))
}

func TestPow0_5(t *testing.T) {
	inputs := torch.NewTensor([][]float32{{0.25, 1}, {4, 9}})
	expected := torch.NewTensor([][]float32{{0.5, 1}, {2, 3}})
	assert.True(t, torch.Equal(expected, inputs.Pow(0.5)))
}

func TestTanh(t *testing.T) {
	a := torch.Rand([]int64{4}, torch.NewTensorOptions())
	b := torch.Tanh(a)
	assert.NotNil(t, b.Pointer)
}

func TestPermute(t *testing.T) {
	a := assert.New(t)
	x := torch.NewTensor([][]float32{{3, 1}, {2, 4}})
	y := x.Permute(1, 0)
	expected := torch.NewTensor([][]float32{{3, 2}, {1, 4}})
	a.True(torch.Equal(expected, y))
}

// >>> torch.nn.functional.log_softmax(torch.tensor([[-0.5, -1.], [1., 0.5]]), dim=1)
// tensor([[-0.4741, -0.9741],
//         [-0.4741, -0.9741]])
func TestLogSoftmax(t *testing.T) {
	tensor := torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}})
	outputs := torch.LogSoftmax(tensor, 1)
	expected := torch.NewTensor([][]float32{{-0.4741, -0.9741}, {-0.4741, -0.9741}})
	assert.True(t, torch.AllClose(expected, outputs, 1e-5, 1e-3), "Got %v expected %v", outputs, expected)
}

// >>> a = torch.tensor([False, False, True, True])
// >>> b = torch.tensor([False, True, False, True])
// >>> torch.logical_and(a, b)
// tensor([False, False, False,  True])
func TestLogicalAnd(t *testing.T) {
	a := torch.NewTensor([]bool{false, false, true, true})
	b := torch.NewTensor([]bool{false, true, false, true})
	outputs := a.LogicalAnd(b)
	expected := torch.NewTensor([]bool{false, false, false, true})
	assert.True(t, torch.Equal(expected, outputs), "Got %v expected %v", outputs, expected)
}

// >>> a = torch.tensor([False, True])
// >>> torch.logical_not(a)
// tensor([ True, False])
func TestLogicalNot(t *testing.T) {
	a := torch.NewTensor([]bool{false, true})
	outputs := a.LogicalNot()
	expected := torch.NewTensor([]bool{true, false})
	assert.True(t, torch.Equal(expected, outputs), "Got %v expected %v", outputs, expected)
}

// >>> a = torch.tensor([False, False, True, True])
// >>> b = torch.tensor([False, True, False, True])
// >>> torch.logical_or(a, b)
// tensor([False,  True,  True,  True])
func TestLogicalOr(t *testing.T) {
	a := torch.NewTensor([]bool{false, false, true, true})
	b := torch.NewTensor([]bool{false, true, false, true})
	outputs := a.LogicalOr(b)
	expected := torch.NewTensor([]bool{false, true, true, true})
	assert.True(t, torch.Equal(expected, outputs), "Got %v expected %v", outputs, expected)
}

// >>> a = torch.tensor([False, False, True, True])
// >>> b = torch.tensor([False, True, False, True])
// >>> torch.logical_xor(a, b)
// tensor([False,  True,  True, False])
func TestLogicalXor(t *testing.T) {
	a := torch.NewTensor([]bool{false, false, true, true})
	b := torch.NewTensor([]bool{false, true, false, true})
	outputs := a.LogicalXor(b)
	expected := torch.NewTensor([]bool{false, true, true, false})
	assert.True(t, torch.Equal(expected, outputs), "Got %v expected %v", outputs, expected)
}

func TestClamp(t *testing.T) {
	tensor := torch.NewTensor([]int32{-2, -1, 0, 1, 2})
	minimum := torch.NewTensor([]int32{-1})
	maximum := torch.NewTensor([]int32{1})
	output := tensor.Clamp(minimum, maximum)
	expected := torch.NewTensor([]int32{-1, -1, 0, 1, 1})
	assert.True(t, expected.Equal(output))
	// The input tensor should not have been modified
	assert.True(t, tensor.Equal(torch.NewTensor([]int32{-2, -1, 0, 1, 2})))
}

func TestClamp_(t *testing.T) {
	tensor := torch.NewTensor([]int32{-2, -1, 0, 1, 2})
	minimum := torch.NewTensor([]int32{-1})
	maximum := torch.NewTensor([]int32{1})
	output := tensor.Clamp_(minimum, maximum)
	expected := torch.NewTensor([]int32{-1, -1, 0, 1, 1})
	assert.True(t, expected.Equal(output))
	// The input tensor should have been updated in-place
	assert.True(t, tensor.Equal(output))
}

func TestClampMax(t *testing.T) {
	tensor := torch.NewTensor([]int32{-2, -1, 0, 1, 2})
	maximum := torch.NewTensor([]int32{1})
	output := tensor.ClampMax(maximum)
	expected := torch.NewTensor([]int32{-2, -1, 0, 1, 1})
	assert.True(t, expected.Equal(output))
	// The input tensor should not have been modified
	assert.True(t, tensor.Equal(torch.NewTensor([]int32{-2, -1, 0, 1, 2})))
}

func TestClampMax_(t *testing.T) {
	tensor := torch.NewTensor([]int32{-2, -1, 0, 1, 2})
	maximum := torch.NewTensor([]int32{1})
	output := tensor.ClampMax_(maximum)
	expected := torch.NewTensor([]int32{-2, -1, 0, 1, 1})
	assert.True(t, expected.Equal(output))
	// The input tensor should have been updated in-place
	assert.True(t, tensor.Equal(output))
}

func TestClampMin(t *testing.T) {
	tensor := torch.NewTensor([]int32{-2, -1, 0, 1, 2})
	minimum := torch.NewTensor([]int32{-1})
	output := tensor.ClampMin(minimum)
	expected := torch.NewTensor([]int32{-1, -1, 0, 1, 2})
	assert.True(t, expected.Equal(output))
	// The input tensor should not have been modified
	assert.True(t, tensor.Equal(torch.NewTensor([]int32{-2, -1, 0, 1, 2})))
}

func TestClampMin_(t *testing.T) {
	tensor := torch.NewTensor([]int32{-2, -1, 0, 1, 2})
	minimum := torch.NewTensor([]int32{-1})
	output := tensor.ClampMin_(minimum)
	expected := torch.NewTensor([]int32{-1, -1, 0, 1, 2})
	assert.True(t, expected.Equal(output))
	// The input tensor should have been updated in-place
	assert.True(t, tensor.Equal(output))
}

// >>> torch.sigmoid(torch.tensor([[-0.5, -1.], [1., 0.5]]))
// tensor([[0.3775, 0.2689],
//         [0.7311, 0.6225]])
func TestSigmoid(t *testing.T) {
	tensor := torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}})
	outputs := torch.Sigmoid(tensor)
	expected := torch.NewTensor([][]float32{{0.3775, 0.2689}, {0.7311, 0.6225}})
	assert.True(t, torch.AllClose(expected, outputs, 1e-5, 1e-3), "Got %v expected %v", outputs, expected)
}

// >>> torch.transpose(torch.tensor([[-0.5, -1.], [1., 0.5]]), 0, 1)
// tensor([[-0.5000,  1.0000],
//         [-1.0000,  0.5000]])
func TestTranspose(t *testing.T) {
	tensor := torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}})
	outputs := tensor.Transpose(0, 1)
	expected := torch.NewTensor([][]float32{{-0.5, 1.0}, {-1.0, 0.5}})
	assert.True(t, torch.Equal(expected, outputs), "Got %v expected %v", outputs, expected)
}

// >>> torch.flatten(torch.tensor([[-0.5, -1.], [1., 0.5]]), 0, 1)
// tensor([-0.5000, -1.0000,  1.0000,  0.5000])
func TestFlatten(t *testing.T) {
	tensor := torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}})
	outputs := tensor.Flatten(0, 1)
	expected := torch.NewTensor([]float32{-0.5, -1.0, 1.0, 0.5})
	assert.True(t, torch.Equal(expected, outputs), "Got %v expected %v", outputs, expected)
}

func TestSqueeze(t *testing.T) {
	tensor := torch.Empty([]int64{2, 1, 2, 1, 2}, torch.NewTensorOptions())
	y := tensor.Squeeze()
	assert.NotNil(t, y.Pointer)
	z := tensor.Squeeze(1)
	assert.NotNil(t, z.Pointer)
	assert.Panics(t, func() { tensor.Squeeze(1, 2) })
}

func TestUnsqueeze(t *testing.T) {
	tensor := torch.Empty([]int64{3}, torch.NewTensorOptions())
	out := tensor.Unsqueeze(0)
	assert.NotNil(t, out.Pointer)
	assert.Equal(t, []int64{1, 3}, out.Shape())
	out = tensor.Unsqueeze(1)
	assert.NotNil(t, out.Pointer)
	assert.Equal(t, []int64{3, 1}, out.Shape())
}

func TestUnsqueezeAllowsNegativeDim(t *testing.T) {
	tensor := torch.Empty([]int64{3}, torch.NewTensorOptions())
	out := tensor.Unsqueeze(-1)
	assert.NotNil(t, out.Pointer)
	assert.Equal(t, []int64{3, 1}, out.Shape())
}

func TestStack(t *testing.T) {
	t1 := torch.Empty([]int64{2, 3}, torch.NewTensorOptions())
	t2 := torch.Empty([]int64{2, 3}, torch.NewTensorOptions())
	out := torch.Stack([]torch.Tensor{t1, t2}, 0)
	assert.Equal(t, []int64{2, 2, 3}, out.Shape())
}

func TestCat(t *testing.T) {
	t1 := torch.Empty([]int64{2, 3}, torch.NewTensorOptions())
	t2 := torch.Empty([]int64{2, 3}, torch.NewTensorOptions())
	out := torch.Cat([]torch.Tensor{t1, t2}, 0)
	assert.Equal(t, []int64{4, 3}, out.Shape())
}

func TestConcat(t *testing.T) {
	t1 := torch.Empty([]int64{2, 3}, torch.NewTensorOptions())
	t2 := torch.Empty([]int64{2, 3}, torch.NewTensorOptions())
	out := torch.Concat([]torch.Tensor{t1, t2}, 0)
	assert.Equal(t, []int64{4, 3}, out.Shape())
}

func TestConcatenate(t *testing.T) {
	t1 := torch.Empty([]int64{2, 3}, torch.NewTensorOptions())
	t2 := torch.Empty([]int64{2, 3}, torch.NewTensorOptions())
	out := torch.Concatenate([]torch.Tensor{t1, t2}, 0)
	assert.Equal(t, []int64{4, 3}, out.Shape())
}

// -----------------------------------------------------------------------------
// MARK: Selection
// -----------------------------------------------------------------------------

func TestSlice(t *testing.T) {
	x := torch.NewTensor([][]float32{
		{1, 2, 3, 4},
		{4, 5, 6, 7},
		{7, 8, 9, 0},
	})

	sliced := x.Slice(0, 1, -1, 1)
	expected := torch.NewTensor([][]float32{
		{4, 5, 6, 7},
	})
	assert.True(t, sliced.Equal(expected), "Expected %v got %v", expected, sliced)

	sliced = x.Slice(1, 1, -1, 1)
	expected = torch.NewTensor([][]float32{
		{2, 3},
		{5, 6},
		{8, 9},
	})
	assert.True(t, sliced.Equal(expected), "Expected %v got %v", expected, sliced)
}

// >>> x = torch.tensor([[1,2,3,4],[4,5,6,7],[7,8,9,0]])
// >>> x
// tensor([[1, 2, 3, 4],
//         [4, 5, 6, 7],
//         [7, 8, 9, 0]])
// >>> idx = torch.tensor([0,2])
// >>> torch.index_select(x, 0, idx)
// tensor([[1, 2, 3, 4],
//         [7, 8, 9, 0]])
// >>> torch.index_select(x, 1, idx)
// tensor([[1, 3],
//         [4, 6],
//         [7, 9]])
func TestIndexSelect(t *testing.T) {
	x := torch.NewTensor([][]float32{{1, 2, 3, 4}, {4, 5, 6, 7}, {7, 8, 9, 0}})
	idx := torch.NewTensor([]int64{0, 2})

	outputs := x.IndexSelect(0, idx)
	expected := torch.NewTensor([][]float32{{1, 2, 3, 4}, {7, 8, 9, 0}})
	assert.True(t, torch.Equal(expected, outputs), "Got %v expected %v", outputs, expected)

	outputs = x.IndexSelect(1, idx)
	expected = torch.NewTensor([][]float32{{1, 3}, {4, 6}, {7, 9}})
	assert.True(t, torch.Equal(expected, outputs), "Got %v expected %v", outputs, expected)
}

// -----------------------------------------------------------------------------
// MARK: Reduction Ops
// -----------------------------------------------------------------------------

func TestArgmax(t *testing.T) {
	x := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
	assert.Equal(t, int64(3), x.Argmax().Item().(int64))
	x = torch.NewTensor([][]float32{{4, 3}, {2, 1}})
	assert.Equal(t, int64(0), x.Argmax().Item().(int64))
}

func TestArgmaxByDim(t *testing.T) {
	// x = torch.tensor([[3,4],[2,1]]
	x := torch.NewTensor([][]float32{{3, 4}, {2, 1}})
	// x.argmax(0)
	expected := torch.NewTensor([]int64{0, 0})
	outputs := x.ArgmaxByDim(0, false)
	assert.True(t, torch.Equal(expected, outputs), "Got %v expected %v", outputs, expected)
	// x.argmax(1)
	expected = torch.NewTensor([]int64{1, 0})
	outputs = x.ArgmaxByDim(1, false)
	assert.True(t, torch.Equal(expected, outputs), "Got %v expected %v", outputs, expected)
	// x.argmax(0, True)
	expected = torch.NewTensor([][]int64{{0, 0}})
	outputs = x.ArgmaxByDim(0, true)
	assert.True(t, torch.Equal(expected, outputs), "Got %v expected %v", outputs, expected)
	// x.argmax(1, True)
	expected = torch.NewTensor([][]int64{{1}, {0}})
	outputs = x.ArgmaxByDim(1, true)
	assert.True(t, torch.Equal(expected, outputs), "Got %v expected %v", outputs, expected)
}

func TestArgmin(t *testing.T) {
	x := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
	assert.Equal(t, int64(0), x.Argmin().Item().(int64))
	x = torch.NewTensor([][]float32{{4, 3}, {2, 1}})
	assert.Equal(t, int64(3), x.Argmin().Item().(int64))
}

func TestArgminByDim(t *testing.T) {
	// x = torch.tensor([[3,4],[2,1]]
	x := torch.NewTensor([][]float32{{3, 4}, {2, 1}})
	// x.argmin(0)
	expected := torch.NewTensor([]int64{1, 1})
	outputs := x.ArgminByDim(0, false)
	assert.True(t, torch.Equal(expected, outputs), "Got %v expected %v", outputs, expected)
	// x.argmin(1)
	expected = torch.NewTensor([]int64{0, 1})
	outputs = x.ArgminByDim(1, false)
	assert.True(t, torch.Equal(expected, outputs), "Got %v expected %v", outputs, expected)
	// x.argmin(0, True)
	expected = torch.NewTensor([][]int64{{1, 1}})
	outputs = x.ArgminByDim(0, true)
	assert.True(t, torch.Equal(expected, outputs), "Got %v expected %v", outputs, expected)
	// x.argmin(1, True)
	expected = torch.NewTensor([][]int64{{0}, {1}})
	outputs = x.ArgminByDim(1, true)
	assert.True(t, torch.Equal(expected, outputs), "Got %v expected %v", outputs, expected)
}

// TODO: amax
// TODO: amin
// TODO: aminmax

// >>> x = torch.tensor([True, True])
// >>> x.all()
// tensor(True)
// >>> x = torch.tensor([True, False])
// >>> x.all()
// tensor(False)
// >>> x = torch.tensor([False, False])
// >>> x.all()
// tensor(False)
func TestAll(t *testing.T) {
	x := torch.NewTensor([]bool{true, true})
	assert.Equal(t, true, x.All().Item().(bool))
	x = torch.NewTensor([]bool{true, false})
	assert.Equal(t, false, x.All().Item().(bool))
	x = torch.NewTensor([]bool{false, false})
	assert.Equal(t, false, x.All().Item().(bool))
}

// >>> x = torch.tensor([[1,1,1,1],[1,0,0,0],[1,0,0,0]]).bool()
// >>> x.all(0, False)
// tensor([ True, False, False, False])
// >>> x.all(0, True)
// tensor([[ True, False, False, False]])
// >>> x.all(1, False)
// tensor([ True, False, False])
// >>> x.all(1, True)
// tensor([[ True],
//         [False],
//         [False]])
func TestAllByDim(t *testing.T) {
	x := torch.NewTensor([][]bool{
		{true, true, true, true },
		{true, false,false,false},
		{true, false,false,false},
	})

	y := x.AllByDim(0, false)
	expected := torch.NewTensor([]bool{true, false, false, false})
	assert.True(t, torch.Equal(expected, y), "Got %v", y)

	y = x.AllByDim(0, true)
	expected = torch.NewTensor([][]bool{{true, false, false, false}})
	assert.True(t, torch.Equal(expected, y), "Got %v", y)

	y = x.AllByDim(1, false)
	expected = torch.NewTensor([]bool{true, false, false})
	assert.True(t, torch.Equal(expected, y), "Got %v", y)

	y = x.AllByDim(1, true)
	expected = torch.NewTensor([][]bool{{true}, {false}, {false}})
	assert.True(t, torch.Equal(expected, y), "Got %v", y)
}

// >>> x = torch.tensor([True, True])
// >>> x.any()
// tensor(True)
// >>> x = torch.tensor([True, False])
// >>> x.any()
// tensor(True)
// >>> x = torch.tensor([False, False])
// >>> x.any()
// tensor(False)
func TestAny(t *testing.T) {
	x := torch.NewTensor([]bool{true, true})
	assert.Equal(t, true, x.Any().Item().(bool))
	x = torch.NewTensor([]bool{true, false})
	assert.Equal(t, true, x.Any().Item().(bool))
	x = torch.NewTensor([]bool{false, false})
	assert.Equal(t, false, x.Any().Item().(bool))
}

// >>> x = torch.tensor([[1,0,0,0],[0,0,0,0],[0,0,0,0]]).bool()
// >>> x.any(0, False)
// tensor([ True, False, False, False])
// >>> x.any(0, True)
// tensor([[ True, False, False, False]])
// >>> x.any(1, False)
// tensor([ True, False, False])
// >>> x.any(1, True)
// tensor([[ True],
//         [False],
//         [False]])
func TestAnyByDim(t *testing.T) {
	x := torch.NewTensor([][]bool{
		{true, false,false,false},
		{false,false,false,false},
		{false,false,false,false},
	})

	y := x.AnyByDim(0, false)
	expected := torch.NewTensor([]bool{true, false, false, false})
	assert.True(t, torch.Equal(expected, y), "Got %v", y)

	y = x.AnyByDim(0, true)
	expected = torch.NewTensor([][]bool{{true, false, false, false}})
	assert.True(t, torch.Equal(expected, y), "Got %v", y)

	y = x.AnyByDim(1, false)
	expected = torch.NewTensor([]bool{true, false, false})
	assert.True(t, torch.Equal(expected, y), "Got %v", y)

	y = x.AnyByDim(1, true)
	expected = torch.NewTensor([][]bool{{true}, {false}, {false}})
	assert.True(t, torch.Equal(expected, y), "Got %v", y)
}

// TODO: dist
// TODO: logsumexp

func TestMax(t *testing.T) {
	// >>> x = torch.tensor([1, 3, 3, 6, 7, 8, 8, 9], dtype=float)
	// >>> x.max()
	// tensor(9., dtype=torch.float64)
	x := torch.NewTensor([]float32{1, 3, 3, 6, 7, 8, 8, 9})
	assert.InEpsilon(t, float32(9), x.Max().Item().(float32), 1e-3)
}

func TestMaxByDim(t *testing.T) {
	// >>> x = torch.arange(12).reshape((3, 4)).float()
	x := torch.Arange(0, 12, 1, torch.NewTensorOptions()).View(3, 4).CastTo(torch.Float)

	// >>> torch.max(x, 0, False)
	// torch.return_types.max(
	// values=tensor([ 8.,  9., 10., 11.]),
	// indices=tensor([2, 2, 2, 2]))
	expected_values := torch.NewTensor([]float32{8.,  9., 10., 11.})
	expected_indices := torch.NewTensor([]int64{2, 2, 2, 2})
	outputs := x.MaxByDim(0, false)
	assert.True(t, torch.AllClose(expected_values, outputs.Values, 1e-5, 1e-3), "Got %v expected %v", outputs.Values, expected_values)
	assert.True(t, torch.AllClose(expected_indices, outputs.Indices, 1e-5, 1e-3), "Got %v expected %v", outputs.Indices, expected_indices)
	// >>> torch.max(x, 0, True)
	// torch.return_types.max(
	// values=tensor([[ 8.,  9., 10., 11.]]),
	// indices=tensor([[2, 2, 2, 2]]))
	expected_values = torch.NewTensor([][]float32{{8.,  9., 10., 11.}})
	expected_indices = torch.NewTensor([][]int64{{2, 2, 2, 2}})
	outputs = x.MaxByDim(0, true)
	assert.True(t, torch.AllClose(expected_values, outputs.Values, 1e-5, 1e-3), "Got %v expected %v", outputs.Values, expected_values)
	assert.True(t, torch.AllClose(expected_indices, outputs.Indices, 1e-5, 1e-3), "Got %v expected %v", outputs.Indices, expected_indices)

	// >>> torch.max(x, 1, False)
	// torch.return_types.max(
	// values=tensor([ 3.,  7., 11.]),
	// indices=tensor([3, 3, 3]))
	expected_values = torch.NewTensor([]float32{3.,  7., 11.})
	expected_indices = torch.NewTensor([]int64{3, 3, 3})
	outputs = x.MaxByDim(1, false)
	assert.True(t, torch.AllClose(expected_values, outputs.Values, 1e-5, 1e-3), "Got %v expected %v", outputs.Values, expected_values)
	assert.True(t, torch.AllClose(expected_indices, outputs.Indices, 1e-5, 1e-3), "Got %v expected %v", outputs.Indices, expected_indices)
	// >>> torch.max(x, 1, True)
	// torch.return_types.max(
	// values=tensor([[ 3.],
	//         [ 7.],
	//         [11.]]),
	// indices=tensor([[3],
	//         [3],
	//         [3]]))
	expected_values = torch.NewTensor([][]float32{{3.}, {7.}, {11.}})
	expected_indices = torch.NewTensor([][]int64{{3}, {3}, {3}})
	outputs = x.MaxByDim(1, true)
	assert.True(t, torch.AllClose(expected_values, outputs.Values, 1e-5, 1e-3), "Got %v expected %v", outputs.Values, expected_values)
	assert.True(t, torch.AllClose(expected_indices, outputs.Indices, 1e-5, 1e-3), "Got %v expected %v", outputs.Indices, expected_indices)
}

func TestMin(t *testing.T) {
	// >>> x = torch.tensor([1, 3, 3, 6, 7, 8, 8, 9], dtype=float)
	// >>> x.min()
	// tensor(1., dtype=torch.float64)
	x := torch.NewTensor([]float32{1, 3, 3, 6, 7, 8, 8, 9})
	assert.InEpsilon(t, float32(1), x.Min().Item().(float32), 1e-3)
}

func TestMinByDim(t *testing.T) {
	// >>> x = torch.arange(12).reshape((3, 4)).float()
	x := torch.Arange(0, 12, 1, torch.NewTensorOptions()).View(3, 4).CastTo(torch.Float)

	// >>> torch.min(x, 0, False)
	// torch.return_types.min(
	// values=tensor([0., 1., 2., 3.]),
	// indices=tensor([0, 0, 0, 0]))
	expected_values := torch.NewTensor([]float32{0., 1., 2., 3.})
	expected_indices := torch.NewTensor([]int64{0, 0, 0, 0})
	outputs := x.MinByDim(0, false)
	assert.True(t, torch.AllClose(expected_values, outputs.Values, 1e-5, 1e-3), "Got %v expected %v", outputs.Values, expected_values)
	assert.True(t, torch.AllClose(expected_indices, outputs.Indices, 1e-5, 1e-3), "Got %v expected %v", outputs.Indices, expected_indices)
	// >>> torch.min(x, 0, True)
	// torch.return_types.min(
	// values=tensor([[0., 1., 2., 3.]]),
	// indices=tensor([[0, 0, 0, 0]]))
	expected_values = torch.NewTensor([][]float32{{0., 1., 2., 3.}})
	expected_indices = torch.NewTensor([][]int64{{0, 0, 0, 0}})
	outputs = x.MinByDim(0, true)
	assert.True(t, torch.AllClose(expected_values, outputs.Values, 1e-5, 1e-3), "Got %v expected %v", outputs.Values, expected_values)
	assert.True(t, torch.AllClose(expected_indices, outputs.Indices, 1e-5, 1e-3), "Got %v expected %v", outputs.Indices, expected_indices)

	// >>> torch.min(x, 1, False)
	// torch.return_types.min(
	// values=tensor([0., 4., 8.]),
	// indices=tensor([0, 0, 0]))
	expected_values = torch.NewTensor([]float32{0., 4., 8.})
	expected_indices = torch.NewTensor([]int64{0, 0, 0})
	outputs = x.MinByDim(1, false)
	assert.True(t, torch.AllClose(expected_values, outputs.Values, 1e-5, 1e-3), "Got %v expected %v", outputs.Values, expected_values)
	assert.True(t, torch.AllClose(expected_indices, outputs.Indices, 1e-5, 1e-3), "Got %v expected %v", outputs.Indices, expected_indices)
	// >>> torch.min(x, 1, True)
	// torch.return_types.min(
	// values=tensor([[0.],
	//         [4.],
	//         [8.]]),
	// indices=tensor([[0],
	//         [0],
	//         [0]]))
	expected_values = torch.NewTensor([][]float32{{0.}, {4.}, {8.}})
	expected_indices = torch.NewTensor([][]int64{{0}, {0}, {0}})
	outputs = x.MinByDim(1, true)
	assert.True(t, torch.AllClose(expected_values, outputs.Values, 1e-5, 1e-3), "Got %v expected %v", outputs.Values, expected_values)
	assert.True(t, torch.AllClose(expected_indices, outputs.Indices, 1e-5, 1e-3), "Got %v expected %v", outputs.Indices, expected_indices)
}

// >>> x = torch.tensor([1, 2, 3], dtype=float)
// >>> x.mean()
// tensor(2.0000, dtype=torch.float64)
func TestMean(t *testing.T) {
	x := torch.NewTensor([]float32{1, 2, 3})
	assert.Equal(t, float32(2), x.Mean().Item().(float32))
}

// >>> x = torch.tensor([[1,4,1,1],[4,1,1,1],[1,1,1,4]], dtype=float)
// >>> x.mean(0, False)
// tensor([1.6667, 1.6667, 1.0000, 1.6667], dtype=torch.float64)
// >>> x.mean(0, True)
// tensor([[1.6667, 1.6667, 1.0000, 1.6667]], dtype=torch.float64)
// >>> x.mean(1, False)
// tensor([1.5000, 1.5000, 1.5000], dtype=torch.float64)
// >>> x.mean(1, True)
// tensor([[1.5000],
//         [1.5000],
//         [1.5000]], dtype=torch.float64)
func TestMeanByDim(t *testing.T) {
	x := torch.NewTensor([][]float32{
		{1, 4, 1, 1},
		{4, 1, 1, 1},
		{1, 1, 1, 4},
	})

	y := x.MeanByDim(0, false)
	expected := torch.NewTensor([]float32{2., 2., 1., 2.})
	assert.True(t, torch.Equal(expected, y), "Got %v", y)

	y = x.MeanByDim(0, true)
	expected = torch.NewTensor([][]float32{{2., 2., 1., 2.}})
	assert.True(t, torch.Equal(expected, y), "Got %v", y)

	y = x.MeanByDim(1, false)
	expected = torch.NewTensor([]float32{1.7500, 1.7500, 1.7500})
	assert.True(t, torch.Equal(expected, y), "Got %v", y)

	y = x.MeanByDim(1, true)
	expected = torch.NewTensor([][]float32{{1.7500}, {1.7500}, {1.7500}})
	assert.True(t, torch.Equal(expected, y), "Got %v", y)
}

// TODO: nanmean

func TestMedian(t *testing.T) {
	// >>> x = torch.tensor([1, 3, 3, 6, 7, 8, 8, 9], dtype=float)
	// >>> x.median()
	// tensor(6., dtype=torch.float64)
	x := torch.NewTensor([]float32{1, 3, 3, 6, 7, 8, 8, 9})
	assert.InEpsilon(t, float32(6), x.Median().Item().(float32), 1e-3)
}

func TestMedianByDim(t *testing.T) {
	// >>> x = torch.arange(12).reshape((3, 4)).float()
	x := torch.Arange(0, 12, 1, torch.NewTensorOptions()).View(3, 4).CastTo(torch.Float)

	// >>> torch.median(x, 0, False)
	// torch.return_types.median(
	// values=tensor([4., 5., 6., 7.]),
	// indices=tensor([1, 1, 1, 1]))
	expected_values := torch.NewTensor([]float32{4., 5., 6., 7.})
	expected_indices := torch.NewTensor([]int64{1, 1, 1, 1})
	outputs := x.MedianByDim(0, false)
	assert.True(t, torch.AllClose(expected_values, outputs.Values, 1e-5, 1e-3), "Got %v expected %v", outputs.Values, expected_values)
	assert.True(t, torch.AllClose(expected_indices, outputs.Indices, 1e-5, 1e-3), "Got %v expected %v", outputs.Indices, expected_indices)
	// >>> torch.median(x, 0, True)
	// torch.return_types.median(
	// values=tensor([[4., 5., 6., 7.]]),
	// indices=tensor([[1, 1, 1, 1]]))
	expected_values = torch.NewTensor([][]float32{{4., 5., 6., 7.}})
	expected_indices = torch.NewTensor([][]int64{{1, 1, 1, 1}})
	outputs = x.MedianByDim(0, true)
	assert.True(t, torch.AllClose(expected_values, outputs.Values, 1e-5, 1e-3), "Got %v expected %v", outputs.Values, expected_values)
	assert.True(t, torch.AllClose(expected_indices, outputs.Indices, 1e-5, 1e-3), "Got %v expected %v", outputs.Indices, expected_indices)

	// >>> torch.median(x, 1, False)
	// torch.return_types.median(
	// values=tensor([1., 5., 9.]),
	// indices=tensor([1, 1, 1]))
	expected_values = torch.NewTensor([]float32{1., 5., 9.})
	expected_indices = torch.NewTensor([]int64{1, 1, 1})
	outputs = x.MedianByDim(1, false)
	assert.True(t, torch.AllClose(expected_values, outputs.Values, 1e-5, 1e-3), "Got %v expected %v", outputs.Values, expected_values)
	assert.True(t, torch.AllClose(expected_indices, outputs.Indices, 1e-5, 1e-3), "Got %v expected %v", outputs.Indices, expected_indices)
	// >>> torch.median(x, 1, True)
	// torch.return_types.median(
	// values=tensor([[1.],
	//         [5.],
	//         [9.]]),
	// indices=tensor([[1],
	//         [1],
	//         [1]]))
	expected_values = torch.NewTensor([][]float32{{1.}, {5.}, {9.}})
	expected_indices = torch.NewTensor([][]int64{{1}, {1}, {1}})
	outputs = x.MedianByDim(1, true)
	assert.True(t, torch.AllClose(expected_values, outputs.Values, 1e-5, 1e-3), "Got %v expected %v", outputs.Values, expected_values)
	assert.True(t, torch.AllClose(expected_indices, outputs.Indices, 1e-5, 1e-3), "Got %v expected %v", outputs.Indices, expected_indices)
}

// TODO: nanmedian
// TODO: mode
// TODO: norm
// TODO: nansum
// TODO: prod
// TODO: quantile
// TODO: nanquantile

func TestStd(t *testing.T) {
	// >>> x = torch.tensor([1, 2, 3], dtype=float)
	// >>> x.std()
	// tensor(1.0000, dtype=torch.float64)
	x := torch.NewTensor([]float32{1, 2, 3})
	assert.Equal(t, float32(1), x.Std().Item().(float32))
}

func TestStdByDim(t *testing.T) {
	// >>> x = torch.arange(12).reshape((3, 4)).float()
	x := torch.Arange(0, 12, 1, torch.NewTensorOptions()).View(3, 4).CastTo(torch.Float)

	// >>> x.std(0, False, False)
	// tensor([3.2660, 3.2660, 3.2660, 3.2660], dtype=torch.float64)
	y := x.StdByDim(0, false, false)
	expected := torch.NewTensor([]float32{3.2660, 3.2660, 3.2660, 3.2660})
	assert.True(t, torch.AllClose(expected, y, 1e-5, 1e-3), "Got %v expected %v", y, expected)
	// >>> x.std(0, False, True)
	// tensor([[3.2660, 3.2660, 3.2660, 3.2660]], dtype=torch.float64)
	y = x.StdByDim(0, false, true)
	expected = torch.NewTensor([][]float32{{3.2660, 3.2660, 3.2660, 3.2660}})
	assert.True(t, torch.AllClose(expected, y, 1e-5, 1e-3), "Got %v expected %v", y, expected)
	// >>> x.std(0, True, False)
	// tensor([4., 4., 4., 4.])
	y = x.StdByDim(0, true, false)
	expected = torch.NewTensor([]float32{4., 4., 4., 4.})
	assert.True(t, torch.AllClose(expected, y, 1e-5, 1e-3), "Got %v expected %v", y, expected)
	// >>> x.std(0, True, True)
	// tensor([[4., 4., 4., 4.]])
	y = x.StdByDim(0, true, true)
	expected = torch.NewTensor([][]float32{{4., 4., 4., 4.}})
	assert.True(t, torch.AllClose(expected, y, 1e-5, 1e-3), "Got %v expected %v", y, expected)

	// >>> x.std(1, False, False)
	// tensor([1.1180, 1.1180, 1.1180])
	y = x.StdByDim(1, false, false)
	expected = torch.NewTensor([]float32{1.1180, 1.1180, 1.1180})
	assert.True(t, torch.AllClose(expected, y, 1e-5, 1e-3), "Got %v expected %v", y, expected)
	// >>> x.std(1, False, True)
	// tensor([[1.1180],
	//         [1.1180],
	//         [1.1180]])
	y = x.StdByDim(1, false, true)
	expected = torch.NewTensor([][]float32{{1.1180}, {1.1180}, {1.1180}})
	assert.True(t, torch.AllClose(expected, y, 1e-5, 1e-3), "Got %v expected %v", y, expected)
	// >>> x.std(1, True, False)
	// tensor([1.2910, 1.2910, 1.2910])
	y = x.StdByDim(1, true, false)
	expected = torch.NewTensor([]float32{1.2910, 1.2910, 1.2910})
	assert.True(t, torch.AllClose(expected, y, 1e-5, 1e-3), "Got %v expected %v", y, expected)
	// >>> x.std(1, True, True)
	// tensor([[1.2910],
	//         [1.2910],
	//         [1.2910]])
	y = x.StdByDim(1, true, true)
	expected = torch.NewTensor([][]float32{{1.2910}, {1.2910}, {1.2910}})
	assert.True(t, torch.AllClose(expected, y, 1e-5, 1e-3), "Got %v expected %v", y, expected)
}


func TestStdMean(t *testing.T) {
	// >>> x = torch.tensor([1, 2, 3], dtype=float)
	// >>> torch.std_mean(x)
	// (tensor(1., dtype=torch.float64), tensor(2., dtype=torch.float64))
	x := torch.NewTensor([]float32{1, 2, 3})
	std, mean := x.StdMean()
	assert.Equal(t, float32(1), std.Item().(float32))
	assert.Equal(t, float32(2), mean.Item().(float32))
}

func TestStdMeanByDim(t *testing.T) {
	// >>> x = torch.arange(12).reshape((3, 4)).float()
	x := torch.Arange(0, 12, 1, torch.NewTensorOptions()).View(3, 4).CastTo(torch.Float)

	// >>> torch.std_mean(x, 0, False, False)
	// (tensor([3.2660, 3.2660, 3.2660, 3.2660]), tensor([4., 5., 6., 7.]))
	expected_std := torch.NewTensor([]float32{3.2660, 3.2660, 3.2660, 3.2660})
	expected_mean := torch.NewTensor([]float32{4., 5., 6., 7.})
	std, mean := x.StdMeanByDim(0, false, false)
	assert.True(t, torch.AllClose(expected_std, std, 1e-5, 1e-3), "Got %v expected %v", std, expected_std)
	assert.True(t, torch.AllClose(expected_mean, mean, 1e-5, 1e-3), "Got %v expected %v", mean, expected_mean)
	// >>> torch.std_mean(x, 0, False, True)
	// (tensor([[3.2660, 3.2660, 3.2660, 3.2660]]), tensor([[4., 5., 6., 7.]]))
	expected_std = torch.NewTensor([][]float32{{3.2660, 3.2660, 3.2660, 3.2660}})
	expected_mean = torch.NewTensor([][]float32{{4., 5., 6., 7.}})
	std, mean = x.StdMeanByDim(0, false, true)
	assert.True(t, torch.AllClose(expected_std, std, 1e-5, 1e-3), "Got %v expected %v", std, expected_std)
	assert.True(t, torch.AllClose(expected_mean, mean, 1e-5, 1e-3), "Got %v expected %v", mean, expected_mean)
	// >>> torch.std_mean(x, 0, True, False)
	// (tensor([4., 4., 4., 4.]), tensor([4., 5., 6., 7.]))
	std, mean = x.StdMeanByDim(0, true, false)
	expected_std = torch.NewTensor([]float32{4., 4., 4., 4.})
	expected_mean = torch.NewTensor([]float32{4., 5., 6., 7.})
	assert.True(t, torch.AllClose(expected_std, std, 1e-5, 1e-3), "Got %v expected %v", std, expected_std)
	assert.True(t, torch.AllClose(expected_mean, mean, 1e-5, 1e-3), "Got %v expected %v", mean, expected_mean)
	// >>> torch.std_mean(x, 0, True, True)
	// (tensor([[4., 4., 4., 4.]]), tensor([[4., 5., 6., 7.]]))
	expected_std = torch.NewTensor([][]float32{{4., 4., 4., 4.}})
	expected_mean = torch.NewTensor([][]float32{{4., 5., 6., 7.}})
	std, mean = x.StdMeanByDim(0, true, true)
	assert.True(t, torch.AllClose(expected_std, std, 1e-5, 1e-3), "Got %v expected %v", std, expected_std)
	assert.True(t, torch.AllClose(expected_mean, mean, 1e-5, 1e-3), "Got %v expected %v", mean, expected_mean)

	// >>> torch.std_mean(x, 1, False, False)
	// (tensor([1.1180, 1.1180, 1.1180]), tensor([1.5000, 5.5000, 9.5000]))
	expected_std = torch.NewTensor([]float32{1.1180, 1.1180, 1.1180})
	expected_mean = torch.NewTensor([]float32{1.5000, 5.5000, 9.5000})
	std, mean = x.StdMeanByDim(1, false, false)
	assert.True(t, torch.AllClose(expected_std, std, 1e-5, 1e-3), "Got %v expected %v", std, expected_std)
	assert.True(t, torch.AllClose(expected_mean, mean, 1e-5, 1e-3), "Got %v expected %v", mean, expected_mean)
	// >>> torch.std_mean(x, 1, False, True)
	// (tensor([[1.1180],
	//         [1.1180],
	//         [1.1180]]), tensor([[1.5000],
	//         [5.5000],
	//         [9.5000]]))
	expected_std = torch.NewTensor([][]float32{{1.1180}, {1.1180}, {1.1180}})
	expected_mean = torch.NewTensor([][]float32{{1.5000}, {5.5000}, {9.5000}})
	std, mean = x.StdMeanByDim(1, false, true)
	assert.True(t, torch.AllClose(expected_std, std, 1e-5, 1e-3), "Got %v expected %v", std, expected_std)
	assert.True(t, torch.AllClose(expected_mean, mean, 1e-5, 1e-3), "Got %v expected %v", mean, expected_mean)
	// >>> torch.std_mean(x, 1, True, False)
	// (tensor([1.2910, 1.2910, 1.2910]), tensor([1.5000, 5.5000, 9.5000]))
	expected_std = torch.NewTensor([]float32{1.2910, 1.2910, 1.2910})
	expected_mean = torch.NewTensor([]float32{1.5000, 5.5000, 9.5000})
	std, mean = x.StdMeanByDim(1, true, false)
	assert.True(t, torch.AllClose(expected_std, std, 1e-5, 1e-3), "Got %v expected %v", std, expected_std)
	assert.True(t, torch.AllClose(expected_mean, mean, 1e-5, 1e-3), "Got %v expected %v", mean, expected_mean)
	// >>> torch.std_mean(x, 1, True, True)
	// (tensor([[1.2910],
	//         [1.2910],
	//         [1.2910]]), tensor([[1.5000],
	//         [5.5000],
	//         [9.5000]]))
	expected_std = torch.NewTensor([][]float32{{1.2910}, {1.2910}, {1.2910}})
	expected_mean = torch.NewTensor([][]float32{{1.5000}, {5.5000}, {9.5000}})
	std, mean = x.StdMeanByDim(1, true, true)
	assert.True(t, torch.AllClose(expected_std, std, 1e-5, 1e-3), "Got %v expected %v", std, expected_std)
	assert.True(t, torch.AllClose(expected_mean, mean, 1e-5, 1e-3), "Got %v expected %v", mean, expected_mean)
}

// >>> x = torch.tensor([1, 2, 3])
// >>> x.sum()
// tensor(6)
func TestSum(t *testing.T) {
	x := torch.NewTensor([]float32{1, 2, 3})
	assert.Equal(t, float32(6), x.Sum().Item().(float32))
}

// >>> x = torch.tensor([[1,2,3,4],[4,5,6,7],[7,8,9,0]])
// >>> x.sum(0, False)
// tensor([12, 15, 18, 11])
// >>> x.sum(0, True)
// tensor([[12, 15, 18, 11]])
// >>> x.sum(1, False)
// tensor([10, 22, 24])
// >>> x.sum(1, True)
// tensor([[10],
//         [22],
//         [24]])
func TestSumByDim(t *testing.T) {
	x := torch.NewTensor([][]float32{
		{1, 2, 3, 4},
		{4, 5, 6, 7},
		{7, 8, 9, 0},
	})

	y := x.SumByDim(0, false)
	expected := torch.NewTensor([]float32{12, 15, 18, 11})
	assert.True(t, torch.Equal(expected, y), "Got %v", y)

	y = x.SumByDim(0, true)
	expected = torch.NewTensor([][]float32{{12, 15, 18, 11}})
	assert.True(t, torch.Equal(expected, y), "Got %v", y)

	y = x.SumByDim(1, false)
	expected = torch.NewTensor([]float32{10, 22, 24})
	assert.True(t, torch.Equal(expected, y), "Got %v", y)

	y = x.SumByDim(1, true)
	expected = torch.NewTensor([][]float32{{10}, {22}, {24}})
	assert.True(t, torch.Equal(expected, y), "Got %v", y)
}

// TODO: unique
// TODO: unique_consecutive

func TestVar(t *testing.T) {
	// >>> x = torch.tensor([1, 2, 3], dtype=float)
	// >>> x.var()
	// tensor(1.0000, dtype=torch.float64)
	x := torch.NewTensor([]float32{1, 2, 3})
	assert.Equal(t, float32(1), x.Var().Item().(float32))
}

func TestVarByDim(t *testing.T) {
	// >>> x = torch.arange(12).reshape((3, 4)).float()
	x := torch.Arange(0, 12, 1, torch.NewTensorOptions()).View(3, 4).CastTo(torch.Float)

	// >>> x.var(0, False, False)
	// tensor([10.6667, 10.6667, 10.6667, 10.6667])
	y := x.VarByDim(0, false, false)
	expected := torch.NewTensor([]float32{10.6667, 10.6667, 10.6667, 10.6667})
	assert.True(t, torch.AllClose(expected, y, 1e-5, 1e-3), "Got %v expected %v", y, expected)
	// >>> x.var(0, False, True)
	// tensor([[10.6667, 10.6667, 10.6667, 10.6667]])
	y = x.VarByDim(0, false, true)
	expected = torch.NewTensor([][]float32{{10.6667, 10.6667, 10.6667, 10.6667}})
	assert.True(t, torch.AllClose(expected, y, 1e-5, 1e-3), "Got %v expected %v", y, expected)
	// >>> x.var(0, True, False)
	// tensor([16., 16., 16., 16.])
	y = x.VarByDim(0, true, false)
	expected = torch.NewTensor([]float32{16., 16., 16., 16.})
	assert.True(t, torch.AllClose(expected, y, 1e-5, 1e-3), "Got %v expected %v", y, expected)
	// >>> x.var(0, True, True)
	// tensor([[16., 16., 16., 16.]])
	y = x.VarByDim(0, true, true)
	expected = torch.NewTensor([][]float32{{16., 16., 16., 16.}})
	assert.True(t, torch.AllClose(expected, y, 1e-5, 1e-3), "Got %v expected %v", y, expected)

	// >>> x.var(1, False, False)
	// tensor([1.2500, 1.2500, 1.2500])
	y = x.VarByDim(1, false, false)
	expected = torch.NewTensor([]float32{1.2500, 1.2500, 1.2500})
	assert.True(t, torch.AllClose(expected, y, 1e-5, 1e-3), "Got %v expected %v", y, expected)
	// >>> x.var(1, False, True)
	// tensor([[1.2500],
	//         [1.2500],
	//         [1.2500]])
	y = x.VarByDim(1, false, true)
	expected = torch.NewTensor([][]float32{{1.2500}, {1.2500}, {1.2500}})
	assert.True(t, torch.AllClose(expected, y, 1e-5, 1e-3), "Got %v expected %v", y, expected)
	// >>> x.var(1, True, False)
	// tensor([1.6667, 1.6667, 1.6667])
	y = x.VarByDim(1, true, false)
	expected = torch.NewTensor([]float32{1.6667, 1.6667, 1.6667})
	assert.True(t, torch.AllClose(expected, y, 1e-5, 1e-3), "Got %v expected %v", y, expected)
	// >>> x.var(1, True, True)
	// tensor([[1.6667],
	//         [1.6667],
	//         [1.6667]])
	y = x.VarByDim(1, true, true)
	expected = torch.NewTensor([][]float32{{1.6667}, {1.6667}, {1.6667}})
	assert.True(t, torch.AllClose(expected, y, 1e-5, 1e-3), "Got %v expected %v", y, expected)
}

func TestVarMean(t *testing.T) {
	// >>> x = torch.tensor([6, 1, 2, 3], dtype=float)
	// >>> torch.var_mean(x)
	// (tensor(4.6667, dtype=torch.float64), tensor(3., dtype=torch.float64))
	x := torch.NewTensor([]float32{6, 1, 2, 3})
	var_, mean := x.VarMean()
	assert.InEpsilon(t, float32(4.6667), var_.Item().(float32), 1e-3)
	assert.InEpsilon(t, float32(3), mean.Item().(float32), 1e-3)
}

func TestVarMeanByDim(t *testing.T) {
	// >>> x = torch.arange(12).reshape((3, 4)).float()
	x := torch.Arange(0, 12, 1, torch.NewTensorOptions()).View(3, 4).CastTo(torch.Float)

	// >>> torch.var_mean(x, 0, False, False)
	// (tensor([10.6667, 10.6667, 10.6667, 10.6667]), tensor([4., 5., 6., 7.]))
	expected_var := torch.NewTensor([]float32{10.6667, 10.6667, 10.6667, 10.6667})
	expected_mean := torch.NewTensor([]float32{4., 5., 6., 7.})
	variance, mean := x.VarMeanByDim(0, false, false)
	assert.True(t, torch.AllClose(expected_var, variance, 1e-5, 1e-3), "Got %v expected %v", variance, expected_var)
	assert.True(t, torch.AllClose(expected_mean, mean, 1e-5, 1e-3), "Got %v expected %v", mean, expected_mean)
	// >>> torch.var_mean(x, 0, False, True)
	// (tensor([[10.6667, 10.6667, 10.6667, 10.6667]]), tensor([[4., 5., 6., 7.]]))
	expected_var = torch.NewTensor([][]float32{{10.6667, 10.6667, 10.6667, 10.6667}})
	expected_mean = torch.NewTensor([][]float32{{4., 5., 6., 7.}})
	variance, mean = x.VarMeanByDim(0, false, true)
	assert.True(t, torch.AllClose(expected_var, variance, 1e-5, 1e-3), "Got %v expected %v", variance, expected_var)
	assert.True(t, torch.AllClose(expected_mean, mean, 1e-5, 1e-3), "Got %v expected %v", mean, expected_mean)
	// >>> torch.var_mean(x, 0, True, False)
	// (tensor([16., 16., 16., 16.]), tensor([4., 5., 6., 7.]))
	variance, mean = x.VarMeanByDim(0, true, false)
	expected_var = torch.NewTensor([]float32{16., 16., 16., 16.})
	expected_mean = torch.NewTensor([]float32{4., 5., 6., 7.})
	assert.True(t, torch.AllClose(expected_var, variance, 1e-5, 1e-3), "Got %v expected %v", variance, expected_var)
	assert.True(t, torch.AllClose(expected_mean, mean, 1e-5, 1e-3), "Got %v expected %v", mean, expected_mean)
	// >>> torch.var_mean(x, 0, True, True)
	// (tensor([[16., 16., 16., 16.]]), tensor([[4., 5., 6., 7.]]))
	expected_var = torch.NewTensor([][]float32{{16., 16., 16., 16.}})
	expected_mean = torch.NewTensor([][]float32{{4., 5., 6., 7.}})
	variance, mean = x.VarMeanByDim(0, true, true)
	assert.True(t, torch.AllClose(expected_var, variance, 1e-5, 1e-3), "Got %v expected %v", variance, expected_var)
	assert.True(t, torch.AllClose(expected_mean, mean, 1e-5, 1e-3), "Got %v expected %v", mean, expected_mean)

	// >>> torch.var_mean(x, 1, False, False)
	// (tensor([1.2500, 1.2500, 1.2500]), tensor([1.5000, 5.5000, 9.5000]))
	expected_var = torch.NewTensor([]float32{1.2500, 1.2500, 1.2500})
	expected_mean = torch.NewTensor([]float32{1.5000, 5.5000, 9.5000})
	variance, mean = x.VarMeanByDim(1, false, false)
	assert.True(t, torch.AllClose(expected_var, variance, 1e-5, 1e-3), "Got %v expected %v", variance, expected_var)
	assert.True(t, torch.AllClose(expected_mean, mean, 1e-5, 1e-3), "Got %v expected %v", mean, expected_mean)
	// >>> torch.var_mean(x, 1, False, True)
	// (tensor([[1.2500],
	//         [1.2500],
	//         [1.2500]]), tensor([[1.5000],
	//         [5.5000],
	//         [9.5000]]))
	expected_var = torch.NewTensor([][]float32{{1.2500}, {1.2500}, {1.2500}})
	expected_mean = torch.NewTensor([][]float32{{1.5000}, {5.5000}, {9.5000}})
	variance, mean = x.VarMeanByDim(1, false, true)
	assert.True(t, torch.AllClose(expected_var, variance, 1e-5, 1e-3), "Got %v expected %v", variance, expected_var)
	assert.True(t, torch.AllClose(expected_mean, mean, 1e-5, 1e-3), "Got %v expected %v", mean, expected_mean)
	// >>> torch.var_mean(x, 1, True, False)
	// (tensor([1.6667, 1.6667, 1.6667]), tensor([1.5000, 5.5000, 9.5000]))
	expected_var = torch.NewTensor([]float32{1.6667, 1.6667, 1.6667})
	expected_mean = torch.NewTensor([]float32{1.5000, 5.5000, 9.5000})
	variance, mean = x.VarMeanByDim(1, true, false)
	assert.True(t, torch.AllClose(expected_var, variance, 1e-5, 1e-3), "Got %v expected %v", variance, expected_var)
	assert.True(t, torch.AllClose(expected_mean, mean, 1e-5, 1e-3), "Got %v expected %v", mean, expected_mean)
	// >>> torch.var_mean(x, 1, True, True)
	// (tensor([[1.6667],
	//         [1.6667],
	//         [1.6667]]), tensor([[1.5000],
	//         [5.5000],
	//         [9.5000]]))
	expected_var = torch.NewTensor([][]float32{{1.6667}, {1.6667}, {1.6667}})
	expected_mean = torch.NewTensor([][]float32{{1.5000}, {5.5000}, {9.5000}})
	variance, mean = x.VarMeanByDim(1, true, true)
	assert.True(t, torch.AllClose(expected_var, variance, 1e-5, 1e-3), "Got %v expected %v", variance, expected_var)
	assert.True(t, torch.AllClose(expected_mean, mean, 1e-5, 1e-3), "Got %v expected %v", mean, expected_mean)
}

// TODO: count_nonzero

// ---------------------------------------------------------------------------
// MARK: Comparison Ops
// ---------------------------------------------------------------------------

func TestAllClose(t *testing.T) {
	a := assert.New(t)
	x := torch.NewTensor([]float32{8.31, 6.55, 1.39})
	y := torch.NewTensor([]float32{2.38, 3.12, 5.23})
	expected := torch.NewTensor([]float32{8.31 * 2.38, 6.55 * 3.12, 1.39 * 5.23})
	a.True(x.Mul(y).AllClose(expected, 1e-5, 1e-8))
}

func TestIsClose(t *testing.T) {
	x := torch.NewTensor([]float32{8.31, 6.55, 1.39})
	y := torch.NewTensor([]float32{2.38, 3.12, 5.23})
	expected := torch.NewTensor([]float32{8.31 * 2.38, 6.55 * 3.12, 1.39 * 5.23})
	assert.True(t, x.Mul(y).IsClose(expected, 1e-5, 1e-8).All().Item().(bool))
}

// TODO: Argsort

// >>> torch.eq(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
// tensor([[ True, False],
//         [False, True]])
func TestEq(t *testing.T) {
	a := torch.NewTensor([][]int16{{1, 2}, {3, 4}})
	b := torch.NewTensor([][]int16{{1, 3}, {2, 4}})
	outputs := a.Eq(b)
	expected := torch.NewTensor([][]bool{{true, false}, {false, true}})
	assert.True(t, torch.Equal(expected, outputs), "Got %v expected %v", outputs, expected)
}

func TestEqual(t *testing.T) {
	a := torch.NewTensor([]int64{1, 2})
	b := torch.NewTensor([]int64{1, 2})
	c := torch.NewTensor([]int64{3, 4})
	assert.True(t, a.Equal(b))
	assert.False(t, b.Equal(c))
}

func TestGreaterEqual(t *testing.T) {
	a := torch.NewTensor([]int32{3, 2, 1, -2})
	b := torch.NewTensor([]int32{3, 3, 0,  3})
	expected := torch.NewTensor([]bool{true, false, true, false})
	assert.True(t, a.GreaterEqual(b).Equal(expected))
}

func TestGreater(t *testing.T) {
	a := torch.NewTensor([]int32{3, 2, 1, -2})
	b := torch.NewTensor([]int32{3, 3, 0,  3})
	expected := torch.NewTensor([]bool{false, false, true, false})
	assert.True(t, a.Greater(b).Equal(expected))
}

func TestLessEqual(t *testing.T) {
	a := torch.NewTensor([]int32{3, 2, 1, -2})
	b := torch.NewTensor([]int32{3, 3, 0,  3})
	expected := torch.NewTensor([]bool{true, true, false, true})
	assert.True(t, a.LessEqual(b).Equal(expected))
}

func TestTensorLess(t *testing.T) {
	a := torch.NewTensor([]int32{3, 2, 1, -2})
	b := torch.NewTensor([]int32{3, 3, 0,  3})
	expected := torch.NewTensor([]bool{false, true, false, true})
	assert.True(t, a.Less(b).Equal(expected))
}

func TestTensorMaximum(t *testing.T) {
	a := torch.NewTensor([]int32{3, 2, 1, -2})
	b := torch.NewTensor([]int32{3, 3, 0,  3})
	e := torch.NewTensor([]int32{3, 3, 1,  3})
	assert.True(t, a.Maximum(b).Equal(e))
}

func TestTensorMinimum(t *testing.T) {
	a := torch.NewTensor([]int32{3, 2, 1, -2})
	b := torch.NewTensor([]int32{3, 3, 0,  3})
	e := torch.NewTensor([]int32{3, 2, 0, -2})
	assert.True(t, a.Minimum(b).Equal(e))
}

func TestTensorNotEqual(t *testing.T) {
	a := torch.NewTensor([]int32{3, 2, 1, -2})
	b := torch.NewTensor([]int32{3, 3, 0,  3})
	expected := torch.NewTensor([]bool{false, true, true, true})
	assert.True(t, a.NotEqual(b).Equal(expected))
}

// >>> torch.isin(torch.tensor([[1, 2], [3, 4]]), torch.tensor([2, 3]))
// tensor([[False,  True],
//         [ True, False]])
func TestTensorIsIn(t *testing.T) {
	a := torch.NewTensor([][]int32{{1, 2}, {3, 4}})
	b := torch.NewTensor([]int32{2, 3})
	expected := torch.NewTensor([][]bool{{false, true}, {true, false}})
	assert.True(t, a.IsIn(b).Equal(expected))
}

func TestTensorIsFinite(t *testing.T) {
	assert.True(t, torch.NewTensor([]float32{1}).IsFinite().All().Item().(bool))
	assert.False(t, torch.NewTensor([]float64{math.NaN()}).IsFinite().All().Item().(bool))
}

func TestTensorIsInf(t *testing.T) {
	assert.False(t, torch.NewTensor([]float32{1}).IsInf().Any().Item().(bool))
	assert.True(t, torch.NewTensor([]float64{math.Inf(0)}).IsInf().Any().Item().(bool))
}

func TestTensorIsPosInf(t *testing.T) {
	assert.False(t, torch.NewTensor([]float32{1}).IsPosInf().Any().Item().(bool))
	assert.True(t, torch.NewTensor([]float64{math.Inf(1)}).IsPosInf().Any().Item().(bool))
}

func TestTensorIsNegInf(t *testing.T) {
	assert.False(t, torch.NewTensor([]float32{1}).IsNegInf().Any().Item().(bool))
	assert.True(t, torch.NewTensor([]float64{math.Inf(-1)}).IsNegInf().Any().Item().(bool))
}

func TestTensorIsNaN(t *testing.T) {
	assert.False(t, torch.NewTensor([]float32{1}).IsNaN().Any().Item().(bool))
	assert.True(t, torch.NewTensor([]float64{math.NaN()}).IsNaN().Any().Item().(bool))
}

func TestTensorIsReal(t *testing.T) {
	assert.True(t, torch.NewTensor([]float32{1}).IsReal().All().Item().(bool))
	assert.False(t, torch.NewTensor([]complex64{complex(1, 0.4)}).IsReal().All().Item().(bool))
}

// TODO: KthValue

// >>> torch.topk(torch.tensor([[-0.5, -1.], [1., 0.5]]), 1, 1, True, True)
// torch.return_types.topk(
// values=tensor([[-0.5000],
//         [ 1.0000]]),
// indices=tensor([[0],
//         [0]]))
func TestTopK(t *testing.T) {
	tensor := torch.NewTensor([][]float64{{-0.5, -1}, {1, 0.5}})
	topk := tensor.TopK(1, 1, true, true)
	values := torch.NewTensor([][]float64{{-0.5}, {1.0}})
	assert.True(t, torch.Equal(values, topk.Values), "Got %v expected %v", topk.Values, values)
	indices := torch.NewTensor([][]int64{{0}, {0}})
	assert.True(t, torch.Equal(indices, topk.Indices), "Got %v expected %v", topk.Indices, indices)
}

func TestSort(t *testing.T) {
	tensor := torch.NewTensor([][]float32{{1.0, 2.1, 0.5}, {5, 1, 2}})

	expected := torch.NewTensor([][]float32{{0.5, 1.0, 2.1}, {1, 2, 5}})
	assert.True(t, torch.Equal(tensor.Sort(1, false).Values, expected))

	expected = torch.NewTensor([][]float32{{2.1, 1.0, 0.5}, {5, 2, 1}})
	assert.True(t, torch.Equal(tensor.Sort(1, true).Values, expected))

	expected = torch.NewTensor([][]float32{{1.0, 1.0, 0.5}, {5, 2.1, 2}})
	assert.True(t, torch.Equal(tensor.Sort(0, false).Values, expected))

	expected = torch.NewTensor([][]float32{{5, 2.1, 2}, {1.0, 1.0, 0.5}})
	assert.True(t, torch.Equal(tensor.Sort(0, true).Values, expected))
}

// ---------------------------------------------------------------------------
// MARK: Other Operations
// ---------------------------------------------------------------------------

// TODO: atleast_1d
// TODO: atleast_2d
// TODO: atleast_3d
// TODO: bincount
// TODO: block_diag
// TODO: broadcast_tensors
// TODO: broadcast_to
// TODO: broadcast_shapes
// TODO: bucketize
// TODO: cartesian_prod
// TODO: cdist
// TODO: clone
// TODO: combinations
// TODO: corrcoef
// TODO: cov
// TODO: cross
// TODO: cummax
// TODO: cummin
// TODO: cumprod
// TODO: cumsum
// TODO: diag
// TODO: diag_embed
// TODO: diagflat
// TODO: diagonal
// TODO: diff
// TODO: einsum
// TODO: flatten
// TODO: flip
// TODO: fliplr
// TODO: flipud
// TODO: kron
// TODO: rot90
// TODO: gcd
// TODO: histc
// TODO: histogram
// TODO: histogramdd
// TODO: meshgrid
// TODO: lcm
// TODO: logcumsumexp
// TODO: ravel
// TODO: renorm
// TODO: repeat_interleave
// TODO: roll
// TODO: searchsorted
// TODO: tensordot
// TODO: trace
// TODO: tril
// TODO: tril_indices
// TODO: triu
// TODO: triu_indices
// TODO: unflatten
// TODO: vander
// TODO: view_as_real
// TODO: view_as_complex
// TODO: resolve_conj
// TODO: resolve_neg

// ---------------------------------------------------------------------------
// MARK: BLAS and LAPACK Operations
// ---------------------------------------------------------------------------

// TODO: addbmm
// TODO: addmm
// TODO: addmv
// TODO: addr
// TODO: baddbmm
// TODO: bmm
// TODO: chain_matmul
// TODO: cholesky
// TODO: cholesky_inverse
// TODO: cholesky_solve
// TODO: dot
// TODO: geqrf
// TODO: ger
// TODO: inner
// TODO: inverse
// TODO: det
// TODO: logdet
// TODO: slogdet
// TODO: lu
// TODO: lu_solve
// TODO: lu_unpack
// TODO: matmul
// TODO: matrix_power
// TODO: matrix_exp

// >>> a = torch.tensor([[-0.5, -1.], [1., 0.5]])
// >>> b = torch.tensor([[0.25, -0.5], [0.1, 2.]])
// >>> a.mm(b)
// tensor([[-0.2250, -1.7500],
//         [ 0.3000,  0.5000]])
func TestMM(t *testing.T) {
	a := torch.NewTensor([][]float32{{-0.5, -1}, {1, 0.5}})
	b := torch.NewTensor([][]float32{{0.25, -0.5}, {0.1, 2}})
	expected := torch.NewTensor([][]float32{{-0.2250, -1.7500}, {0.3000, 0.5000}})
	assert.True(t, torch.Equal(expected, a.MM(b)))
}

// TODO: mv
// TODO: orgqr
// TODO: ormqr
// TODO: outer
// TODO: pinverse
// TODO: qr
// TODO: svd
// TODO: svd_lowrank
// TODO: pca_lowrank
// TODO: symeig
// TODO: lobpcg
// TODO: trapz
// TODO: trapezoid
// TODO: cumulative_trapezoid
// TODO: triangular_solve
// TODO: vdot
