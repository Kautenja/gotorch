// test cases for types.go
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

package internal_test

import (
	"testing"
	"github.com/stretchr/testify/assert"
	"reflect"
	"unsafe"
	"github.com/Kautenja/gotorch/internal"
)

// MARK: GetSizesAndKindOfSlice

func TestGetSizesAndKindOfSlice_Scalar(t *testing.T) {
	data := float32(0)
	shape, _ := internal.GetSizesAndKindOfSlice(data)
	assert.Equal(t, 0, len(shape))
}

func TestGetSizesAndKindOfSlice_EmptySlice(t *testing.T) {
	data := []float32{}
	shape, _ := internal.GetSizesAndKindOfSlice(data)
	assert.Equal(t, []int64{0}, shape)
}

func TestGetSizesAndKindOfSlice_1DSlice(t *testing.T) {
	data := []float32{1, 2}
	shape, _ := internal.GetSizesAndKindOfSlice(data)
	assert.Equal(t, []int64{2}, shape)
}

func TestGetSizesAndKindOfSlice_2DSlice(t *testing.T) {
	data := [][]float32{{1, 2}, {3, 4}, {5, 6}}
	shape, _ := internal.GetSizesAndKindOfSlice(data)
	assert.Equal(t, []int64{3, 2}, shape)
}

func TestGetSizesAndKindOfSlice_String(t *testing.T) {
	data := "foo"
	_, kind := internal.GetSizesAndKindOfSlice(data)
	assert.Equal(t, kind, reflect.String)
}

func TestGetSizesAndKindOfSlice_Bool(t *testing.T) {
	data := true
	_, kind := internal.GetSizesAndKindOfSlice(data)
	assert.Equal(t, reflect.Bool, kind)
}

func TestGetSizesAndKindOfSlice_Uint8(t *testing.T) {
	data := uint8(1)
	_, kind := internal.GetSizesAndKindOfSlice(data)
	assert.Equal(t, reflect.Uint8, kind)
}

func TestGetSizesAndKindOfSlice_Int8(t *testing.T) {
	data := int8(1)
	_, kind := internal.GetSizesAndKindOfSlice(data)
	assert.Equal(t, reflect.Int8, kind)
}

func TestGetSizesAndKindOfSlice_Int16(t *testing.T) {
	data := int16(1)
	_, kind := internal.GetSizesAndKindOfSlice(data)
	assert.Equal(t, reflect.Int16, kind)
}

func TestGetSizesAndKindOfSlice_Uint16(t *testing.T) {
	data := uint16(1)
	_, kind := internal.GetSizesAndKindOfSlice(data)
	assert.Equal(t, reflect.Uint16, kind)
}

func TestGetSizesAndKindOfSlice_Int32(t *testing.T) {
	data := int32(1)
	_, kind := internal.GetSizesAndKindOfSlice(data)
	assert.Equal(t, reflect.Int32, kind)
}

func TestGetSizesAndKindOfSlice_Int64(t *testing.T) {
	data := int64(1)
	_, kind := internal.GetSizesAndKindOfSlice(data)
	assert.Equal(t, reflect.Int64, kind)
}

func TestGetSizesAndKindOfSlice_Float32(t *testing.T) {
	data := float32(1)
	_, kind := internal.GetSizesAndKindOfSlice(data)
	assert.Equal(t, reflect.Float32, kind)
}

func TestGetSizesAndKindOfSlice_Float64(t *testing.T) {
	data := float64(1)
	_, kind := internal.GetSizesAndKindOfSlice(data)
	assert.Equal(t, reflect.Float64, kind)
}

func TestGetSizesAndKindOfSlice_Complex64(t *testing.T) {
	data := complex64(complex(1, 0.5))
	_, kind := internal.GetSizesAndKindOfSlice(data)
	assert.Equal(t, reflect.Complex64, kind)
}

func TestGetSizesAndKindOfSlice_Complex128(t *testing.T) {
	data := complex128(complex(1, 0.5))
	_, kind := internal.GetSizesAndKindOfSlice(data)
	assert.Equal(t, reflect.Complex128, kind)
}

// MARK: FlattenSlice

func TestFlattenSlice_PanicsOnInvalidKindInput(t *testing.T) {
	assert.PanicsWithValue(t, "FlattenSlice not implemented for kind array", func() {
		internal.FlattenSlice(nil, reflect.Array)
	})
	assert.PanicsWithValue(t, "FlattenSlice not implemented for kind chan", func() {
		internal.FlattenSlice(nil, reflect.Chan)
	})
	assert.PanicsWithValue(t, "FlattenSlice not implemented for kind func", func() {
		internal.FlattenSlice(nil, reflect.Func)
	})
	assert.PanicsWithValue(t, "FlattenSlice not implemented for kind interface", func() {
		internal.FlattenSlice(nil, reflect.Interface)
	})
	assert.PanicsWithValue(t, "FlattenSlice not implemented for kind invalid", func() {
		internal.FlattenSlice(nil, reflect.Invalid)
	})
	assert.PanicsWithValue(t, "FlattenSlice not implemented for kind map", func() {
		internal.FlattenSlice(nil, reflect.Map)
	})
	assert.PanicsWithValue(t, "FlattenSlice not implemented for kind string", func() {
		internal.FlattenSlice(nil, reflect.String)
	})
	assert.PanicsWithValue(t, "FlattenSlice not implemented for kind struct", func() {
		internal.FlattenSlice(nil, reflect.Struct)
	})
	assert.PanicsWithValue(t, "FlattenSlice not implemented for kind slice", func() {
		internal.FlattenSlice(nil, reflect.Slice)
	})
}


func TestFlattenSlice_Bool(t *testing.T) {
	slice := [][]bool{{true, true}, {false, true}, {false, false}}
	// FlattenSlice returns raw data, we need to mock a slice with the buffer.
	var data []bool
	header := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	header.Data = uintptr(internal.FlattenSlice(slice, reflect.Bool))
	header.Len = 6
	header.Cap = 6
	// Check for equality based on expected flat view.
	assert.Equal(t, true,  data[0])
	assert.Equal(t, true,  data[1])
	assert.Equal(t, false, data[2])
	assert.Equal(t, true,  data[3])
	assert.Equal(t, false, data[4])
	assert.Equal(t, false, data[5])
}

func TestFlattenSlice_Uint8(t *testing.T) {
	slice := [][]uint8{{1, 1}, {0, 1}, {0, 0}}
	// FlattenSlice returns raw data, we need to mock a slice with the buffer.
	var data []uint8
	header := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	header.Data = uintptr(internal.FlattenSlice(slice, reflect.Uint8))
	header.Len = 6
	header.Cap = 6
	// Check for equality based on expected flat view.
	assert.Equal(t, uint8(1), data[0])
	assert.Equal(t, uint8(1), data[1])
	assert.Equal(t, uint8(0), data[2])
	assert.Equal(t, uint8(1), data[3])
	assert.Equal(t, uint8(0), data[4])
	assert.Equal(t, uint8(0), data[5])
}

func TestFlattenSlice_Int8(t *testing.T) {
	slice := [][]int8{{1, 1}, {0, 1}, {0, 0}}
	// FlattenSlice returns raw data, we need to mock a slice with the buffer.
	var data []int8
	header := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	header.Data = uintptr(internal.FlattenSlice(slice, reflect.Int8))
	header.Len = 6
	header.Cap = 6
	// Check for equality based on expected flat view.
	assert.Equal(t, int8(1), data[0])
	assert.Equal(t, int8(1), data[1])
	assert.Equal(t, int8(0), data[2])
	assert.Equal(t, int8(1), data[3])
	assert.Equal(t, int8(0), data[4])
	assert.Equal(t, int8(0), data[5])
}

func TestFlattenSlice_Int16(t *testing.T) {
	slice := [][]int16{{1, 1}, {0, 1}, {0, 0}}
	// FlattenSlice returns raw data, we need to mock a slice with the buffer.
	var data []int16
	header := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	header.Data = uintptr(internal.FlattenSlice(slice, reflect.Int16))
	header.Len = 6
	header.Cap = 6
	// Check for equality based on expected flat view.
	assert.Equal(t, int16(1), data[0])
	assert.Equal(t, int16(1), data[1])
	assert.Equal(t, int16(0), data[2])
	assert.Equal(t, int16(1), data[3])
	assert.Equal(t, int16(0), data[4])
	assert.Equal(t, int16(0), data[5])
}

func TestFlattenSlice_Int32(t *testing.T) {
	slice := [][]int32{{1, 1}, {0, 1}, {0, 0}}
	// FlattenSlice returns raw data, we need to mock a slice with the buffer.
	var data []int32
	header := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	header.Data = uintptr(internal.FlattenSlice(slice, reflect.Int32))
	header.Len = 6
	header.Cap = 6
	// Check for equality based on expected flat view.
	assert.Equal(t, int32(1), data[0])
	assert.Equal(t, int32(1), data[1])
	assert.Equal(t, int32(0), data[2])
	assert.Equal(t, int32(1), data[3])
	assert.Equal(t, int32(0), data[4])
	assert.Equal(t, int32(0), data[5])
}

func TestFlattenSlice_Int64(t *testing.T) {
	slice := [][]int64{{1, 1}, {0, 1}, {0, 0}}
	// FlattenSlice returns raw data, we need to mock a slice with the buffer.
	var data []int64
	header := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	header.Data = uintptr(internal.FlattenSlice(slice, reflect.Int64))
	header.Len = 6
	header.Cap = 6
	// Check for equality based on expected flat view.
	assert.Equal(t, int64(1), data[0])
	assert.Equal(t, int64(1), data[1])
	assert.Equal(t, int64(0), data[2])
	assert.Equal(t, int64(1), data[3])
	assert.Equal(t, int64(0), data[4])
	assert.Equal(t, int64(0), data[5])
}

func TestFlattenSlice_Uint16(t *testing.T) {
	slice := [][]uint16{{1, 1}, {0, 1}, {0, 0}}
	// FlattenSlice returns raw data, we need to mock a slice with the buffer.
	var data []uint16
	header := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	header.Data = uintptr(internal.FlattenSlice(slice, reflect.Uint16))
	header.Len = 6
	header.Cap = 6
	// Check for equality based on expected flat view.
	assert.Equal(t, uint16(1), data[0])
	assert.Equal(t, uint16(1), data[1])
	assert.Equal(t, uint16(0), data[2])
	assert.Equal(t, uint16(1), data[3])
	assert.Equal(t, uint16(0), data[4])
	assert.Equal(t, uint16(0), data[5])
}

func TestFlattenSlice_Float32(t *testing.T) {
	slice := [][]float32{{1, 1}, {0, 1}, {0, 0}}
	// FlattenSlice returns raw data, we need to mock a slice with the buffer.
	var data []float32
	header := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	header.Data = uintptr(internal.FlattenSlice(slice, reflect.Float32))
	header.Len = 6
	header.Cap = 6
	// Check for equality based on expected flat view.
	assert.Equal(t, float32(1), data[0])
	assert.Equal(t, float32(1), data[1])
	assert.Equal(t, float32(0), data[2])
	assert.Equal(t, float32(1), data[3])
	assert.Equal(t, float32(0), data[4])
	assert.Equal(t, float32(0), data[5])
}

func TestFlattenSlice_Float64(t *testing.T) {
	slice := [][]float64{{1, 1}, {0, 1}, {0, 0}}
	// FlattenSlice returns raw data, we need to mock a slice with the buffer.
	var data []float64
	header := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	header.Data = uintptr(internal.FlattenSlice(slice, reflect.Float64))
	header.Len = 6
	header.Cap = 6
	// Check for equality based on expected flat view.
	assert.Equal(t, float64(1), data[0])
	assert.Equal(t, float64(1), data[1])
	assert.Equal(t, float64(0), data[2])
	assert.Equal(t, float64(1), data[3])
	assert.Equal(t, float64(0), data[4])
	assert.Equal(t, float64(0), data[5])
}

func TestFlattenSlice_Complex64(t *testing.T) {
	slice := [][]complex64{
		{complex(1.0, 0.5), complex(2.0, 0.0)},
		{complex(0.5, 1.0), complex(1.2, 4.5)},
		{complex(1.3, 0.4), complex(1.3, 0.0)},
	}
	// FlattenSlice returns raw data, we need to mock a slice with the buffer.
	var data []complex64
	header := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	header.Data = uintptr(internal.FlattenSlice(slice, reflect.Complex64))
	header.Len = 6
	header.Cap = 6
	// Check for equality based on expected flat view.
	assert.Equal(t, complex64(complex(1.0, 0.5)), data[0])
	assert.Equal(t, complex64(complex(2.0, 0.0)), data[1])
	assert.Equal(t, complex64(complex(0.5, 1.0)), data[2])
	assert.Equal(t, complex64(complex(1.2, 4.5)), data[3])
	assert.Equal(t, complex64(complex(1.3, 0.4)), data[4])
	assert.Equal(t, complex64(complex(1.3, 0.0)), data[5])
}

func TestFlattenSlice_Complex128(t *testing.T) {
	slice := [][]complex128{
		{complex(1.0, 0.5), complex(2.0, 0.0)},
		{complex(0.5, 1.0), complex(1.2, 4.5)},
		{complex(1.3, 0.4), complex(1.3, 0.0)},
	}
	// FlattenSlice returns raw data, we need to mock a slice with the buffer.
	var data []complex128
	header := (*reflect.SliceHeader)(unsafe.Pointer(&data))
	header.Data = uintptr(internal.FlattenSlice(slice, reflect.Complex128))
	header.Len = 6
	header.Cap = 6
	// Check for equality based on expected flat view.
	assert.Equal(t, complex128(complex(1.0, 0.5)), data[0])
	assert.Equal(t, complex128(complex(2.0, 0.0)), data[1])
	assert.Equal(t, complex128(complex(0.5, 1.0)), data[2])
	assert.Equal(t, complex128(complex(1.2, 4.5)), data[3])
	assert.Equal(t, complex128(complex(1.3, 0.4)), data[4])
	assert.Equal(t, complex128(complex(1.3, 0.0)), data[5])
}
