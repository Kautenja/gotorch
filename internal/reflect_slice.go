// Internal helper functions for interpreting Go slices as tensors.
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

package torch_internal

import (
    "fmt"
    "reflect"
    "unsafe"
)

// Extract the shape and data-type of a slice. Returns a tuple of the shape of
// the tensor-like data structure, and the kind of the elements contained by
// the tensor-like structure.
func GetSizesAndKindOfSlice(data interface{}) ([]int64, reflect.Kind) {
    var size []int64
    current_vector := reflect.ValueOf(data)
    for {  // Recursively unwrap the interface to find the shape and data-type.
        // Check the type of the data, if it isn't a slice then we've reached
        // the bottom of the recursion and can return the shape and data type.
        kind := current_vector.Type().Kind()
        if kind != reflect.Slice {
            return size, kind
        }
        // Otherwise, we're working with a slice, add the length of the slice to
        // the output shape.
        size = append(size, int64(current_vector.Len()))
        // It's possible that we're dealing with an empty slice. This can be the
        // case for inputs like `[]float32{}` expressing an empty vector. In
        // This special case we return the current size (which validly contains)
        // a 0 for the current dimension and introspect the type of the empty
        // slice using `Elem`.
        if current_vector.Len() == 0 {
            return size, current_vector.Type().Elem().Kind()
        }
        // If we reach this point, unwrap the next slice/value and recurse.
        current_vector = current_vector.Index(0)
    }
}

// Flatten a slice of the given kind to a 1-dimensional buffer of contiguous
// data. The input is expected to be an n-dimensional slice of data and the
// output is the `Data` buffer of the flattened representation of the input.
//
// https://medium.com/@the1mills/flattening-arrays-slices-with-golang-c796905debbe
// provides a way to flatten any recursive slices into []interface{}.  However,
// we cannot reuse this solution here, because libtorch wants
// []float32/float64/..., instead of []interface{}.  Without type template as
// that in C++, we have to write the following Go switch over types.
func FlattenSlice(slc interface{}, kind reflect.Kind) unsafe.Pointer {
    switch kind {
    case reflect.Bool:
        f := flattenSliceBool(nil, reflect.ValueOf(slc))
        return unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&f)).Data)
    case reflect.Uint8:
        f := flattenSliceByte(nil, reflect.ValueOf(slc))
        return unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&f)).Data)
    case reflect.Int8:
        f := flattenSliceChar(nil, reflect.ValueOf(slc))
        return unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&f)).Data)
    case reflect.Int16:
        f := flattenSliceShort(nil, reflect.ValueOf(slc))
        return unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&f)).Data)
    case reflect.Int32:
        f := flattenSliceInt(nil, reflect.ValueOf(slc))
        return unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&f)).Data)
    case reflect.Int64:
        f := flattenSliceLong(nil, reflect.ValueOf(slc))
        return unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&f)).Data)
    case reflect.Uint16:
        f := flattenSliceUint16(nil, reflect.ValueOf(slc))
        return unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&f)).Data)
    case reflect.Float32:
        f := flattenSliceFloat32(nil, reflect.ValueOf(slc))
        return unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&f)).Data)
    case reflect.Float64:
        f := flattenSliceFloat64(nil, reflect.ValueOf(slc))
        return unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&f)).Data)
    case reflect.Complex64:
        f := flattenSliceComplex64(nil, reflect.ValueOf(slc))
        return unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&f)).Data)
    case reflect.Complex128:
        f := flattenSliceComplex128(nil, reflect.ValueOf(slc))
        return unsafe.Pointer((*reflect.SliceHeader)(unsafe.Pointer(&f)).Data)
    }
    panic(fmt.Sprintf("FlattenSlice not implemented for kind %v", kind))
}

func flattenSliceBool(args []bool, v reflect.Value) []bool {
    if v.Kind() == reflect.Array || v.Kind() == reflect.Slice {
        for i := 0; i < v.Len(); i++ {
            args = flattenSliceBool(args, v.Index(i))
        }
    } else {
        args = append(args, v.Bool())
    }
    return args
}

func flattenSliceByte(args []uint8, v reflect.Value) []uint8 {
    if v.Kind() == reflect.Array || v.Kind() == reflect.Slice {
        for i := 0; i < v.Len(); i++ {
            args = flattenSliceByte(args, v.Index(i))
        }
    } else {
        args = append(args, uint8(v.Uint()))
    }
    return args
}

func flattenSliceChar(args []int8, v reflect.Value) []int8 {
    if v.Kind() == reflect.Array || v.Kind() == reflect.Slice {
        for i := 0; i < v.Len(); i++ {
            args = flattenSliceChar(args, v.Index(i))
        }
    } else {
        args = append(args, int8(v.Int()))
    }
    return args
}

func flattenSliceShort(args []int16, v reflect.Value) []int16 {
    if v.Kind() == reflect.Array || v.Kind() == reflect.Slice {
        for i := 0; i < v.Len(); i++ {
            args = flattenSliceShort(args, v.Index(i))
        }
    } else {
        args = append(args, int16(v.Int()))
    }
    return args
}

func flattenSliceInt(args []int32, v reflect.Value) []int32 {
    if v.Kind() == reflect.Array || v.Kind() == reflect.Slice {
        for i := 0; i < v.Len(); i++ {
            args = flattenSliceInt(args, v.Index(i))
        }
    } else {
        args = append(args, int32(v.Int()))
    }
    return args
}

func flattenSliceLong(args []int64, v reflect.Value) []int64 {
    if v.Kind() == reflect.Array || v.Kind() == reflect.Slice {
        for i := 0; i < v.Len(); i++ {
            args = flattenSliceLong(args, v.Index(i))
        }
    } else {
        args = append(args, v.Int())
    }
    return args
}

func flattenSliceUint16(args []uint16, v reflect.Value) []uint16 {
    if v.Kind() == reflect.Array || v.Kind() == reflect.Slice {
        for i := 0; i < v.Len(); i++ {
            args = flattenSliceUint16(args, v.Index(i))
        }
    } else {
        args = append(args, uint16(v.Uint()))
    }
    return args
}

func flattenSliceFloat32(args []float32, v reflect.Value) []float32 {
    if v.Kind() == reflect.Array || v.Kind() == reflect.Slice {
        for i := 0; i < v.Len(); i++ {
            args = flattenSliceFloat32(args, v.Index(i))
        }
    } else {
        args = append(args, float32(v.Float()))
    }
    return args
}

func flattenSliceFloat64(args []float64, v reflect.Value) []float64 {
    if v.Kind() == reflect.Array || v.Kind() == reflect.Slice {
        for i := 0; i < v.Len(); i++ {
            args = flattenSliceFloat64(args, v.Index(i))
        }
    } else {
        args = append(args, v.Float())
    }
    return args
}

func flattenSliceComplex64(args []complex64, v reflect.Value) []complex64 {
    if v.Kind() == reflect.Array || v.Kind() == reflect.Slice {
        for i := 0; i < v.Len(); i++ {
            args = flattenSliceComplex64(args, v.Index(i))
        }
    } else {
        args = append(args, complex64(v.Complex()))
    }
    return args
}

func flattenSliceComplex128(args []complex128, v reflect.Value) []complex128 {
    if v.Kind() == reflect.Array || v.Kind() == reflect.Slice {
        for i := 0; i < v.Len(); i++ {
            args = flattenSliceComplex128(args, v.Index(i))
        }
    } else {
        args = append(args, v.Complex())
    }
    return args
}
