// Go bindings for torch::dtype.
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

package torch

import (
	"fmt"
	"reflect"
)

// An enumeration of the data-types available in torch. Typically it would be
// standard for "Invalid" to correspond to value 0; however, these codes
// are one-to-one with C++ codes in torch.
type Dtype int8
const (
	Byte Dtype = iota
	Char
	Short
	Int
	Long
	Half
	Float
	Double
	ComplexHalf
	ComplexFloat
	ComplexDouble
	Bool
	QInt8
	QUInt8
	QInt32
	BFloat16
	Invalid Dtype = -1
)

var (
	// A mapping of Golang element types to their associated Torch data-type.
	// https://pytorch.org/docs/stable/tensors.html#torch-tensor
	reflectKindToDtype = map[reflect.Kind]Dtype{
		reflect.Bool:       Bool,
		reflect.Uint8:      Byte,
		reflect.Int8:       Char,
		reflect.Int16:      Short,
		reflect.Int32:      Int,
		reflect.Int64:      Long,
		reflect.Uint16:     Half, // TODO: add Bfloat16.
		reflect.Float32:    Float,
		reflect.Float64:    Double,
		// reflect.Uint32:     ComplexHalf,
		reflect.Complex64:  ComplexFloat,
		reflect.Complex128: ComplexDouble,
	}

	// A mapping of Torch data-type to the number of bytes they require.
	dtypeToNumBytes = map[Dtype]int64{
		Byte:          1,
		Char:          1,
		Short:         2,
		Int:           4,
		Long:          8,
		Half:          2,
		Float:         4,
		Double:        8,
		ComplexHalf:   2,
		ComplexFloat:  4,
		ComplexDouble: 8,
		Bool:          1,
		QInt8:         1,
		QUInt8:        1,
		QInt32:        4,
		BFloat16:      2,
	}
)

// Map an element type kind to its associated Dtype.
func GetDtypeOfKind(kind reflect.Kind) Dtype {
	if dtype, ok := reflectKindToDtype[kind]; ok {
		return dtype
	}
	return Invalid
}

// Return the number of bytes consumed by each element of the given data-type.
func (dtype Dtype) NumBytes() int64 {
	if numBytes, ok := dtypeToNumBytes[dtype]; ok {
		return numBytes
	}
	panic(fmt.Sprintf("Received invalid dtype %v", dtype))
}
