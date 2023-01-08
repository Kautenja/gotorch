// Go bindings for torch::dtype.
//
// Copyright (c) 2022 Christian Kauten
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

import "reflect"

// An enumeration of the data-types in libtorch.
type Dtype int8
const (
    Byte Dtype = iota  // Byte Dtype 0
    Char               // Char Dtype 1
    Short              // Short Dtype 2
    Int                // Int Dtype 3
    Long               // Long Dtype 4
    Half               // Half Dtype 5
    Float              // Float Dtype 6
    Double             // Double Dtype 7
    ComplexHalf        // ComplexHalf Dtype 8
    ComplexFloat       // ComplexFloat Dtype 9
    ComplexDouble      // ComplexDouble Dtype 10
    Bool               // Bool Dtype 11
    QInt8              // QInt8 Dtype 12
    QUInt8             // QUInt8 Dtype 13
    QInt32             // QInt32 Dtype 14
    BFloat16           // BFloat16 Dtype 15
    Invalid Dtype = -1 // Invalid Dtype
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
)

// Map an element type kind to its associated Dtype.
func GetDtypeOfKind(kind reflect.Kind) Dtype {
    if dtype, ok := reflectKindToDtype[kind]; ok {
        return dtype
    }
    return Invalid
}
