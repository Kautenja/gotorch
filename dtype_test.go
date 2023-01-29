// test cases for types.go
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
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/Kautenja/gotorch"
    "reflect"
)

func TestGetDtypeOfKind(t *testing.T) {
    assert.Equal(t, torch.Bool,          torch.GetDtypeOfKind(reflect.Bool))
    assert.Equal(t, torch.Byte,          torch.GetDtypeOfKind(reflect.Uint8))
    assert.Equal(t, torch.Char,          torch.GetDtypeOfKind(reflect.Int8))
    assert.Equal(t, torch.Short,         torch.GetDtypeOfKind(reflect.Int16))
    assert.Equal(t, torch.Int,           torch.GetDtypeOfKind(reflect.Int32))
    assert.Equal(t, torch.Long,          torch.GetDtypeOfKind(reflect.Int64))
    assert.Equal(t, torch.Half,          torch.GetDtypeOfKind(reflect.Uint16))
    assert.Equal(t, torch.Float,         torch.GetDtypeOfKind(reflect.Float32))
    assert.Equal(t, torch.Double,        torch.GetDtypeOfKind(reflect.Float64))
    assert.Equal(t, torch.ComplexFloat,  torch.GetDtypeOfKind(reflect.Complex64))
    assert.Equal(t, torch.ComplexDouble, torch.GetDtypeOfKind(reflect.Complex128))
    // Types that are NOT supported in the libtorch back-end.
    assert.Equal(t, torch.Invalid, torch.GetDtypeOfKind(reflect.Array))
    assert.Equal(t, torch.Invalid, torch.GetDtypeOfKind(reflect.Chan))
    assert.Equal(t, torch.Invalid, torch.GetDtypeOfKind(reflect.Func))
    assert.Equal(t, torch.Invalid, torch.GetDtypeOfKind(reflect.Interface))
    assert.Equal(t, torch.Invalid, torch.GetDtypeOfKind(reflect.Invalid))
    assert.Equal(t, torch.Invalid, torch.GetDtypeOfKind(reflect.Map))
    assert.Equal(t, torch.Invalid, torch.GetDtypeOfKind(reflect.String))
    assert.Equal(t, torch.Invalid, torch.GetDtypeOfKind(reflect.Struct))
    assert.Equal(t, torch.Invalid, torch.GetDtypeOfKind(reflect.Slice))
}

func TestDtypeNumBytes(t *testing.T) {
    assert.Equal(t, int64(1), torch.Bool.NumBytes())
    assert.Equal(t, int64(1), torch.Byte.NumBytes())
    assert.Equal(t, int64(1), torch.Char.NumBytes())
    assert.Equal(t, int64(2), torch.Short.NumBytes())
    assert.Equal(t, int64(4), torch.Int.NumBytes())
    assert.Equal(t, int64(8), torch.Long.NumBytes())
    assert.Equal(t, int64(2), torch.Half.NumBytes())
    assert.Equal(t, int64(4), torch.Float.NumBytes())
    assert.Equal(t, int64(8), torch.Double.NumBytes())
    assert.Equal(t, int64(2), torch.ComplexHalf.NumBytes())
    assert.Equal(t, int64(4), torch.ComplexFloat.NumBytes())
    assert.Equal(t, int64(8), torch.ComplexDouble.NumBytes())
    assert.Equal(t, int64(1), torch.QInt8.NumBytes())
    assert.Equal(t, int64(1), torch.QUInt8.NumBytes())
    assert.Equal(t, int64(4), torch.QInt32.NumBytes())
    assert.Equal(t, int64(2), torch.BFloat16.NumBytes())
    assert.PanicsWithValue(t, "Received invalid dtype -1", func() {
        torch.Invalid.NumBytes()
    })
}
