// test cases for ivalue.go
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

package torch_test

import (
    "runtime"
    "testing"
    "github.com/stretchr/testify/assert"
    "github.com/Kautenja/gotorch"
)

// This test checks that the finalizer runs when the jit module is de-allocated.
func TestIValueGarbageCollection(t *testing.T) {
    tensor := torch.NewTensor([]float32{1})
    _ = torch.NewIValue(tensor)
    runtime.GC()
}

func TestIValuePanicsOnUnknownType(t *testing.T) {
    type MockedThing struct { }
    expected := "IValue not supported for data of type torch_test.MockedThing"
    assert.PanicsWithValue(t, expected, func() { _ = torch.NewIValue(MockedThing{}) })
}

func TestIValueTensor(t *testing.T) {
    tensor := torch.NewTensor([]float32{1})
    data := torch.NewIValue(tensor)
    assert.NotNil(t, data.T)
    assert.False(t, data.IsNil())
    assert.True(t, data.IsTensor())
    value := data.ToTensor()
    assert.True(t, torch.Equal(tensor, value))
    assert.NotEqual(t, tensor.T, value.T)
}

func TestIValueFloat32(t *testing.T) {
    data := torch.NewIValue(float32(7))
    assert.NotNil(t, data.T)
    assert.False(t, data.IsNil())
    assert.True(t, data.IsScalar())
    assert.True(t, data.IsDouble())
    assert.Equal(t, float64(7), data.ToDouble())
}

func TestIValueFloat64(t *testing.T) {
    data := torch.NewIValue(float64(7))
    assert.NotNil(t, data.T)
    assert.False(t, data.IsNil())
    assert.True(t, data.IsScalar())
    assert.True(t, data.IsDouble())
    assert.Equal(t, float64(7), data.ToDouble())
}

func TestIValueInt(t *testing.T) {
    data := torch.NewIValue(7)
    assert.NotNil(t, data.T)
    assert.False(t, data.IsNil())
    assert.True(t, data.IsScalar())
    assert.True(t, data.IsInt())
    assert.Equal(t, 7, data.ToInt())
}

func TestIValueInt32(t *testing.T) {
    data := torch.NewIValue(int32(7))
    assert.NotNil(t, data.T)
    assert.False(t, data.IsNil())
    assert.True(t, data.IsScalar())
    assert.True(t, data.IsInt())
    assert.Equal(t, 7, data.ToInt())
}

// func TestIValueInt64(t *testing.T) {
//     data := torch.NewIValue(int64(7))
//     assert.NotNil(t, data.T)
//     assert.False(t, data.IsNil())
//     assert.True(t, data.IsScalar())
//     assert.True(t, data.IsInt())
//     assert.Equal(t, 7, data.ToInt())
// }

func TestIValueBool(t *testing.T) {
    data := torch.NewIValue(true)
    assert.NotNil(t, data.T)
    assert.False(t, data.IsNil())
    assert.True(t, data.IsScalar())
    assert.True(t, data.IsBool())
    assert.True(t, data.ToBool())
}

func TestIValueString(t *testing.T) {
    data := torch.NewIValue("foo")
    assert.NotNil(t, data.T)
    assert.False(t, data.IsNil())
    assert.True(t, data.IsString())
    assert.Equal(t, "foo", data.ToString())
}

func TestIValueComplex64(t *testing.T) {
    data := torch.NewIValue(complex64(complex(0.8, 0.6)))
    assert.NotNil(t, data.T)
    assert.False(t, data.IsNil())
    assert.True(t, data.IsScalar())
    assert.True(t, data.IsComplexDouble())
    output := data.ToComplexDouble()
    assert.InEpsilon(t, float64(0.8), real(output), 1e-5)
    assert.InEpsilon(t, float64(0.6), imag(output), 1e-5)
}

func TestIValueComplex128(t *testing.T) {
    data := torch.NewIValue(complex128(complex(0.8, 0.6)))
    assert.NotNil(t, data.T)
    assert.False(t, data.IsNil())
    assert.True(t, data.IsScalar())
    assert.True(t, data.IsComplexDouble())
    output := data.ToComplexDouble()
    assert.InEpsilon(t, float64(0.8), real(output), 1e-5)
    assert.InEpsilon(t, float64(0.6), imag(output), 1e-5)
}

// func TestIValueList(t *testing.T) {
//     valueA := torch.NewIValue(7)
//     valueB := torch.NewIValue(2.22)
//     data := torch.NewIValue([]torch.IValue{valueA, valueB})
//     assert.NotNil(t, data.T)
//     assert.False(t, data.IsNil())
//     assert.True(t, data.IsList())
// }

func TestIValueBoolList(t *testing.T) {
    slice := []bool{true, false, false, true, true}
    var data torch.IValue
    assert.NotPanics(t, func() { data = torch.NewIValue(slice) })
    assert.NotNil(t, data.T)
    assert.False(t, data.IsNil())
    assert.Equal(t, int64(5), data.LengthList())
    // ToBoolList API
    assert.True(t, data.IsBoolList())
    // outputs := data.ToBoolList()
    // assert.Equal(t, 5, len(outputs))
    // for idx, value := range slice {
    //     assert.Equal(t, value, outputs[idx])
    // }
    // ToList API
    assert.True(t, data.IsList())
    ivalues := data.ToList()
    assert.Equal(t, 5, len(ivalues))
    for idx, value := range slice {
        assert.True(t, ivalues[idx].IsBool())
        assert.Equal(t, value, ivalues[idx].ToBool())
    }
}

// func TestIValueInt32List(t *testing.T) {
//     slice := []int32{1, 0, 0, 1, 1}
//     var data torch.IValue
//     assert.NotPanics(t, func() { data = torch.NewIValue(slice) })
//     assert.NotNil(t, data.T)
//     assert.False(t, data.IsNil())
//     assert.Equal(t, int64(5), data.LengthList())
//     // ToBoolList API
//     assert.True(t, data.IsIntList())
//     // outputs := data.ToIntList()
//     // assert.Equal(t, 5, len(outputs))
//     // for idx, value := range slice {
//     //     assert.Equal(t, value, outputs[idx])
//     // }
//     // ToList API
//     assert.True(t, data.IsList())
//     ivalues := data.ToList()
//     assert.Equal(t, 5, len(ivalues))
//     for idx, value := range slice {
//         assert.True(t, ivalues[idx].IsInt())
//         assert.Equal(t, value, ivalues[idx].ToInt())
//     }
// }

// func TestIValueIntList(t *testing.T) {
//     slice := []int{1, 0, 0, 1, 1}
//     var data torch.IValue
//     assert.NotPanics(t, func() { data = torch.NewIValue(slice) })
//     assert.NotNil(t, data.T)
//     assert.False(t, data.IsNil())
//     assert.Equal(t, int64(5), data.LengthList())
//     // ToBoolList API
//     assert.True(t, data.IsIntList())
//     // outputs := data.ToIntList()
//     // assert.Equal(t, 5, len(outputs))
//     // for idx, value := range slice {
//     //     assert.Equal(t, value, outputs[idx])
//     // }
//     // ToList API
//     assert.True(t, data.IsList())
//     ivalues := data.ToList()
//     assert.Equal(t, 5, len(ivalues))
//     for idx, value := range slice {
//         assert.True(t, ivalues[idx].IsInt())
//         assert.Equal(t, value, ivalues[idx].ToInt())
//     }
// }

func TestIValueFloat32List(t *testing.T) {
    slice := []float32{1, 0, 0, 1, 1}
    var data torch.IValue
    assert.NotPanics(t, func() { data = torch.NewIValue(slice) })
    assert.NotNil(t, data.T)
    assert.False(t, data.IsNil())
    assert.Equal(t, int64(5), data.LengthList())
    // ToBoolList API
    assert.True(t, data.IsDoubleList())
    // outputs := data.ToFloatList()
    // assert.Equal(t, 5, len(outputs))
    // for idx, value := range slice {
    //     assert.Equal(t, float64(value), outputs[idx])
    // }
    // ToList API
    assert.True(t, data.IsList())
    ivalues := data.ToList()
    assert.Equal(t, 5, len(ivalues))
    for idx, value := range slice {
        assert.True(t, ivalues[idx].IsDouble())
        assert.Equal(t, float64(value),  ivalues[idx].ToDouble())
    }
}

func TestIValueFloat64List(t *testing.T) {
    slice := []float64{1, 0, 0, 1, 1}
    var data torch.IValue
    assert.NotPanics(t, func() { data = torch.NewIValue(slice) })
    assert.NotNil(t, data.T)
    assert.False(t, data.IsNil())
    assert.Equal(t, int64(5), data.LengthList())
    // ToBoolList API
    assert.True(t, data.IsDoubleList())
    // outputs := data.ToDoubleList()
    // assert.Equal(t, 5, len(outputs))
    // for idx, value := range slice {
    //     assert.Equal(t, value, outputs[idx])
    // }
    // ToList API
    assert.True(t, data.IsList())
    ivalues := data.ToList()
    assert.Equal(t, 5, len(ivalues))
    for idx, value := range slice {
        assert.True(t, ivalues[idx].IsDouble())
        assert.Equal(t, value,  ivalues[idx].ToDouble())
    }
}

func TestIValueTensorList(t *testing.T) {
    tensorA := torch.NewTensor([]float32{1})
    tensorB := torch.NewTensor([]float32{2})
    data := torch.NewIValue([]torch.Tensor{tensorA, tensorB})
    assert.NotNil(t, data.T)
    assert.False(t, data.IsNil())
    assert.True(t, data.IsTensorList())
    assert.Equal(t, int64(2), data.LengthList())
    // ToTensorList API
    outputs := data.ToTensorList()
    assert.Equal(t, 2, len(outputs))
    assert.True(t, outputs[0].Equal(tensorA),
        "Got %v, expected %v", outputs[0], tensorA)
    assert.True(t, outputs[1].Equal(tensorB),
        "Got %v, expected %v", outputs[1], tensorB)
    // ToList API
    assert.True(t, data.IsList())
    ivalues := data.ToList()
    assert.Equal(t, 2, len(ivalues))
    assert.True(t, ivalues[0].IsTensor())
    assert.True(t, ivalues[1].IsTensor())
    assert.True(t, ivalues[0].ToTensor().Equal(tensorA),
        "Got %v, expected %v", ivalues[0].ToTensor(), tensorA)
    assert.True(t, ivalues[1].ToTensor().Equal(tensorB),
        "Got %v, expected %v", ivalues[1].ToTensor(), tensorB)
}

func TestIValueNil(t *testing.T) {
    data := torch.NewIValue(nil)
    assert.NotNil(t, data.T)
    assert.False(t, data.IsTensor())
    assert.True(t, data.IsNil())
    assert.Equal(t, "None", data.ToNone())
}

func TestIValueDevice(t *testing.T) {
    device := torch.NewDevice("cpu")
    data := torch.NewIValue(device)
    assert.NotNil(t, data.T)
    assert.False(t, data.IsNil())
    assert.True(t, data.IsDevice())
    assert.NotEqual(t, device, data.ToDevice())
    // TODO: test device name for equality
}
