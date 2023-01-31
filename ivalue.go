// Go bindings for c10::IValue.
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

package torch

// #cgo CFLAGS: -I ${SRCDIR}
// #cgo LDFLAGS: -L ${SRCDIR}/build -Wl,-rpath ${SRCDIR}/build -lcgotorch
// #include <stdio.h>
// #include <stdlib.h>
// #include <complex.h>
// #include "cgotorch/cgotorch.h"
import "C"
import (
    "unsafe"
    "reflect"
    "fmt"
    "runtime"
    internal "github.com/Kautenja/gotorch/internal"
)

// IValue wraps a pointer to a C.IValue as an unsafe Pointer.
type IValue struct {
    // A pointer to a C.IValue.
    T *unsafe.Pointer
}

// ---------------------------------------------------------------------------
// MARK: Constructors
// ---------------------------------------------------------------------------

// Create a new IValue from arbitrary data.
func NewIValue(data interface{}) IValue {
    var output C.IValue
    switch t := data.(type) {
    case nil:
        internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromNone(&output)))
        break
    case bool:
        internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromBool(&output, C.bool(t))))
        break
    case int:
        internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromInt(&output, C.int(t))))
        break
    case int32:
        internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromInt(&output, C.int(t))))
        break
    // case int64:
    //     internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromInt(&output, C.int(t))))
    //     break
    case float32:
        internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromDouble(&output, C.double(t))))
        break
    case float64:
        internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromDouble(&output, C.double(t))))
        break
    case complex64:
        internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromComplexDouble(&output, C.complexdouble(t))))
        break
    case complex128:
        internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromComplexDouble(&output, C.complexdouble(t))))
        break
    case string:
        stringData := C.CString(t)
        defer C.free(unsafe.Pointer(stringData))
        internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromString(&output, stringData)))
        break
    case Tensor:
        internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromTensor(&output, (C.Tensor)(*t.T))))
        break
    case []bool:
        internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromBoolList(
            &output,
            (*C.bool)(unsafe.Pointer(&t[0])),
            C.int(len(t)),
        )))
        break
    // case []int:
    //     internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromIntList(
    //         &output,
    //         (*C.int)(unsafe.Pointer(&t[0])),
    //         C.int(len(t)),
    //     )))
    //     break
    // case []int32:
    //     internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromIntList(
    //         &output,
    //         (*C.int)(unsafe.Pointer(&t[0])),
    //         C.int(len(t)),
    //     )))
    //     break
    // case []int64:
    //     internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromIntList(&output, (*C.int)(&t[0]))))
    //     break
    case []float32:  // TODO: implement C-level function for float32
        datums := []float64{}
        for _, datum := range t { datums = append(datums, float64(datum)) }
        internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromDoubleList(
            &output,
            (*C.double)(unsafe.Pointer(&datums[0])),
            C.int(len(datums)),
        )))
        break
    case []float64:
        internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromDoubleList(
            &output,
            (*C.double)(unsafe.Pointer(&t[0])),
            C.int(len(t)),
        )))
        break
    // case []complex64:
    //     internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromComplexDoubleList(&output, (*C.complexdouble)(&t[0]))))
    //     break
    // case []complex128:
    //     internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromComplexDoubleList(&output, (*C.complexdouble)(&t[0]))))
    //     break
    case []Tensor:
        tensors := []C.Tensor{}
        for _, tensor := range t { tensors = append(tensors, C.Tensor(*tensor.T)) }
        internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromTensorList(&output, (*C.Tensor)(&tensors[0]), C.int(len(tensors)))))
        break
    // case []IValue:
    //     values := []C.IValue{}
    //     for _, value := range t { values = append(values, C.IValue(*value.T)) }
    //     internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromList(&output, (*C.IValue)(&values[0]), C.int(len(values)))))
    //     break
    case Device:
        internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromDevice(&output, t.T)))
        break
    default:
        panic(fmt.Sprintf("IValue not supported for data of type %s", reflect.TypeOf(data)))
    }
    ivalue := IValue{(*unsafe.Pointer)(&output)}
    // Set the finalizer for the Go structure to free the heap-allocated C
    // memory when the garbage collector finalizes the object.
    runtime.SetFinalizer((*unsafe.Pointer)(ivalue.T), func(t *unsafe.Pointer) {
        C.Torch_IValue_Free(C.IValue(*t))
    })
    return ivalue
}

// ---------------------------------------------------------------------------
// MARK: Type checkers
// ---------------------------------------------------------------------------

// Return true if the IValue references a null pointer.
func (ivalue IValue) IsNil() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsNone(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a scalar.
func (ivalue IValue) IsScalar() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsScalar(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a boolean.
func (ivalue IValue) IsBool() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsBool(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is an integer.
func (ivalue IValue) IsInt() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsInt(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a double-precision floating-point value.
func (ivalue IValue) IsDouble() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsDouble(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a complex double-precision floating-point value.
func (ivalue IValue) IsComplexDouble() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsComplexDouble(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a string.
func (ivalue IValue) IsString() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsString(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a tensor.
func (ivalue IValue) IsTensor() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsTensor(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a tuple.
func (ivalue IValue) IsTuple() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsTuple(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a list.
func (ivalue IValue) IsList() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsList(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a list of booleans.
func (ivalue IValue) IsBoolList() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsBoolList(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a list of integers.
func (ivalue IValue) IsIntList() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsIntList(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a list of double-precision floats.
func (ivalue IValue) IsDoubleList() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsDoubleList(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a list of complex double-precision floats.
func (ivalue IValue) IsComplexDoubleList() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsComplexDoubleList(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a list of tensors.
func (ivalue IValue) IsTensorList() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsTensorList(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// // Return true if the IValue is an optional tensor list.
// func (ivalue IValue) IsOptionalTensorList() bool {
//  var output C.bool
//  internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsOptionalTensorList(&output, (C.IValue)(*ivalue.T))))
//  return bool(output)
// }

// Return true if the IValue is a dictionary.
func (ivalue IValue) IsGenericDict() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsGenericDict(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a torch device.
func (ivalue IValue) IsDevice() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsDevice(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a storage medium.
func (ivalue IValue) IsStorage() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsStorage(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a capsule.
func (ivalue IValue) IsCapsule() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsCapsule(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a custom class instance.
func (ivalue IValue) IsCustomClass() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsCustomClass(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a future.
func (ivalue IValue) IsFuture() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsFuture(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is an RRef.
func (ivalue IValue) IsRRef() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsRRef(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a quantizer.
func (ivalue IValue) IsQuantizer() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsQuantizer(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// // Return true if the IValue is a sym integer.
// func (ivalue IValue) IsSymInt() bool {
//  var output C.bool
//  internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsSymInt(&output, (C.IValue)(*ivalue.T))))
//  return bool(output)
// }

// // Return true if the IValue is a sym float.
// func (ivalue IValue) IsSymFloat() bool {
//  var output C.bool
//  internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsSymFloat(&output, (C.IValue)(*ivalue.T))))
//  return bool(output)
// }

// Return true if the IValue is an object.
func (ivalue IValue) IsObject() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsObject(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a module.
func (ivalue IValue) IsModule() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsModule(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a Python object instance.
func (ivalue IValue) IsPyObject() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsPyObject(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is an enumeration.
func (ivalue IValue) IsEnum() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsEnum(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a data stream (e.g., a CUDA stream.)
func (ivalue IValue) IsStream() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsStream(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a Python generator.
func (ivalue IValue) IsGenerator() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsGenerator(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Return true if the IValue is a void*.
func (ivalue IValue) IsPtrType() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsPtrType(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// ---------------------------------------------------------------------------
// MARK: Container length checkers
// ---------------------------------------------------------------------------

// If the ivalue is a tuple, return the number of items in the tuple.
func (ivalue IValue) LengthTuple() int64 {
    var output C.int64_t
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_LengthTuple(&output, (C.IValue)(*ivalue.T))))
    return int64(output)
}

// If the ivalue is a list, return the number of items in the list.
func (ivalue IValue) LengthList() int64 {
    var output C.int64_t
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_LengthList(&output, (C.IValue)(*ivalue.T))))
    return int64(output)
}

// If the ivalue is a dictionary, return the number of items in the dictionary.
func (ivalue IValue) LengthDict() int64 {
    var output C.int64_t
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_LengthDict(&output, (C.IValue)(*ivalue.T))))
    return int64(output)
}

// ---------------------------------------------------------------------------
// MARK: Data retrieval
// ---------------------------------------------------------------------------

// Convert the IValue to a null pointer (i.e., the string "None".)
func (ivalue IValue) ToNone() string {
    var output *C.char
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToNone(&output, (C.IValue)(*ivalue.T))))
    // free is not necessary until the C call succeeds and allocates the string
    defer C.free(unsafe.Pointer(output))
    return C.GoString(output)
}

// // Convert the IValue to a generic scalar representation.
// func (ivalue IValue) ToScalar()

// Convert the IValue to a boolean.
func (ivalue IValue) ToBool() bool {
    var output C.bool
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToBool(&output, (C.IValue)(*ivalue.T))))
    return bool(output)
}

// Convert the IValue to an integer.
func (ivalue IValue) ToInt() int {
    var output C.int32_t
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToInt(&output, (C.IValue)(*ivalue.T))))
    return int(output)
}

// Convert the IValue to a double-precision float.
func (ivalue IValue) ToDouble() float64 {
    var output C.double
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToDouble(&output, (C.IValue)(*ivalue.T))))
    return float64(output)
}

// Convert the IValue to a complex double-precision float.
func (ivalue IValue) ToComplexDouble() complex128 {
    var output C.complexdouble
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToComplexDouble(&output, (C.IValue)(*ivalue.T))))
    return complex128(output)
}

// Convert the IValue to a string.
func (ivalue IValue) ToString() string {
    var output *C.char
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToString(&output, (C.IValue)(*ivalue.T))))
    // free is not necessary until the C call succeeds and allocates the string
    defer C.free(unsafe.Pointer(output))
    return C.GoString(output)
}

// Convert the IValue to a tensor.
func (ivalue IValue) ToTensor() Tensor {
    var output C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToTensor(&output, (C.IValue)(*ivalue.T))))
    return NewTorchTensor((*unsafe.Pointer)(&output))
}

// // Convert the IValue to a list of booleans.
// func (ivalue IValue) ToBoolList()

// // Convert the IValue to a list of integers.
// func (ivalue IValue) ToIntList()

// // Convert the IValue to a list of double-precision floats.
// func (ivalue IValue) ToDoubleList()

// // Convert the IValue to a list of complex double-precision floats.
// func (ivalue IValue) ToComplexDoubleList()

// Convert the IValue to a list of tensors.
func (ivalue IValue) ToTensorList() []Tensor {
    pointers := make([]C.Tensor, ivalue.LengthList())
    if len(pointers) == 0 { return []Tensor{} }
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToTensorList(
        (*C.Tensor)(unsafe.Pointer(&pointers[0])),
        C.int64_t(len(pointers)),
        (C.IValue)(*ivalue.T),
    )))
    // Wrap the pointers with finalized Go structs
    tensors := []Tensor{}
    for index, _ := range pointers {
        pointer := (C.Tensor)(pointers[index])
        tensors = append(tensors, NewTorchTensor((*unsafe.Pointer)(&pointer)))
    }
    return tensors
}

// // Convert the IValue to an optional tensor list.
// func (ivalue IValue) ToOptionalTensorList()

// Convert the IValue to a generic list of IValues (as a slice.)
func (ivalue IValue) ToList() []IValue {
    pointers := make([]C.IValue, ivalue.LengthList())
    if len(pointers) == 0 { return []IValue{} }
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToList(
        (*C.IValue)(unsafe.Pointer(&pointers[0])),
        C.int64_t(len(pointers)),
        (C.IValue)(*ivalue.T),
    )))
    // Wrap the pointers with finalized Go structs
    ivalues := []IValue{}
    for index, _ := range pointers {
        pointer := (C.IValue)(pointers[index])
        ivalue := IValue{(*unsafe.Pointer)(&pointer)}
        // Set the finalizer for the Go structure to free the heap-allocated C
        // memory when the garbage collector finalizes the object.
        runtime.SetFinalizer((*unsafe.Pointer)(ivalue.T), func(t *unsafe.Pointer) {
            C.Torch_IValue_Free(C.IValue(*t))
        })
        ivalues = append(ivalues, ivalue)
    }
    return ivalues
}

// Convert the IValue to a generic tuple of IValues (as a slice.)
func (ivalue IValue) ToTuple() []IValue {
    pointers := make([]C.IValue, ivalue.LengthTuple())
    // Empty tuples are impossible by design, so if LengthTuple returns, it is
    // guaranteed to return a non-null integer. Still check for 0 in the case
    // that sanity breaks.
    if len(pointers) == 0 { return []IValue{} }
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToTuple(
        (*C.IValue)(unsafe.Pointer(&pointers[0])),
        C.int64_t(len(pointers)),
        (C.IValue)(*ivalue.T),
    )))
    // Wrap the pointers with finalized Go structs
    ivalues := []IValue{}
    for index, _ := range pointers {
        pointer := (C.IValue)(pointers[index])
        ivalue := IValue{(*unsafe.Pointer)(&pointer)}
        // Set the finalizer for the Go structure to free the heap-allocated C
        // memory when the garbage collector finalizes the object.
        runtime.SetFinalizer((*unsafe.Pointer)(ivalue.T), func(t *unsafe.Pointer) {
            C.Torch_IValue_Free(C.IValue(*t))
        })
        ivalues = append(ivalues, ivalue)
    }
    return ivalues
}

// Convert the IValue to a generic dictionary of IValues (as a map of interfaces
// to IValues.)
func (ivalue IValue) ToGenericDict() map[interface{}]IValue {
    length := ivalue.LengthDict()
    if length == 0 { return map[interface{}]IValue{} }
    keyPointers := make([]C.IValue, length)
    valPointers := make([]C.IValue, length)
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToGenericDict(
        (*C.IValue)(unsafe.Pointer(&keyPointers[0])),
        (*C.IValue)(unsafe.Pointer(&valPointers[0])),
        C.int64_t(length),
        (C.IValue)(*ivalue.T),
    )))
    // Parse the zipped dictionary into a Go map.
    var output = make(map[interface{}]IValue)
    for index, _ := range keyPointers {
        // Setup the 'key' IValue. The key will only be used internally, so we
        // can defer the freeing of the C memory instead of the usual garbage
        // collection approach using runtime.SetFinalizer.
        keyPointer := (C.IValue)(keyPointers[index])
        keyIValue := IValue{(*unsafe.Pointer)(&keyPointer)}
        defer C.Torch_IValue_Free(keyPointer)
        // Switch over the data-type of the IValue. The list of supported types
        // is derived from the definition of Dict in aten/src/ATen/core/Dict.h
        var key interface{}
        if        keyIValue.IsInt() {
            key = keyIValue.ToInt()
        } else if keyIValue.IsString() {
            key = keyIValue.ToString()
        } else if keyIValue.IsDouble() {
            key = keyIValue.ToDouble()
        } else if keyIValue.IsComplexDouble() {
            key = keyIValue.ToComplexDouble()
        } else if keyIValue.IsBool() {
            key = keyIValue.ToBool()
        } else if keyIValue.IsTensor() {  // TODO: will tensor keys work as intended?
            key = keyIValue.ToTensor()
        } else {
            panic(
                "Found unexpected key type, did the libtorch API change? " +
                "Cross-reference with ivalue.go and aten/src/ATen/core/Dict.h",
            )
        }
        // Setup the 'value' IValue. The output map values are of type IValue,
        // so we need to setup the runtime finalizer to free these values when
        // the garbage collector triggers.
        valPointer := (C.IValue)(valPointers[index])
        value := IValue{(*unsafe.Pointer)(&valPointer)}
        runtime.SetFinalizer((*unsafe.Pointer)(value.T), func(t *unsafe.Pointer) {
            C.Torch_IValue_Free(C.IValue(*t))
        })
        output[key] = value
    }
    return output
}

// Convert the IValue to a torch device.
func (ivalue IValue) ToDevice() Device {
    var output C.Device
    internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToDevice(&output, (C.IValue)(*ivalue.T))))
    return Device{output}
}

// func (ivalue IValue) ToStorage()
// func (ivalue IValue) ToCapsule()
// func (ivalue IValue) ToCustomClass()
// func (ivalue IValue) ToFuture()
// func (ivalue IValue) ToRRef()
// func (ivalue IValue) ToQuantizer()
// func (ivalue IValue) ToSymInt()
// func (ivalue IValue) ToSymFloat()
// func (ivalue IValue) ToObject()
// func (ivalue IValue) ToModule()
// func (ivalue IValue) ToPyObject()
// func (ivalue IValue) ToEnum()
// func (ivalue IValue) ToStream()
// func (ivalue IValue) ToGenerator()
// func (ivalue IValue) ToPtrType()
