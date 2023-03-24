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

// #cgo CPPFLAGS: -I/usr/local/include -I/usr/local/include/cgotorch
// #cgo LDFLAGS: -L/usr/local/lib -lc10 -ltorch_cpu -ltorch -lcgotorch
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

// IValue wraps a C.IValue.
type IValue struct {
	Pointer C.IValue
}

// ---------------------------------------------------------------------------
// MARK: Constructors
// ---------------------------------------------------------------------------

// Create a new IValue from arbitrary data.
func NewIValue(data interface{}) (ivalue *IValue) {
	ivalue = &IValue{}
	switch t := data.(type) {
	case nil:
		internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromNone(&ivalue.Pointer)))
		break
	case bool:
		internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromBool(&ivalue.Pointer, C.bool(t))))
		break
	case int:
		internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromInt(&ivalue.Pointer, C.int(t))))
		break
	case int32:
		internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromInt(&ivalue.Pointer, C.int(t))))
		break
	// case int64:
	//     internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromInt(&ivalue.Pointer, C.int(t))))
	//     break
	case float32:
		internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromDouble(&ivalue.Pointer, C.double(t))))
		break
	case float64:
		internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromDouble(&ivalue.Pointer, C.double(t))))
		break
	case complex64:
		internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromComplexDouble(&ivalue.Pointer, C.complexdouble(t))))
		break
	case complex128:
		internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromComplexDouble(&ivalue.Pointer, C.complexdouble(t))))
		break
	case string:
		stringData := C.CString(t)
		defer C.free(unsafe.Pointer(stringData))
		internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromString(&ivalue.Pointer, stringData)))
		break
	case Tensor:
		internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromTensor(&ivalue.Pointer, (C.Tensor)(*t.T))))
		break
	case []bool:
		internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromBoolList(
			&ivalue.Pointer,
			(*C.bool)(unsafe.Pointer(&t[0])),
			C.int(len(t)),
		)))
		break
	// case []int:
	//     internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromIntList(
	//         &ivalue.Pointer,
	//         (*C.int)(unsafe.Pointer(&t[0])),
	//         C.int(len(t)),
	//     )))
	//     break
	// case []int32:
	//     internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromIntList(
	//         &ivalue.Pointer,
	//         (*C.int)(unsafe.Pointer(&t[0])),
	//         C.int(len(t)),
	//     )))
	//     break
	// case []int64:
	//     internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromIntList(&ivalue.Pointer, (*C.int)(&t[0]))))
	//     break
	case []float32:  // TODO: implement C-level function for float32
		datums := []float64{}
		for _, datum := range t { datums = append(datums, float64(datum)) }
		internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromDoubleList(
			&ivalue.Pointer,
			(*C.double)(unsafe.Pointer(&datums[0])),
			C.int(len(datums)),
		)))
		break
	case []float64:
		internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromDoubleList(
			&ivalue.Pointer,
			(*C.double)(unsafe.Pointer(&t[0])),
			C.int(len(t)),
		)))
		break
	// case []complex64:
	//     internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromComplexDoubleList(&ivalue.Pointer, (*C.complexdouble)(&t[0]))))
	//     break
	// case []complex128:
	//     internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromComplexDoubleList(&ivalue.Pointer, (*C.complexdouble)(&t[0]))))
	//     break
	case []Tensor:
		tensors := []C.Tensor{}
		for _, tensor := range t { tensors = append(tensors, C.Tensor(*tensor.T)) }
		internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromTensorList(
			&ivalue.Pointer,
			(*C.Tensor)(&tensors[0]),
			C.int(len(tensors)),
		)))
		break
	// case []IValue:
	//     values := []C.IValue{}
	//     for _, value := range t { values = append(values, C.IValue(*value.Pointer)) }
	//     internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromList(&ivalue.Pointer, (*C.IValue)(&values[0]), C.int(len(values)))))
	//     break
	case *Device:
		internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_FromDevice(&ivalue.Pointer, t.Pointer)))
		break
	default:
		panic(fmt.Sprintf("IValue not supported for data of type %s", reflect.TypeOf(data)))
	}
	runtime.SetFinalizer(ivalue, (*IValue).free)
	return
}

// Free an ivalue from memory.
func (ivalue *IValue) free() {
	if ivalue.Pointer == nil {
		panic("Attempting to free an ivalue that has already been freed!")
	}
	C.Torch_IValue_Free(ivalue.Pointer)
	ivalue.Pointer = nil
}

// ---------------------------------------------------------------------------
// MARK: Type checkers
// ---------------------------------------------------------------------------

// Return true if the IValue references a null pointer.
func (ivalue *IValue) IsNil() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsNone(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a scalar.
func (ivalue *IValue) IsScalar() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsScalar(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a boolean.
func (ivalue *IValue) IsBool() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsBool(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is an integer.
func (ivalue *IValue) IsInt() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsInt(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a double-precision floating-point value.
func (ivalue *IValue) IsDouble() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsDouble(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a complex double-precision floating-point value.
func (ivalue *IValue) IsComplexDouble() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsComplexDouble(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a string.
func (ivalue *IValue) IsString() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsString(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a tensor.
func (ivalue *IValue) IsTensor() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsTensor(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a tuple.
func (ivalue *IValue) IsTuple() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsTuple(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a list.
func (ivalue *IValue) IsList() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsList(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a list of booleans.
func (ivalue *IValue) IsBoolList() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsBoolList(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a list of integers.
func (ivalue *IValue) IsIntList() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsIntList(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a list of double-precision floats.
func (ivalue *IValue) IsDoubleList() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsDoubleList(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a list of complex double-precision floats.
func (ivalue *IValue) IsComplexDoubleList() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsComplexDoubleList(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a list of tensors.
func (ivalue *IValue) IsTensorList() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsTensorList(&output, ivalue.Pointer)))
	return bool(output)
}

// // Return true if the IValue is an optional tensor list.
// func (ivalue *IValue) IsOptionalTensorList() bool {
//  var output C.bool
//  internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsOptionalTensorList(&output, ivalue.Pointer)))
//  return bool(output)
// }

// Return true if the IValue is a dictionary.
func (ivalue *IValue) IsGenericDict() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsGenericDict(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a torch device.
func (ivalue *IValue) IsDevice() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsDevice(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a storage medium.
func (ivalue *IValue) IsStorage() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsStorage(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a capsule.
func (ivalue *IValue) IsCapsule() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsCapsule(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a custom class instance.
func (ivalue *IValue) IsCustomClass() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsCustomClass(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a future.
func (ivalue *IValue) IsFuture() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsFuture(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is an RRef.
func (ivalue *IValue) IsRRef() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsRRef(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a quantizer.
func (ivalue *IValue) IsQuantizer() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsQuantizer(&output, ivalue.Pointer)))
	return bool(output)
}

// // Return true if the IValue is a sym integer.
// func (ivalue *IValue) IsSymInt() bool {
//  var output C.bool
//  internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsSymInt(&output, ivalue.Pointer)))
//  return bool(output)
// }

// // Return true if the IValue is a sym float.
// func (ivalue *IValue) IsSymFloat() bool {
//  var output C.bool
//  internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsSymFloat(&output, ivalue.Pointer)))
//  return bool(output)
// }

// Return true if the IValue is an object.
func (ivalue *IValue) IsObject() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsObject(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a module.
func (ivalue *IValue) IsModule() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsModule(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a Python object instance.
func (ivalue *IValue) IsPyObject() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsPyObject(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is an enumeration.
func (ivalue *IValue) IsEnum() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsEnum(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a data stream (e.g., a CUDA stream.)
func (ivalue *IValue) IsStream() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsStream(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a Python generator.
func (ivalue *IValue) IsGenerator() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsGenerator(&output, ivalue.Pointer)))
	return bool(output)
}

// Return true if the IValue is a void*.
func (ivalue *IValue) IsPtrType() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_IsPtrType(&output, ivalue.Pointer)))
	return bool(output)
}

// ---------------------------------------------------------------------------
// MARK: Container length checkers
// ---------------------------------------------------------------------------

// If the ivalue is a tuple, return the number of items in the tuple.
func (ivalue *IValue) LengthTuple() int64 {
	var output C.int64_t
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_LengthTuple(&output, ivalue.Pointer)))
	return int64(output)
}

// If the ivalue is a list, return the number of items in the list.
func (ivalue *IValue) LengthList() int64 {
	var output C.int64_t
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_LengthList(&output, ivalue.Pointer)))
	return int64(output)
}

// If the ivalue is a dictionary, return the number of items in the dictionary.
func (ivalue *IValue) LengthDict() int64 {
	var output C.int64_t
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_LengthDict(&output, ivalue.Pointer)))
	return int64(output)
}

// ---------------------------------------------------------------------------
// MARK: Data retrieval
// ---------------------------------------------------------------------------

// Convert the IValue to a null pointer (i.e., the string "None".)
func (ivalue *IValue) ToNone() string {
	var output *C.char
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToNone(&output, ivalue.Pointer)))
	// free is not necessary until the C call succeeds and allocates the string
	defer C.free(unsafe.Pointer(output))
	return C.GoString(output)
}

// // Convert the IValue to a generic scalar representation.
// func (ivalue *IValue) ToScalar()

// Convert the IValue to a boolean.
func (ivalue *IValue) ToBool() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToBool(&output, ivalue.Pointer)))
	return bool(output)
}

// Convert the IValue to an integer.
func (ivalue *IValue) ToInt() int {
	var output C.int32_t
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToInt(&output, ivalue.Pointer)))
	return int(output)
}

// Convert the IValue to a double-precision float.
func (ivalue *IValue) ToDouble() float64 {
	var output C.double
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToDouble(&output, ivalue.Pointer)))
	return float64(output)
}

// Convert the IValue to a complex double-precision float.
func (ivalue *IValue) ToComplexDouble() complex128 {
	var output C.complexdouble
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToComplexDouble(&output, ivalue.Pointer)))
	return complex128(output)
}

// Convert the IValue to a string.
func (ivalue *IValue) ToString() string {
	var output *C.char
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToString(&output, ivalue.Pointer)))
	// free is not necessary until the C call succeeds and allocates the string
	defer C.free(unsafe.Pointer(output))
	return C.GoString(output)
}

// Convert the IValue to a tensor.
func (ivalue *IValue) ToTensor() Tensor {
	var output C.Tensor
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToTensor(&output, ivalue.Pointer)))
	return NewTorchTensor((*unsafe.Pointer)(&output))
}

// // Convert the IValue to a list of booleans.
// func (ivalue *IValue) ToBoolList()

// // Convert the IValue to a list of integers.
// func (ivalue *IValue) ToIntList()

// // Convert the IValue to a list of double-precision floats.
// func (ivalue *IValue) ToDoubleList()

// // Convert the IValue to a list of complex double-precision floats.
// func (ivalue *IValue) ToComplexDoubleList()

// Convert the IValue to a list of tensors.
func (ivalue *IValue) ToTensorList() []Tensor {
	pointers := make([]C.Tensor, ivalue.LengthList())
	if len(pointers) == 0 { return []Tensor{} }
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToTensorList(
		(*C.Tensor)(unsafe.Pointer(&pointers[0])),
		C.int64_t(len(pointers)),
		ivalue.Pointer,
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
// func (ivalue *IValue) ToOptionalTensorList()

// Convert the IValue to a generic list of IValues (as a slice.)
func (ivalue *IValue) ToList() []*IValue {
	pointers := make([]C.IValue, ivalue.LengthList())
	if len(pointers) == 0 { return []*IValue{} }
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToList(
		&pointers[0],
		C.int64_t(len(pointers)),
		ivalue.Pointer,
	)))
	// Wrap the pointers with finalized Go structs
	ivalues := []*IValue{}
	for _, pointer := range pointers {
		ivalue := &IValue{pointer}
		runtime.SetFinalizer(ivalue, (*IValue).free)
		ivalues = append(ivalues, ivalue)
	}
	return ivalues
}

// Convert the IValue to a generic tuple of IValues (as a slice.)
func (ivalue *IValue) ToTuple() []*IValue {
	pointers := make([]C.IValue, ivalue.LengthTuple())
	// Empty tuples are impossible by design, so if LengthTuple returns, it is
	// guaranteed to return a non-null integer. Still check for 0 in the case
	// that sanity breaks.
	if len(pointers) == 0 { return []*IValue{} }
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToTuple(
		&pointers[0],
		C.int64_t(len(pointers)),
		ivalue.Pointer,
	)))
	// Wrap the pointers with finalized Go structs
	ivalues := []*IValue{}
	for _, pointer := range pointers {
		ivalue := &IValue{pointer}
		runtime.SetFinalizer(ivalue, (*IValue).free)
		ivalues = append(ivalues, ivalue)
	}
	return ivalues
}

// Convert the IValue to a generic dictionary of IValues (as a map of interfaces
// to IValues.)
func (ivalue *IValue) ToGenericDict() map[interface{}]*IValue {
	length := ivalue.LengthDict()
	if length == 0 { return map[interface{}]*IValue{} }
	keyPointers := make([]C.IValue, length)
	valPointers := make([]C.IValue, length)
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToGenericDict(
		&keyPointers[0],
		&valPointers[0],
		C.int64_t(length),
		ivalue.Pointer,
	)))
	// Parse the zipped dictionary into a Go map.
	var output = make(map[interface{}]*IValue)
	for index, keyPointer := range keyPointers {
		// Setup the 'key' IValue. The key will only be used internally, so we
		// can defer the freeing of the C memory instead of the usual garbage
		// collection approach using runtime.SetFinalizer.
		defer C.Torch_IValue_Free(keyPointer)
		keyIValue := IValue{keyPointer}
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
		output[key] = &IValue{valPointers[index]}
		runtime.SetFinalizer(output[key], (*IValue).free)
	}
	return output
}

// Convert the IValue to a torch device.
func (ivalue *IValue) ToDevice() (device *Device) {
	device = &Device{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IValue_ToDevice(&device.Pointer, ivalue.Pointer)))
	runtime.SetFinalizer(device, (*Device).free)
	return
}

// func (ivalue *IValue) ToStorage()
// func (ivalue *IValue) ToCapsule()
// func (ivalue *IValue) ToCustomClass()
// func (ivalue *IValue) ToFuture()
// func (ivalue *IValue) ToRRef()
// func (ivalue *IValue) ToQuantizer()
// func (ivalue *IValue) ToSymInt()
// func (ivalue *IValue) ToSymFloat()
// func (ivalue *IValue) ToObject()
// func (ivalue *IValue) ToModule()
// func (ivalue *IValue) ToPyObject()
// func (ivalue *IValue) ToEnum()
// func (ivalue *IValue) ToStream()
// func (ivalue *IValue) ToGenerator()
// func (ivalue *IValue) ToPtrType()
