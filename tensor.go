// Go bindings for at::Tensor.
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

// #cgo CFLAGS: -I ${SRCDIR}
// #cgo LDFLAGS: -L ${SRCDIR}/build -Wl,-rpath ${SRCDIR}/build -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/build/libtorch/lib -Wl,-rpath ${SRCDIR}/build/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include <stdio.h>
// #include <stdlib.h>
// #include "cgotorch/cgotorch.h"
import "C"
import (
    "fmt"
    "unsafe"
    "reflect"
    "runtime"
    internal "github.com/Kautenja/gotorch/internal"
)

// Tensor wraps a pointer to a C.Tensor as an unsafe Pointer.
type Tensor struct {
    T *unsafe.Pointer
}

// Create a new tensor.
func NewTorchTensor(tensor *unsafe.Pointer) Tensor {
    runtime.SetFinalizer(tensor, func(ct *unsafe.Pointer) {
        C.Torch_Tensor_Close(C.Tensor(*ct))
    })
    return Tensor{tensor}
}

// Create a tensor view that wraps around existing contiguous memory pointed to
// by data, of given data-type, and with given size. This function does not
// copy the data buffer so in-place operations performed on the tensor will
// mutate the input buffer.
func TensorFromBlob(data unsafe.Pointer, dtype Dtype, sizes []int64) Tensor {
    var tensor C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_FromBlob(&tensor, data,
        C.int8_t(dtype),
        (*C.int64_t)(unsafe.Pointer(&sizes[0])),
        C.int64_t(len(sizes)),
    )))
    return NewTorchTensor((*unsafe.Pointer)(&tensor))
}

// Create a new tensor that clones existing contiguous memory pointed to by
// data, of given data-type, and with given size. This function copies the
// input data, so subsequent in-place operations performed on the tensor will
// not mutate the input data.
func NewTensorFromBlob(data unsafe.Pointer, dtype Dtype, sizes []int64) Tensor {
    var tensor C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor(&tensor, data,
        C.int8_t(dtype),
        (*C.int64_t)(unsafe.Pointer(&sizes[0])),
        C.int64_t(len(sizes)),
    )))
    return NewTorchTensor((*unsafe.Pointer)(&tensor))
}

// Create a new tensor from a Go slice.
func NewTensor(data interface{}) Tensor {
    // Ensure that the input data is a slice.
    if reflect.TypeOf(data).Kind() != reflect.Slice {
        panic(fmt.Sprintf("Expected slice but got data of type %v", reflect.TypeOf(data).Kind()))
    }
    // Reflect information about the tensor size and data type.
    sizes, kind := internal.GetSizesAndKindOfSlice(data)
    dtype := GetDtypeOfKind(kind)
    if dtype == Invalid {
        panic(fmt.Sprintf("Unrecognized dtype kind %v", kind))
    }
    // Convert the data a 1-dimensional buffer
    flat_data := internal.FlattenSlice(data, kind)
    header := (*reflect.SliceHeader)(unsafe.Pointer(&flat_data))
    // Create a new tensor from the blob (that copies the data.)
    return NewTensorFromBlob(unsafe.Pointer(header.Data), dtype, sizes)
}

// Convert a tensor to a raw binary representation as a byte slice. Note that
// this is a copy-free operation and simply returns a slice with it's header
// updated to point to the underlying tensor data. If the tensor is garbage
// collected, the slice will be invalidated.
func (tensor Tensor) ToBytes() []byte {
    // Create a pointer to reference the underlying data.
    var buffer *C.uint8_t
    internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_ToBytes(&buffer, C.Tensor(*tensor.T))))
    // Create a blank byte slice and update the header to reference tensor data.
    var slice []byte
    header := (*reflect.SliceHeader)(unsafe.Pointer(&slice))
    header.Len = int(tensor.Dtype().NumBytes() * tensor.Numel())
    header.Cap = header.Len
    header.Data = uintptr(unsafe.Pointer(buffer))
    return slice
}

// Create a clone of an existing tensor.
func (tensor Tensor) Clone() Tensor {
    var output C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_Clone(&output, C.Tensor(*tensor.T))))
    return NewTorchTensor((*unsafe.Pointer)(&output))
}

// Convert the tensor to a readable string representation.
func (tensor Tensor) String() string {
    cstring := C.Torch_Tensor_String(C.Tensor(*tensor.T))
    defer C.free(unsafe.Pointer(cstring))
    output := C.GoString(cstring)
    return output
}

// Save the tensor to the given path.
func (tensor Tensor) Save(path string) error {
    // Wrap the GoString with a C string and defer the release of the memory.
    path_cstring := C.CString(path)
    defer C.free(unsafe.Pointer(path_cstring))
    // Attempt to save the tensor to the given path and catch any errors.
    if err := unsafe.Pointer(C.Torch_Tensor_Save(path_cstring, C.Tensor(*tensor.T))); err != nil {
        return internal.NewTorchError(err)
    }
    return nil
}

// Load a tensor from the given path.
func Load(path string) (Tensor, error) {
    // Wrap the GoString with a C string and defer the release of the memory.
    path_cstring := C.CString(path)
    defer C.free(unsafe.Pointer(path_cstring))
    // Attempt to load the tensor from the given path and catch any errors.
    var tensor C.Tensor
    if err := unsafe.Pointer(C.Torch_Tensor_Load(&tensor, path_cstring)); err != nil {
        return Tensor{}, internal.NewTorchError(err)
    }
    return NewTorchTensor((*unsafe.Pointer)(&tensor)), nil
}

// Encode a tensor into a pickled representation. Tensors are copied to the CPU
// before encoding.
func (tensor Tensor) Encode() ([]byte, error) {
    var buffer C.ByteBuffer
    err := unsafe.Pointer(C.Torch_Tensor_Encode((*C.ByteBuffer)(unsafe.Pointer(&buffer)), C.Tensor(*tensor.T)))
    if err != nil {
        return nil, internal.NewTorchError(err)
    }
    bytes := C.GoBytes(C.ByteBuffer_Data(buffer), C.int(int(int64(C.ByteBuffer_Size(buffer)))))
    C.ByteBuffer_Free(buffer)
    return bytes, nil
}

// Decode a pickled tensor back into a structured numerical format.
func Decode(buffer []byte) (Tensor, error) {
    var tensor C.Tensor
    err := unsafe.Pointer(C.Torch_Tensor_Decode(&tensor, C.CBytes(buffer), C.int64_t(int64(len(buffer)))))
    if err != nil {
        return Tensor{}, internal.NewTorchError(err)
    }
    return NewTorchTensor((*unsafe.Pointer)(&tensor)), nil
}

// Return the data-type of the tensor.
func (tensor Tensor) Dtype() Dtype {
    var dtype Dtype
    internal.PanicOnCException(unsafe.Pointer(
        C.Torch_Tensor_Dtype((*C.int8_t)(unsafe.Pointer(&dtype)), C.Tensor(*tensor.T)),
    ))
    return dtype
}

// Return the number of dimensions the tensor occupies.
func (tensor Tensor) Dim() int64 {
    var dim int64
    internal.PanicOnCException(unsafe.Pointer(
        C.Torch_Tensor_Dim((*C.int64_t)(&dim), C.Tensor(*tensor.T)),
    ))
    return dim
}

// Return the shape of the tensor data.
func (tensor Tensor) Shape() []int64 {
    shape := make([]int64, tensor.Dim())
    if len(shape) == 0 {
        return shape
    }
    internal.PanicOnCException(unsafe.Pointer(
        C.Torch_Tensor_Shape((*C.int64_t)(unsafe.Pointer(&shape[0])), C.Tensor(*tensor.T)),
    ))
    return shape
}

// Create a new tensor with an updated view of the underlying data.
func (tensor Tensor) View(shape ...int64) Tensor {
    var output C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_View(
        &output,
        C.Tensor(*tensor.T),
        (*C.int64_t)(unsafe.Pointer(&shape[0])),
        C.int64_t(len(shape)),
    )))
    return NewTorchTensor((*unsafe.Pointer)(&output))
}

// Create a new tensor with an updated view of the underlying data that matches
// that of a reference tensor.
func (tensor Tensor) ViewAs(other Tensor) Tensor {
    var output C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_ViewAs(
        &output,
        C.Tensor(*tensor.T),
        C.Tensor(*other.T),
    )))
    return NewTorchTensor((*unsafe.Pointer)(&output))
}

// Create a new tensor with an updated shape.
func (tensor Tensor) Reshape(shape ...int64) Tensor {
    var output C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_Reshape(
        &output,
        C.Tensor(*tensor.T),
        (*C.int64_t)(unsafe.Pointer(&shape[0])),
        C.int64_t(len(shape)),
    )))
    return NewTorchTensor((*unsafe.Pointer)(&output))
}

// Create a new tensor with an updated shape.
func Reshape(tensor Tensor, shape ...int64) Tensor {
    return tensor.Reshape(shape...)
}

// Create a new tensor with an updated shape according to a reference tensor.
func (tensor Tensor) ReshapeAs(other Tensor) Tensor {
    var output C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_ReshapeAs(
        &output,
        C.Tensor(*tensor.T),
        C.Tensor(*other.T),
    )))
    return NewTorchTensor((*unsafe.Pointer)(&output))
}

// Expand the dimensions of the tensor.
func (tensor Tensor) Expand(shape ...int64) Tensor {
    var output C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_Expand(
        &output,
        C.Tensor(*tensor.T),
        (*C.int64_t)(unsafe.Pointer(&shape[0])),
        C.int64_t(len(shape)),
    )))
    return NewTorchTensor((*unsafe.Pointer)(&output))
}

// Expand the dimensions of the tensor to match that of a reference.
func (tensor Tensor) ExpandAs(other Tensor) Tensor {
    var output C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_ExpandAs(
        &output,
        C.Tensor(*tensor.T),
        C.Tensor(*other.T),
    )))
    return NewTorchTensor((*unsafe.Pointer)(&output))
}

// Sets the tensor data to that of a separate reference tensor.
func (tensor Tensor) SetData(b Tensor) {
    internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_SetData(C.Tensor(*tensor.T), C.Tensor(*b.T))))
}

// SetData sets the tensor data held by b to a
func (tensor Tensor) Copy_(b Tensor) {
    internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_Copy_(C.Tensor(*tensor.T), C.Tensor(*b.T))))
}

// Create a new tensor with data cast to given type.
func (tensor Tensor) CastTo(dtype Dtype) Tensor {
    var output C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_CastTo(
        &output,
        C.Tensor(*tensor.T),
        C.int8_t(dtype),
    )))
    return NewTorchTensor((*unsafe.Pointer)(&output))
}

// Create a new tensor with data copied to given device.
func (tensor Tensor) CopyTo(device Device) Tensor {
    var output C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_CopyTo(
        &output,
        C.Tensor(*tensor.T),
        device.T,
    )))
    return NewTorchTensor((*unsafe.Pointer)(&output))
}

// Create a new tensor with data copied to given device and cast to given type.
func (tensor Tensor) To(device Device, dtype Dtype) Tensor {
    var output C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_To(
        &output,
        C.Tensor(*tensor.T),
        device.T,
        C.int8_t(dtype),
    )))
    return NewTorchTensor((*unsafe.Pointer)(&output))
}

// // Return a tensor in pinned memory. Pinned memory requires CUDA.
// func (tensor Tensor) PinMemory() Tensor {
//     if !cuda.IsAvailable() {
//         return tensor
//     }
//     var t C.Tensor
//     internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_PinMemory(&t, C.Tensor(*tensor.T))))
//     return NewTorchTensor((*unsafe.Pointer)(&t))
// }

// Return true if the tensor is taping gradients.
func (tensor Tensor) RequiresGrad() bool {
    var requiresGrad C.bool
    C.Torch_Tensor_RequiresGrad(&requiresGrad, C.Tensor(*tensor.T))
    return bool(requiresGrad)
}

// Set the gradient taping state of the tensor to a new value.
func (tensor Tensor) SetRequiresGrad(requiresGrad bool) {
    C.Torch_Tensor_SetRequiresGrad(C.Tensor(*tensor.T), C.bool(requiresGrad))
}

// Compute gradients based on the results of a forward pass through the tensor.
func (tensor Tensor) Backward() {
    internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_Backward(C.Tensor(*tensor.T))))
}

// Access the underlying gradients of the tensor.
func (tensor Tensor) Grad() Tensor {
    var output C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_Grad(&output, C.Tensor(*tensor.T))))
    return NewTorchTensor((*unsafe.Pointer)(&output))
}

// Create a new tensor with any taped gradients detached and grad disabled.
func (tensor Tensor) Detach() Tensor {
    var output C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_Detach(&output, C.Tensor(*tensor.T))))
    return NewTorchTensor((*unsafe.Pointer)(&output))
}

// Access elements in a tensor using an index tensor for reference.
func (tensor Tensor) Index(index Tensor) Tensor {
    var output C.Tensor
    internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_Index(
        &output,
        C.Tensor(*tensor.T),
        C.Tensor(*index.T),
    )))
    return NewTorchTensor((*unsafe.Pointer)(&output))
}

// Return the value of this tensor as a standard Go number. This only works for
// tensors with one element.
//
// Users can do type assertion and get the value like:
//
// ```
// if value, ok := a.Item().(float64); ok {
//     // process the value
// }
// ```
//
// This function currently only supports signed data types.
//
func (tensor Tensor) Item() interface{} {
    dtype := tensor.Dtype()
    switch dtype {
    case Byte:
        var output byte
        internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_ItemUint8((*C.uint8_t)(unsafe.Pointer(&output)), C.Tensor(*tensor.T))))
        return output
    case Char:
        var output int8
        internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_ItemInt8((*C.int8_t)(&output), C.Tensor(*tensor.T))))
        return output
    case Short:
        var output int16
        internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_ItemInt16((*C.int16_t)(&output), C.Tensor(*tensor.T))))
        return output
    case Int:
        var output int32
        internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_ItemInt32((*C.int32_t)(&output), C.Tensor(*tensor.T))))
        return output
    case Long:
        var output int64
        internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_ItemInt64((*C.int64_t)(&output), C.Tensor(*tensor.T))))
        return output
    case Half, Float:
        var output float32
        internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_ItemFloat32((*C.float)(&output), C.Tensor(*tensor.T))))
        return output
    case Double:
        var output float64
        internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_ItemFloat64((*C.double)(&output), C.Tensor(*tensor.T))))
        return output
    // case ComplexHalf: TODO
    // case ComplexFloat: TODO
    // case ComplexDouble: TODO
    case Bool:
        var output bool
        internal.PanicOnCException(unsafe.Pointer(C.Torch_Tensor_ItemBool((*C.bool)(&output), C.Tensor(*tensor.T))))
        return output
    // case QInt8: TODO
    // case QUInt8: TODO
    // case QInt32: TODO
    // case BFloat16: TODO
    default:  // Invalid, etc.
        panic(fmt.Sprintf("Dtype %d is not supported by Item", dtype))
    }
}
