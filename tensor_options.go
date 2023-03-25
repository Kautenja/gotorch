// Go bindings for at::TensorOptions.
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
// #include "cgotorch/cgotorch.h"
import "C"
import (
	"unsafe"
	"runtime"
	"github.com/Kautenja/gotorch/internal"
)

// TensorOptions wraps a C.TensorOptions.
type TensorOptions struct {
	Pointer C.TensorOptions
}

// Create a new TensorOptions.
func NewTensorOptions() (options *TensorOptions) {
	options = &TensorOptions{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_TensorOptions(&options.Pointer)))
	return options.withFinalizerSet()
}

// Free the heap-allocated C memory when the garbage collector finalizes.
// Normally we would define this internally within a global constructor, but
// the C-level interface of the TensorOptions structure is "functional" and
// requires a copy for each change to the object's state. This means every
// state change implies a new object and needs to finalize the resulting go
// struct. This is all due to a decision within the C10 library to make the
// `set_*` semantics of TensorOptions private to the structure. Ideally we
// would create a single TensorOptions and mutate its state and mock the
// functional interface here at the Go layer, but this is not possible ATM.
func (options *TensorOptions) withFinalizerSet() *TensorOptions {
	runtime.SetFinalizer(options, (*TensorOptions).free)
	return options
}

// Free a tensor options from memory.
func (options *TensorOptions) free() {
	if options.Pointer == nil {
		panic("Attempting to free a tensor options that has already been freed!")
	}
	C.Torch_TensorOptions_Free(options.Pointer)
	options.Pointer = nil
}

// Create a new TensorOptions with the given data type.
func (options *TensorOptions) Dtype(value Dtype) *TensorOptions {
	output := &TensorOptions{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_TensorOptions_Dtype(
		&output.Pointer,
		options.Pointer,
		C.int8_t(value),
	)))
	return output.withFinalizerSet()
}

// TODO:
// enum class Layout : int8_t {
//   Strided,
//   Sparse,
//   SparseCsr,
//   Mkldnn,
//   SparseCsc,
//   SparseBsr,
//   SparseBsc,
//   NumOptions
// };

// // Create a new TensorOptions with the given data layout.
// func (options *TensorOptions) Layout(value Layout) *TensorOptions {
//     output := &TensorOptions{}
//     internal.PanicOnCException(unsafe.Pointer(C.Torch_TensorOptions_Layout(
//         &output.Pointer,
//         options.Pointer,
//         C.int8_t(value),
//     )))
//     return output.withFinalizerSet()
// }

// TODO:
// enum class MemoryFormat : int8_t {
//   Contiguous,
//   Preserve,
//   ChannelsLast,
//   ChannelsLast3d,
//   NumOptions
// };

// // Create a new TensorOptions with the given memory format.
// func (options *TensorOptions) MemoryFormat(value MemoryFormat) *TensorOptions {
//     output := &TensorOptions{}
//     internal.PanicOnCException(unsafe.Pointer(C.Torch_TensorOptions_MemoryFormat(
//         &output.Pointer,
//         options.Pointer,
//         C.int8_t(value),
//     )))
//     return output.withFinalizerSet()
// }

// Create a new TensorOptions with the given compute device.
func (options *TensorOptions) Device(device Device) *TensorOptions {
	output := &TensorOptions{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_TensorOptions_Device(
		&output.Pointer,
		options.Pointer,
		(C.Device)(device.Pointer),
	)))
	return output.withFinalizerSet()
}

// Create a new TensorOptions with the given gradient taping state.
func (options *TensorOptions) RequiresGrad(requiresGrad bool) *TensorOptions {
	output := &TensorOptions{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_TensorOptions_RequiresGrad(
		&output.Pointer,
		options.Pointer,
		C.bool(requiresGrad),
	)))
	return output.withFinalizerSet()
}

// Create a new TensorOptions with the given memory pinning state.
func (options *TensorOptions) PinnedMemory(pinnedMemory bool) *TensorOptions {
	output := &TensorOptions{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_TensorOptions_PinnedMemory(
		&output.Pointer,
		options.Pointer,
		C.bool(pinnedMemory),
	)))
	return output.withFinalizerSet()
}
