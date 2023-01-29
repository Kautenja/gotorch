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

// #cgo CFLAGS: -I ${SRCDIR}
// #cgo LDFLAGS: -L ${SRCDIR}/build -Wl,-rpath ${SRCDIR}/build -lcgotorch
// #cgo LDFLAGS: -L ${SRCDIR}/build/libtorch/lib -Wl,-rpath ${SRCDIR}/build/libtorch/lib -lc10 -ltorch -ltorch_cpu
// #include <stdio.h>
// #include <stdlib.h>
// #include "cgotorch/cgotorch.h"
import "C"
import (
    "unsafe"
    "runtime"
    internal "github.com/Kautenja/gotorch/internal"
)

// TensorOptions wraps a pointer to a C.TensorOptions as an unsafe Pointer.
type TensorOptions struct {
    // A pointer to a C.TensorOptions.
    T *unsafe.Pointer
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
func (options TensorOptions) withFinalizerSet() TensorOptions {
    runtime.SetFinalizer((*unsafe.Pointer)(options.T), func(t *unsafe.Pointer) {
        C.Torch_TensorOptions_Free(C.TensorOptions(*t))
    })
    return options
}

// Create a new TensorOptions.
func NewTensorOptions() TensorOptions {
    var output C.TensorOptions
    internal.PanicOnCException(unsafe.Pointer(C.Torch_TensorOptions(&output)))
    return TensorOptions{(*unsafe.Pointer)(&output)}.withFinalizerSet()
}

// Create a new TensorOptions with the given data type.
func (options TensorOptions) Dtype(value Dtype) TensorOptions {
    var output C.TensorOptions
    internal.PanicOnCException(unsafe.Pointer(C.Torch_TensorOptions_Dtype(
        &output,
        C.TensorOptions(*options.T),
        C.int8_t(value),
    )))
    return TensorOptions{(*unsafe.Pointer)(&output)}.withFinalizerSet()
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
// func (options TensorOptions) Layout(value Layout) TensorOptions {
//     var output C.TensorOptions
//     internal.PanicOnCException(unsafe.Pointer(C.Torch_TensorOptions_Layout(
//         &output,
//         C.TensorOptions(*options.T),
//         C.int8_t(value),
//     )))
//     return TensorOptions{(*unsafe.Pointer)(&output)}.withFinalizerSet()
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
// func (options TensorOptions) MemoryFormat(value MemoryFormat) TensorOptions {
//     var output C.TensorOptions
//     internal.PanicOnCException(unsafe.Pointer(C.Torch_TensorOptions_MemoryFormat(
//         &output,
//         C.TensorOptions(*options.T),
//         C.int8_t(value),
//     )))
//     return TensorOptions{(*unsafe.Pointer)(&output)}.withFinalizerSet()
// }

// Create a new TensorOptions with the given compute device.
func (options TensorOptions) Device(device Device) TensorOptions {
    var output C.TensorOptions
    internal.PanicOnCException(unsafe.Pointer(C.Torch_TensorOptions_Device(
        &output,
        C.TensorOptions(*options.T),
        (C.Device)(device.T),
    )))
    return TensorOptions{(*unsafe.Pointer)(&output)}.withFinalizerSet()
}

// Create a new TensorOptions with the given gradient taping state.
func (options TensorOptions) RequiresGrad(requiresGrad bool) TensorOptions {
    var output C.TensorOptions
    internal.PanicOnCException(unsafe.Pointer(C.Torch_TensorOptions_RequiresGrad(
        &output,
        C.TensorOptions(*options.T),
        C.bool(requiresGrad),
    )))
    return TensorOptions{(*unsafe.Pointer)(&output)}.withFinalizerSet()
}

// Create a new TensorOptions with the given memory pinning state.
func (options TensorOptions) PinnedMemory(pinnedMemory bool) TensorOptions {
    var output C.TensorOptions
    internal.PanicOnCException(unsafe.Pointer(C.Torch_TensorOptions_PinnedMemory(
        &output,
        C.TensorOptions(*options.T),
        C.bool(pinnedMemory),
    )))
    return TensorOptions{(*unsafe.Pointer)(&output)}.withFinalizerSet()
}
