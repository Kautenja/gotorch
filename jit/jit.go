// Go bindings for torch::jit.
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

package jit

// #cgo CPPFLAGS: -I/usr/local/include -I/usr/local/include/cgotorch
// #cgo LDFLAGS: -L/usr/local/lib -lc10 -ltorch_cpu -ltorch -lcgotorch
// #include <stdio.h>
// #include <stdlib.h>
// #include "cgotorch/cgotorch.h"
import "C"
import (
	"runtime"
	"unsafe"
	"github.com/Kautenja/gotorch"
	internal "github.com/Kautenja/gotorch/internal"
)

// A container for a torch script module in C++.
type JitModule struct {
	Pointer C.JitModule
}

func Load(path string, device *torch.Device) (*JitModule, error) {
	module := &JitModule{}
	path_cstring := C.CString(path)
	defer C.free(unsafe.Pointer(path_cstring))
	internalErr := unsafe.Pointer(C.Torch_Jit_Load(
		&module.Pointer,
		path_cstring,
		(C.Device)(device.Pointer),
	))
	if internalErr != nil {
		return nil, internal.NewTorchError(internalErr)
	}
	runtime.SetFinalizer(module, (*JitModule).free)
	return module, nil
}

// Free a JIT module from memory.
func (module *JitModule) free() {
	if module.Pointer == nil {
		panic("Attempting to free a module that has already been freed!")
	}
	C.Torch_Jit_Module_Free(module.Pointer)
    module.Pointer = nil
}

// Save the module to the given path.
func (module *JitModule) Save(path string) error {
	// Wrap the GoString with a C string and defer the release of the memory.
	path_cstring := C.CString(path)
	defer C.free(unsafe.Pointer(path_cstring))
	// Attempt to save the tensor to the given path and catch any errors.
	err := unsafe.Pointer(C.Torch_Jit_Module_Save(path_cstring, module.Pointer))
	if err != nil {
		return internal.NewTorchError(err)
	}
	return nil
}

// Convert the module to a human-readable string representation.
func (module *JitModule) String() string {
	cstring := C.Torch_Jit_Module_String(module.Pointer)
	defer C.free(unsafe.Pointer(cstring))
	output := C.GoString(cstring)
	return output
}

// Return true if training features are enabled for the module, false otherwise.
func (module *JitModule) IsTraining() bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Jit_Module_IsTraining(&output, module.Pointer)))
	return bool(output)
}

// // Return true if the module is optimized for inference, false otherwise.
// func (module *JitModule) IsOptimized() bool {
//  var output C.bool
//  internal.PanicOnCException(unsafe.Pointer(C.Torch_Jit_Module_IsOptimized(&output, module.Pointer)))
//  return bool(output)
// }

// // Enable/disable JIT optimization features for the module.
// func (module *JitModule) SetOptimized(mode bool) *JitModule {
//  internal.PanicOnCException(unsafe.Pointer(C.Torch_Jit_Module_SetOptimized(module.Pointer, C.bool(mode))))
//  return module
// }

// Enable/disable training features for the module.
func (module *JitModule) Train(mode bool) *JitModule {
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Jit_Module_Train(module.Pointer, C.bool(mode))))
	return module
}

// Set the module to evaluation (e.g., inference) mode.
func (module *JitModule) Eval() *JitModule {
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Jit_Module_Eval(module.Pointer)))
	return module
}

// Cast the model's parameters to the given data-type in-place.
func (module *JitModule) CastTo(dtype torch.Dtype) *JitModule {
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Jit_Module_CastTo(
		module.Pointer,
		C.int8_t(dtype),
	)))
	return module
}

// Copy the model's parameters to the given compute accelerator in-place.
func (module *JitModule) CopyTo(device *torch.Device) *JitModule {
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Jit_Module_CopyTo(
		module.Pointer,
		(C.Device)(device.Pointer),
	)))
	return module
}

// Cast the model's parameters to the given data-type in-place.
func (module *JitModule) To(device *torch.Device, dtype torch.Dtype) *JitModule {
	return module.CopyTo(device).CastTo(dtype)
	// internal.PanicOnCException(unsafe.Pointer(C.Torch_Jit_Module_To(
	// 	module.Pointer,
	// 	(C.Device)(device.Pointer),
	// 	C.int8_t(dtype),
	// )))
	// return module
}

// TODO: func (module *JitModule) Copy() { }
// TODO: func (module *JitModule) DeepCopy() { }
// TODO: func (module *JitModule) Clone(inplace bool) { }

// Forward pass IValues through the module and return the resulting IValue.
func (module *JitModule) Forward(inputs []torch.IValue) torch.IValue {
	// Convert the torch IValues to C IValues.
	var ivalues []C.IValue
	for _, element := range inputs {
		ivalues = append(ivalues, (C.IValue)(*element.T))
	}
	// Call the forward method with a reference to the output.
	var output C.IValue
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Jit_Module_Forward(
		&output,
		module.Pointer,
		&ivalues[0],
		C.int64_t(len(ivalues)),
	)))
	return torch.IValue{(*unsafe.Pointer)(&output)}
}
