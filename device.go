// Go bindings for torch::Device.
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
    "runtime"
    "unsafe"
    internal "github.com/Kautenja/gotorch/internal"
)

// Device wrapper a pointer to C.Device
type Device struct {
    T C.Device
}

// NewDevice returns a Device
func NewDevice(deviceName string) Device {
    var device C.Device
    deviceNameCString := C.CString(deviceName)
    defer C.free(unsafe.Pointer(deviceNameCString))
    internal.PanicOnCException(unsafe.Pointer(C.Torch_Device(&device, deviceNameCString)))
    // Set the finalizer for the Go structure to free the heap-allocated C
    // memory when the garbage collector finalizes the object.
    runtime.SetFinalizer((*unsafe.Pointer)(&device), func(t *unsafe.Pointer) {
        C.Torch_Device_Free(C.Device(*t))
    })
    return Device{device}
}

/// Return true if the given device is valid, false otherwise.
func IsDevice(deviceName string) bool {
    deviceNameCString := C.CString(deviceName)
    defer C.free(unsafe.Pointer(deviceNameCString))
    return bool(C.Torch_IsDevice(deviceNameCString))
}
