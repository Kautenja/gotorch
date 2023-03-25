// Go bindings for torch::Device.
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

package torch

// #cgo CPPFLAGS: -I/usr/local/include -I/usr/local/include/cgotorch
// #cgo LDFLAGS: -L/usr/local/lib -lc10 -ltorch_cpu -ltorch -lcgotorch
// #include <stdio.h>
// #include <stdlib.h>
// #include "cgotorch/cgotorch.h"
import "C"
import (
	"runtime"
	"unsafe"
	"github.com/Kautenja/gotorch/internal"
)

// Device wrapper a pointer to C.Device
type Device struct {
	Pointer C.Device
}

// Create a new Device.
func NewDevice(deviceName string) (device *Device) {
	device = &Device{}
	deviceNameCString := C.CString(deviceName)
	defer C.free(unsafe.Pointer(deviceNameCString))
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Device(&device.Pointer, deviceNameCString)))
	runtime.SetFinalizer(device, (*Device).free)
	return
}

// Free a device from memory.
func (device *Device) free() {
	if device.Pointer == nil {
		panic("Attempting to free a device that has already been freed!")
	}
	C.Torch_Device_Free(device.Pointer)
	device.Pointer = nil
}

// Return true if the given device is valid, false otherwise.
func IsDevice(deviceName string) bool {
	deviceNameCString := C.CString(deviceName)
	defer C.free(unsafe.Pointer(deviceNameCString))
	return bool(C.Torch_IsDevice(deviceNameCString))
}
