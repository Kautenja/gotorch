// Error handling utilities for GoTorch.
//
// Copyright (c) 2022 Christian Kauten
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

package torch_internal

// #include <stdio.h>
// #include <stdlib.h>
import "C"
import (
    "strings"
    "unsafe"
    "errors"
)

// Panic if a system error is caught in the error buffer. err is a *C.Char to
// an error string that may be nil.
func PanicOnCException(err unsafe.Pointer) {
    if err != nil {
        defer C.free(err)  // TODO: it would be better to have the C-layer
        // perform the truncation to prevent unnecessary memory transfer.
        message := C.GoString((*C.char)(err))
        panic(strings.Split(message, "\n")[0])
    }
}

// Create a new error from a verified char*. `err` is the char* that is
// guaranteed to not be nil. `err` is expected to be managed by the Golang
// context. When the string is copied, it will be freed using CGo.
func NewTorchError(err unsafe.Pointer) error {
    defer C.free(err)  // TODO: it would be better to have the C-layer
    // perform the truncation to prevent unnecessary memory transfer.
    message := C.GoString((*C.char)(err))
    return errors.New(strings.Split(message, "\n")[0])
}

// Create a mock C error with a message. We intentionally do not defer the
// freeing of the C-string to mock the functionality of the C interface that
// will transfer ownership of non-nil pointer returns to the caller.
func MockCException(message string) unsafe.Pointer {
    return unsafe.Pointer(C.CString(message))
}
