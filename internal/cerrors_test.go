// test cases for cerrors.go
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

package torch_internal_test

import (
    "testing"
    "github.com/stretchr/testify/assert"
    internal "github.com/Kautenja/gotorch/internal"
)

func Test_PanicOnCException_WithNilInput(t *testing.T) {
    assert.NotPanics(t, func() { internal.PanicOnCException(nil) })
}

func Test_PanicOnCException_WithMockedCException(t *testing.T) {
    message := "an error message"
    assert.PanicsWithValue(t, message, func() {
        internal.PanicOnCException(internal.MockCException(message))
    })
}

func Test_NewTorchError_WithNilInput(t *testing.T) {
    error := internal.NewTorchError(nil)
    assert.NotNil(t, error)
    assert.Equal(t, "", error.Error())
}

func Test_NewTorchError_WithMockedCException(t *testing.T) {
    message := "an error message"
    error := internal.NewTorchError(internal.MockCException(message))
    assert.NotNil(t, error)
    assert.Equal(t, message, error.Error())
}
