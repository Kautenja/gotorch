// Go bindings for at::Tensor functions.
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
// #include "cgotorch/cgotorch.h"
import "C"
import (
	"unsafe"
	"runtime"
	"github.com/Kautenja/gotorch/internal"
)

// A representation of the return type for a paired value/index selection call.
type ValueIndexPair struct {
	Values, Indices *Tensor
}

// ---------------------------------------------------------------------------
// MARK: Tensor Metadata
// ---------------------------------------------------------------------------

// Return the number of elements in the tensor.
func Numel(tensor *Tensor) int64 {
	var output C.int64_t
	C.Torch_Tensor_Numel(&output, tensor.Pointer)
	runtime.KeepAlive(tensor)
	return int64(output)
}

// Return the number of elements in the tensor.
func (tensor *Tensor) Numel() int64 {
	return Numel(tensor)
}

// Return true if the tensor of a complex data-type.
func IsComplex(tensor *Tensor) bool {
	var output C.bool
	C.Torch_Tensor_Is_Complex(&output, tensor.Pointer)
	runtime.KeepAlive(tensor)
	return bool(output)
}

// Return true if the tensor of a complex data-type.
func (tensor *Tensor) IsComplex() bool {
	return IsComplex(tensor)
}

// Return true if the (complex) tensor is in conjugated form.
func IsConj(tensor *Tensor) bool {
	var output C.bool
	C.Torch_Tensor_Is_Conj(&output, tensor.Pointer)
	runtime.KeepAlive(tensor)
	return bool(output)
}

// Return true if the (complex) tensor is in conjugated form.
func (tensor *Tensor) IsConj() bool {
	return IsConj(tensor)
}

// Return true if the tensor of a floating-point data-type.
func IsFloatingPoint(tensor *Tensor) bool {
	var output C.bool
	C.Torch_Tensor_Is_Floating_Point(&output, tensor.Pointer)
	runtime.KeepAlive(tensor)
	return bool(output)
}

// Return true if the tensor of a floating-point data-type.
func (tensor *Tensor) IsFloatingPoint() bool {
	return IsFloatingPoint(tensor)
}

// Return true if the tensor is a single non-zero element (i.e., scalar.)
func IsNonzero(tensor *Tensor) bool {
	var output C.bool
	C.Torch_Tensor_Is_Nonzero(&output, tensor.Pointer)
	runtime.KeepAlive(tensor)
	return bool(output)
}

// Return true if the tensor is a single non-zero element (i.e., scalar.)
func (tensor *Tensor) IsNonzero() bool {
	return IsNonzero(tensor)
}

// ---------------------------------------------------------------------------
// MARK: Tensor Creation
// ---------------------------------------------------------------------------

// Create a new tensor of given size filled with zeros.
func Zeros(size []int64, options *TensorOptions) *Tensor {
	if len(size) == 0 { panic("size is empty") }
	tensor := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Zeros(
		&tensor.Pointer,
		(*C.int64_t)(unsafe.Pointer(&size[0])),
		C.int64_t(len(size)),
		options.Pointer,
	)))
	runtime.KeepAlive(options)
	runtime.SetFinalizer(tensor, (*Tensor).free)
	return tensor
}

// Create a tensor filled with zeros in the shape of a reference.
func ZerosLike(reference *Tensor) *Tensor {
	if reference.Pointer == nil { panic("input tensor is nil") }
	tensor := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_ZerosLike(
		&tensor.Pointer,
		reference.Pointer,
	)))
	runtime.KeepAlive(reference)
	runtime.SetFinalizer(tensor, (*Tensor).free)
	return tensor
}

// Create a new tensor of given size filled with ones.
func Ones(size []int64, options *TensorOptions) *Tensor {
	if len(size) == 0 { panic("size is empty") }
	tensor := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Ones(
		&tensor.Pointer,
		(*C.int64_t)(unsafe.Pointer(&size[0])),
		C.int64_t(len(size)),
		options.Pointer,
	)))
	runtime.KeepAlive(options)
	runtime.SetFinalizer(tensor, (*Tensor).free)
	return tensor
}

// Create a tensor filled with ones in the shape of a reference.
func OnesLike(reference *Tensor) *Tensor {
	if reference.Pointer == nil { panic("input tensor is nil") }
	tensor := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_OnesLike(
		&tensor.Pointer,
		reference.Pointer,
	)))
	runtime.KeepAlive(reference)
	runtime.SetFinalizer(tensor, (*Tensor).free)
	return tensor
}

// Create an inclusive range tensor from begin position to end position along
// integer step size.
func Arange(begin, end, step float32, options *TensorOptions) *Tensor {
	tensor := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Arange(
		&tensor.Pointer,
		C.float(begin),
		C.float(end),
		C.float(step),
		options.Pointer,
	)))
	runtime.KeepAlive(options)
	runtime.SetFinalizer(tensor, (*Tensor).free)
	return tensor
}

// Create an exclusive range tensor from begin position to end position along
// integer step size.
func Range(begin, end, step float32, options *TensorOptions) *Tensor {
	tensor := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Range(
		&tensor.Pointer,
		C.float(begin),
		C.float(end),
		C.float(step),
		options.Pointer,
	)))
	runtime.KeepAlive(options)
	runtime.SetFinalizer(tensor, (*Tensor).free)
	return tensor
}

// Create an linear space tensor from begin to end with given number of total
// steps.
func Linspace(begin, end float32, steps int64, options *TensorOptions) *Tensor {
	tensor := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Linspace(
		&tensor.Pointer,
		C.float(begin),
		C.float(end),
		C.int64_t(steps),
		options.Pointer,
	)))
	runtime.KeepAlive(options)
	runtime.SetFinalizer(tensor, (*Tensor).free)
	return tensor
}

// Create an logarithmic space tensor from begin to end with given number of
// total steps.
func Logspace(begin, end float32, steps int64, base float64, options *TensorOptions) *Tensor {
	tensor := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Logspace(
		&tensor.Pointer,
		C.float(begin),
		C.float(end),
		C.int64_t(steps),
		C.double(base),
		options.Pointer,
	)))
	runtime.KeepAlive(options)
	runtime.SetFinalizer(tensor, (*Tensor).free)
	return tensor
}

// Create an NxM identity matrix.
func Eye(n, m int64, options *TensorOptions) *Tensor {
	tensor := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Eye(
		&tensor.Pointer,
		C.int64_t(n),
		C.int64_t(m),
		options.Pointer,
	)))
	runtime.KeepAlive(options)
	runtime.SetFinalizer(tensor, (*Tensor).free)
	return tensor
}

// Create a new tensor of given size filled with empty values.
func Empty(size []int64, options *TensorOptions) *Tensor {
	if len(size) == 0 { panic("size is empty") }
	tensor := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Empty(
		&tensor.Pointer,
		(*C.int64_t)(unsafe.Pointer(&size[0])),
		C.int64_t(len(size)),
		options.Pointer,
	)))
	runtime.KeepAlive(options)
	runtime.SetFinalizer(tensor, (*Tensor).free)
	return tensor
}

// Create a tensor filled with empty values in the shape of a reference.
func EmptyLike(reference *Tensor) *Tensor {
	if reference.Pointer == nil { panic("input tensor is nil") }
	tensor := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_EmptyLike(
		&tensor.Pointer,
		reference.Pointer,
	)))
	runtime.KeepAlive(reference)
	runtime.SetFinalizer(tensor, (*Tensor).free)
	return tensor
}

// Create a new tensor of given size filled with static values.
func Full(size []int64, value float32, options *TensorOptions) *Tensor {
	if len(size) == 0 { panic("size is empty") }
	tensor := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Full(
		&tensor.Pointer,
		(*C.int64_t)(unsafe.Pointer(&size[0])),
		C.int64_t(len(size)), C.float(value),
		options.Pointer,
	)))
	runtime.KeepAlive(options)
	runtime.SetFinalizer(tensor, (*Tensor).free)
	return tensor
}

// Create a tensor filled with static values in the shape of a reference.
func FullLike(reference *Tensor, value float32) *Tensor {
	if reference.Pointer == nil { panic("input tensor is nil") }
	tensor := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_FullLike(
		&tensor.Pointer,
		reference.Pointer,
		C.float(value),
	)))
	runtime.KeepAlive(reference)
	runtime.SetFinalizer(tensor, (*Tensor).free)
	return tensor
}

// Create a new tensor of given size filled with uniform random values.
func Rand(size []int64, options *TensorOptions) *Tensor {
	if len(size) == 0 { panic("size is empty") }
	tensor := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Rand(
		&tensor.Pointer,
		(*C.int64_t)(unsafe.Pointer(&size[0])),
		C.int64_t(len(size)),
		options.Pointer,
	)))
	runtime.KeepAlive(options)
	runtime.SetFinalizer(tensor, (*Tensor).free)
	return tensor
}

// Create a tensor filled with uniform random values in the shape of a reference.
func RandLike(reference *Tensor) *Tensor {
	if reference.Pointer == nil { panic("input tensor is nil") }
	tensor := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_RandLike(
		&tensor.Pointer, reference.Pointer,
	)))
	runtime.KeepAlive(reference)
	runtime.SetFinalizer(tensor, (*Tensor).free)
	return tensor
}

// Create a new tensor of given size filled with random integers in [low, high).
func RandInt(size []int64, low int64, high int64, options *TensorOptions) *Tensor {
	if len(size) == 0 { panic("size is empty") }
	tensor := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_RandInt(
		&tensor.Pointer,
		(*C.int64_t)(unsafe.Pointer(&size[0])),
		C.int64_t(len(size)),
		C.int64_t(low),
		C.int64_t(high),
		options.Pointer,
	)))
	runtime.KeepAlive(options)
	runtime.SetFinalizer(tensor, (*Tensor).free)
	return tensor
}

// Create a tensor filled with random integers in [low, high) in the shape of
// a reference.
func RandIntLike(reference *Tensor, low int64, high int64) *Tensor {
	if reference.Pointer == nil { panic("input tensor is nil") }
	tensor := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_RandIntLike(
		&tensor.Pointer,
		reference.Pointer,
		C.int64_t(low),
		C.int64_t(high),
	)))
	runtime.KeepAlive(reference)
	runtime.SetFinalizer(tensor, (*Tensor).free)
	return tensor
}

// Create a new tensor of given size filled with Gaussian random values.
func RandN(size []int64, options *TensorOptions) *Tensor {
	if len(size) == 0 { panic("size is empty") }
	tensor := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_RandN(
		&tensor.Pointer,
		(*C.int64_t)(unsafe.Pointer(&size[0])),
		C.int64_t(len(size)),
		options.Pointer,
	)))
	runtime.KeepAlive(options)
	runtime.SetFinalizer(tensor, (*Tensor).free)
	return tensor
}

// Create a tensor filled with Gaussian random values in the shape of a
// reference.
func RandNLike(reference *Tensor) *Tensor {
	if reference.Pointer == nil { panic("input tensor is nil") }
	tensor := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_RandNLike(
		&tensor.Pointer,
		reference.Pointer,
	)))
	runtime.KeepAlive(reference)
	runtime.SetFinalizer(tensor, (*Tensor).free)
	return tensor
}

// ---------------------------------------------------------------------------
// MARK: ToSlice
// ---------------------------------------------------------------------------

// Convert a torch Tensor to a Go slice. This function implies a flattening
// of the tensor to return 1-dimensional vectors.
func ToSlice(tensor *Tensor) interface{} {
	dtype := tensor.Dtype()
	tensor = tensor.Flatten(0, -1)
	length := tensor.Shape()[0]
	switch dtype {
	case Byte:
		output := make([]uint8, length)
		blob := TensorFromBlob(unsafe.Pointer(&output[0]), Byte, []int64{length})
		blob.Copy_(tensor)
		return output
	case Char:
		output := make([]int8, length)
		blob := TensorFromBlob(unsafe.Pointer(&output[0]), Char, []int64{length})
		blob.Copy_(tensor)
		return output
	case Short:
		output := make([]int16, length)
		blob := TensorFromBlob(unsafe.Pointer(&output[0]), Short, []int64{length})
		blob.Copy_(tensor)
		return output
	case Int:
		output := make([]int32, length)
		blob := TensorFromBlob(unsafe.Pointer(&output[0]), Int, []int64{length})
		blob.Copy_(tensor)
		return output
	case Long:
		output := make([]int64, length)
		blob := TensorFromBlob(unsafe.Pointer(&output[0]), Long, []int64{length})
		blob.Copy_(tensor)
		return output
	case Half:  // Not natively supported, use uint16 container
		output := make([]uint16, length)
		blob := TensorFromBlob(unsafe.Pointer(&output[0]), Half, []int64{length})
		blob.Copy_(tensor)
		return output
	case Float:
		output := make([]float32, length)
		blob := TensorFromBlob(unsafe.Pointer(&output[0]), Float, []int64{length})
		blob.Copy_(tensor)
		return output
	case Double:
		output := make([]float64, length)
		blob := TensorFromBlob(unsafe.Pointer(&output[0]), Double, []int64{length})
		blob.Copy_(tensor)
		return output
	// case ComplexHalf:  // Not natively supported, use uint32 container
	//     output := make([]uint32, length)
	//     blob := TensorFromBlob(unsafe.Pointer(&output[0]), ComplexHalf, []int64{length})
	//     blob.Copy_(tensor)
	//     return output
	case ComplexFloat:
		output := make([]complex64, length)
		blob := TensorFromBlob(unsafe.Pointer(&output[0]), ComplexFloat, []int64{length})
		blob.Copy_(tensor)
		return output
	case ComplexDouble:
		output := make([]complex128, length)
		blob := TensorFromBlob(unsafe.Pointer(&output[0]), ComplexDouble, []int64{length})
		blob.Copy_(tensor)
		return output
	case Bool:
		output := make([]bool, length)
		blob := TensorFromBlob(unsafe.Pointer(&output[0]), Bool, []int64{length})
		blob.Copy_(tensor)
		return output
	// case QInt8: TODO
	// case QUInt8: TODO
	// case QInt32: TODO
	// case BFloat16: TODO
	default:
		panic("ToSlice is not supported for dtype")
	}
}

// Convert the Tensor to a Go slice. This function implies a flattening of the
// tensor to return a 1-dimensional vector.
func (tensor *Tensor) ToSlice() interface{} {
	return ToSlice(tensor)
}

// ---------------------------------------------------------------------------
// MARK: Maths
// ---------------------------------------------------------------------------

// Adds other, scaled by alpha, to input.
func Add(tensor, other *Tensor, alpha float32) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Add(
		tensor.Pointer,
		other.Pointer,
		C.float(alpha),
		&output.Pointer,
	)))
	runtime.KeepAlive(tensor)
	runtime.KeepAlive(other)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Adds other, scaled by alpha, to input.
func (tensor *Tensor) Add(other *Tensor, alpha float32) *Tensor {
	return Add(tensor, other, alpha)
}

// In-place version of Add().
func (tensor *Tensor) Add_(other *Tensor, alpha float32) *Tensor {
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Add_(
		tensor.Pointer,
		other.Pointer,
		C.float(alpha),
	)))
	runtime.KeepAlive(tensor)
	runtime.KeepAlive(other)
	return tensor
}

// TODO: func AddScalar(tensor *Tensor, scalar interface{}) *Tensor { }
// TODO: func (tensor *Tensor) AddScalar(scalar interface{}) *Tensor { }

// Subtracts other, scaled by alpha, from input.
func Sub(tensor, other *Tensor, alpha float32) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Sub(
		tensor.Pointer,
		other.Pointer,
		C.float(alpha),
		&output.Pointer,
	)))
	runtime.KeepAlive(tensor)
	runtime.KeepAlive(other)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Subtracts other, scaled by alpha, from input.
func (tensor *Tensor) Sub(other *Tensor, alpha float32) *Tensor {
	return Sub(tensor, other, alpha)
}

// In-place version of Sub().
func (tensor *Tensor) Sub_(other *Tensor, alpha float32) *Tensor {
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Sub_(
		tensor.Pointer,
		other.Pointer,
		C.float(alpha),
	)))
	runtime.KeepAlive(tensor)
	runtime.KeepAlive(other)
	return tensor
}

// TODO: func SubScalar(tensor *Tensor, scalar interface{}) *Tensor { }
// TODO: func (tensor *Tensor) SubScalar(scalar interface{}) *Tensor { }

// Multiplies input by other.
func Mul(tensor, other *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Mul(
		tensor.Pointer,
		other.Pointer,
		&output.Pointer,
	)))
	runtime.KeepAlive(tensor)
	runtime.KeepAlive(other)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Multiplies input by other.
func (tensor *Tensor) Mul(other *Tensor) *Tensor {
	return Mul(tensor, other)
}

// In-place version of Mul().
func (tensor *Tensor) Mul_(other *Tensor) *Tensor {
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Mul_(
		tensor.Pointer,
		other.Pointer,
	)))
	runtime.KeepAlive(tensor)
	runtime.KeepAlive(other)
	return tensor
}

// TODO: func MulScalar(tensor *Tensor, scalar interface{}) *Tensor { }
// TODO: func (tensor *Tensor) MulScalar(scalar interface{}) *Tensor { }

// Divides input by other.
func Div(tensor, other *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Div(
		tensor.Pointer,
		other.Pointer,
		&output.Pointer,
	)))
	runtime.KeepAlive(tensor)
	runtime.KeepAlive(other)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Divides input by other.
func (tensor *Tensor) Div(other *Tensor) *Tensor {
	return Div(tensor, other)
}

// In-place version of Div().
func (tensor *Tensor) Div_(other *Tensor) *Tensor {
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Div_(
		tensor.Pointer,
		other.Pointer,
	)))
	runtime.KeepAlive(tensor)
	runtime.KeepAlive(other)
	return tensor
}

// TODO: func DivScalar(tensor *Tensor, scalar interface{}) *Tensor { }
// TODO: func (tensor *Tensor) DivScalar(scalar interface{}) *Tensor { }

// Take the absolute value of input.
func Abs(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Abs(
		tensor.Pointer,
		&output.Pointer,
	)))
	runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Take the absolute value of input.
func (tensor *Tensor) Abs() *Tensor {
	return Abs(tensor)
}

// In-place version of Abs().
func Abs_(tensor *Tensor) *Tensor {
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Abs_(tensor.Pointer)))
	runtime.KeepAlive(tensor)
	return tensor
}

// In-place version of Abs().
func (tensor *Tensor) Abs_() *Tensor {
	return Abs_(tensor)
}

// Take the square of input.
func Square(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Square(
		tensor.Pointer,
		&output.Pointer,
	)))
	runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Take the square of input.
func (tensor *Tensor) Square() *Tensor {
	return Square(tensor)
}

// In-place version of Square().
func Square_(tensor *Tensor) *Tensor {
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Square_(tensor.Pointer)))
	runtime.KeepAlive(tensor)
	return tensor
}

// In-place version of Square().
func (tensor *Tensor) Square_() *Tensor {
	return Square_(tensor)
}

// Take the square-root of input.
func Sqrt(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Sqrt(
		tensor.Pointer,
		&output.Pointer,
	)))
	runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Take the square-root of input.
func (tensor *Tensor) Sqrt() *Tensor {
	return Sqrt(tensor)
}

// In-place version of Sqrt().
func Sqrt_(tensor *Tensor) *Tensor {
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Sqrt_(tensor.Pointer)))
	runtime.KeepAlive(tensor)
	return tensor
}

// In-place version of Sqrt().
func (tensor *Tensor) Sqrt_() *Tensor {
	return Sqrt_(tensor)
}

// Take the power of each element in input with exponent and returns a tensor with the result.
func Pow(tensor *Tensor, exponent float64) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Pow(
		tensor.Pointer,
		C.double(exponent),
		&output.Pointer,
	)))
	runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Take the power of each element in input with exponent and returns a tensor with the result.
func (tensor *Tensor) Pow(exponent float64) *Tensor {
	return Pow(tensor, exponent)
}

// Return a new tensor with the hyperbolic tangent of the elements of input.
func Tanh(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Tanh(
		tensor.Pointer,
		&output.Pointer,
	)))
	runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Return a new tensor with the hyperbolic tangent of the elements of input.
func (tensor *Tensor) Tanh() *Tensor {
	return Tanh(tensor)
}

// Computes the expit (also known as the logistic sigmoid function) of the
// elements of input.
func Sigmoid(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Sigmoid(
		tensor.Pointer,
		&output.Pointer,
	)))
	runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Computes the expit (also known as the logistic sigmoid function) of the
// elements of input.
func (tensor *Tensor) Sigmoid() *Tensor {
	return Sigmoid(tensor)
}

// Apply a softmax followed by a logarithm.
//
// While mathematically equivalent to log(softmax(x)), doing these two
// operations separately is slower and numerically unstable. This function uses
// an alternative formulation to compute the output and gradient correctly.
func LogSoftmax(tensor *Tensor, dim int64) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_LogSoftmax(
		tensor.Pointer,
		C.int64_t(dim),
		&output.Pointer,
	)))
	runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Apply a softmax followed by a logarithm.
//
// While mathematically equivalent to log(softmax(x)), doing these two
// operations separately is slower and numerically unstable. This function uses
// an alternative formulation to compute the output and gradient correctly.
func (tensor *Tensor) LogSoftmax(dim int64) *Tensor {
	return LogSoftmax(tensor, dim)
}

// Compute the element-wise logical AND of the given input tensors. Zeros are
// treated as False and non-zeros are treated as True.
func LogicalAnd(tensor, other *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_LogicalAnd(
		&output.Pointer,
		tensor.Pointer,
		other.Pointer,
	)))
	runtime.KeepAlive(tensor)
	runtime.KeepAlive(other)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Compute the element-wise logical AND of the given input tensors. Zeros are
// treated as False and non-zeros are treated as True.
func (tensor *Tensor) LogicalAnd(other *Tensor) *Tensor {
	return LogicalAnd(tensor, other)
}

// Computes the element-wise logical NOT of the given input tensor. If the
// input tensor is not a bool tensor, zeros are treated as False and non-zeros
// are treated as True.
func LogicalNot(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_LogicalNot(
		&output.Pointer,
		tensor.Pointer,
	)))
	runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Computes the element-wise logical NOT of the given input tensor. If the
// input tensor is not a bool tensor, zeros are treated as False and non-zeros
// are treated as True.
func (tensor *Tensor) LogicalNot() *Tensor {
	return LogicalNot(tensor)
}

// Compute the element-wise logical OR of the given input tensors. Zeros are
// treated as False and non-zeros are treated as True.
func LogicalOr(tensor, other *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_LogicalOr(
		&output.Pointer,
		tensor.Pointer,
		other.Pointer,
	)))
	runtime.KeepAlive(tensor)
	runtime.KeepAlive(other)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Compute the element-wise logical OR of the given input tensors. Zeros are
// treated as False and non-zeros are treated as True.
func (tensor *Tensor) LogicalOr(other *Tensor) *Tensor {
	return LogicalOr(tensor, other)
}

// Compute the element-wise logical XOR of the given input tensors. Zeros are
// treated as False and non-zeros are treated as True.
func LogicalXor(tensor, other *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_LogicalXor(
		&output.Pointer,
		tensor.Pointer,
		other.Pointer,
	)))
	runtime.KeepAlive(tensor)
	runtime.KeepAlive(other)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Compute the element-wise logical XOR of the given input tensors. Zeros are
// treated as False and non-zeros are treated as True.
func (tensor *Tensor) LogicalXor(other *Tensor) *Tensor {
	return LogicalXor(tensor, other)
}

// Clamps all elements in input into the range [min, max].
func Clamp(tensor, minimum, maximum *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Clamp(
		&output.Pointer,
		tensor.Pointer,
		minimum.Pointer,
		maximum.Pointer,
	)))
	runtime.KeepAlive(tensor)
    runtime.KeepAlive(minimum)
    runtime.KeepAlive(maximum)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Clamps all elements in input into the range [min, max].
func (tensor *Tensor) Clamp(minimum, maximum *Tensor) *Tensor {
	return Clamp(tensor, minimum, maximum)
}

// In-place version of torch.Clamp()
func Clamp_(tensor, minimum, maximum *Tensor) *Tensor {
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Clamp_(
		tensor.Pointer,
		minimum.Pointer,
		maximum.Pointer,
	)))
	runtime.KeepAlive(tensor)
	runtime.KeepAlive(minimum)
	runtime.KeepAlive(maximum)
	return tensor
}

// In-place version of torch.Clamp()
func (tensor *Tensor) Clamp_(minimum, maximum *Tensor) *Tensor {
	return Clamp_(tensor, minimum, maximum)
}

// Clamps all elements in input to have maximum value of input tensor.
func ClampMax(tensor, maximum *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_ClampMax(
		&output.Pointer,
		tensor.Pointer,
		maximum.Pointer,
	)))
    runtime.KeepAlive(tensor)
    runtime.KeepAlive(maximum)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Clamps all elements in input to have maximum value of input tensor.
func (tensor *Tensor) ClampMax(maximum *Tensor) *Tensor {
	return ClampMax(tensor, maximum)
}

// In-place version of torch.ClampMax()
func ClampMax_(tensor, maximum *Tensor) *Tensor {
	internal.PanicOnCException(unsafe.Pointer(C.Torch_ClampMax_(
		tensor.Pointer,
		maximum.Pointer,
	)))
    runtime.KeepAlive(tensor)
    runtime.KeepAlive(maximum)
	return tensor
}

// In-place version of torch.ClampMax()
func (tensor *Tensor) ClampMax_(maximum *Tensor) *Tensor {
	return ClampMax_(tensor, maximum)
}

// Clamps all elements in input to have minimum value of input tensor.
func ClampMin(tensor, minimum *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_ClampMin(
		&output.Pointer,
		tensor.Pointer,
		minimum.Pointer,
	)))
    runtime.KeepAlive(tensor)
    runtime.KeepAlive(minimum)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Clamps all elements in input to have minimum value of input tensor.
func (tensor *Tensor) ClampMin(minimum *Tensor) *Tensor {
	return ClampMin(tensor, minimum)
}

// In-place version of torch.ClampMin()
func ClampMin_(tensor, minimum *Tensor) *Tensor {
	internal.PanicOnCException(unsafe.Pointer(C.Torch_ClampMin_(
		tensor.Pointer,
		minimum.Pointer,
	)))
    runtime.KeepAlive(tensor)
    runtime.KeepAlive(minimum)
	return tensor
}

// In-place version of torch.ClampMin()
func (tensor *Tensor) ClampMin_(minimum *Tensor) *Tensor {
	return ClampMin_(tensor, minimum)
}

// ---------------------------------------------------------------------------
// MARK: Data layout
// ---------------------------------------------------------------------------

// Return a view of the original tensor input with its dimensions permuted.
func Permute(tensor *Tensor, dims ...int64) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Permute(
		tensor.Pointer,
		(*C.int64_t)(&dims[0]),
		C.int64_t(len(dims)),
		&output.Pointer,
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Return a view of the original tensor input with its dimensions permuted.
func (tensor *Tensor) Permute(dims ...int64) *Tensor {
	return Permute(tensor, dims...)
}

// Return a tensor that is a transposed version of input. The given dimensions
// dim0 and dim1 are swapped.
//
// If input is a strided tensor then the resulting out tensor shares its
// underlying storage with the input tensor, so changing the content of one
// would change the content of the other.
//
// If input is a sparse tensor then the resulting out tensor does not share the
// underlying storage with the input tensor.
//
// If input is a sparse tensor with compressed layout (SparseCSR, SparseBSR,
// SparseCSC or SparseBSC) the arguments dim0 and dim1 must be both batch
// dimensions, or must both be sparse dimensions. The batch dimensions of a
// sparse tensor are the dimensions preceding the sparse dimensions.
func Transpose(tensor *Tensor, dim0, dim1 int64) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Transpose(
		tensor.Pointer,
		C.int64_t(dim0),
		C.int64_t(dim1),
		&output.Pointer,
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Return a tensor that is a transposed version of input. The given dimensions
// dim0 and dim1 are swapped.
//
// If input is a strided tensor then the resulting out tensor shares its
// underlying storage with the input tensor, so changing the content of one
// would change the content of the other.
//
// If input is a sparse tensor then the resulting out tensor does not share the
// underlying storage with the input tensor.
//
// If input is a sparse tensor with compressed layout (SparseCSR, SparseBSR,
// SparseCSC or SparseBSC) the arguments dim0 and dim1 must be both batch
// dimensions, or must both be sparse dimensions. The batch dimensions of a
// sparse tensor are the dimensions preceding the sparse dimensions.
func (tensor *Tensor) Transpose(dim0, dim1 int64) *Tensor {
	return Transpose(tensor, dim0, dim1)
}

// Flattens input dimensions by reshaping them into a one-dimensional tensor.
// Only dimensions starting with start_dim and ending with end_dim are
// flattened. The order of elements in input is unchanged.
//
// Unlike NumPy’s flatten, which always copies input’s data, this function may
// return the original object, a view, or copy. If no dimensions are flattened,
// then the original object input is returned. Otherwise, if input can be
// viewed as the flattened shape, then that view is returned. Finally, only if
// the input cannot be viewed as the flattened shape is input’s data copied.
// See (torch.Tensor).View() for details on when a view will be returned.
func Flatten(tensor *Tensor, startDim, endDim int64) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Flatten(
		tensor.Pointer,
		C.int64_t(startDim),
		C.int64_t(endDim),
		&output.Pointer,
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Flattens input dimensions by reshaping them into a one-dimensional tensor.
// Only dimensions starting with start_dim and ending with end_dim are
// flattened. The order of elements in input is unchanged.
//
// Unlike NumPy’s flatten, which always copies input’s data, this function may
// return the original object, a view, or copy. If no dimensions are flattened,
// then the original object input is returned. Otherwise, if input can be
// viewed as the flattened shape, then that view is returned. Finally, only if
// the input cannot be viewed as the flattened shape is input’s data copied.
// See (torch.Tensor).View() for details on when a view will be returned.
func (tensor *Tensor) Flatten(startDim, endDim int64) *Tensor {
	return Flatten(tensor, startDim, endDim)
}

// Return a tensor with all the dimensions of input of size 1 removed. For
// example, if input is of shape: (A×1×B×C×1×D) then the out tensor will be of
// shape: (A×B×C×D). When dim is given, a squeeze operation is done only in
// the given dimension. If input is of shape: (A×1×B), squeeze(input, 0) leaves
// the tensor unchanged, but squeeze(input, 1) will squeeze the tensor to the
// shape (A×B).
func Squeeze(tensor *Tensor, dim ...int64) *Tensor {
	output := &Tensor{}
	switch len(dim) {
	case 0:
		internal.PanicOnCException(unsafe.Pointer(C.Torch_Squeeze(
			tensor.Pointer,
			&output.Pointer,
		)))
	case 1:
		internal.PanicOnCException(unsafe.Pointer(C.Torch_SqueezeWithDim(
			tensor.Pointer,
			C.int64_t(dim[0]),
			&output.Pointer,
		)))
	default:
		panic("Squeeze only accepts 0-1 dim as input")
	}
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Return a tensor with all the dimensions of input of size 1 removed. For
// example, if input is of shape: (A×1×B×C×1×D) then the out tensor will be of
// shape: (A×B×C×D). When dim is given, a squeeze operation is done only in
// the given dimension. If input is of shape: (A×1×B), squeeze(input, 0) leaves
// the tensor unchanged, but squeeze(input, 1) will squeeze the tensor to the
// shape (A×B).
func (tensor *Tensor) Squeeze(dim ...int64) *Tensor {
	return Squeeze(tensor, dim...)
}

// Return a new tensor with a dimension of size one inserted at the specified
// position. The returned tensor shares the same underlying data with this
// tensor. A dim value within the range [-input.dim() - 1, input.dim() + 1) can
// be used. Negative dim will correspond to unsqueeze() applied at
// dim = dim + input.dim() + 1.
func Unsqueeze(tensor *Tensor, dim int64) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Unsqueeze(
		&output.Pointer,
		tensor.Pointer,
		C.int64_t(dim),
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Return a new tensor with a dimension of size one inserted at the specified
// position. The returned tensor shares the same underlying data with this
// tensor. A dim value within the range [-input.dim() - 1, input.dim() + 1) can
// be used. Negative dim will correspond to unsqueeze() applied at
// dim = dim + input.dim() + 1.
func (tensor *Tensor) Unsqueeze(dim int64) *Tensor {
	return Unsqueeze(tensor, dim)
}

// Concatenate a sequence of tensors along a new dimension. All tensors need to
// be of the same size.
func Stack(tensors []*Tensor, dim int64) *Tensor {
	pointers := []C.Tensor{}
	for _, tensor := range tensors {
		pointers = append(pointers, tensor.Pointer)
	}
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Stack(
		&output.Pointer,
		&pointers[0],
		C.int64_t(len(pointers)),
		C.int64_t(dim),
	)))
    runtime.KeepAlive(tensors)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Concatenate the given sequence of tensors in the given dimension. All
// tensors must either have the same shape (except in the concatenating
// dimension) or be empty.
func Cat(tensors []*Tensor, dim int64) *Tensor {
	pointers := []C.Tensor{}
	for _, tensor := range tensors {
		pointers = append(pointers, tensor.Pointer)
	}
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Cat(
		&output.Pointer,
		&pointers[0],
		C.int64_t(len(pointers)),
		C.int64_t(dim),
	)))
    runtime.KeepAlive(tensors)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Alias of torch.Cat.
func Concat(tensors []*Tensor, dim int64) *Tensor {
	return Cat(tensors, dim)
}

// Alias of torch.Cat.
func Concatenate(tensors []*Tensor, dim int64) *Tensor {
	return Cat(tensors, dim)
}

// ---------------------------------------------------------------------------
// MARK: Selection
// ---------------------------------------------------------------------------

// Perform NumPy-like tensor slicing where dim is the dimension to slice along,
// start and stop determine the [start, stop) bounds of the index, and step
// describes the spacing between elements in the slice.
func Slice(tensor *Tensor, dim, start, stop, step int64) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Slice(
		&output.Pointer,
		tensor.Pointer,
		C.int64_t(dim),
		C.int64_t(start),
		C.int64_t(stop),
		C.int64_t(step),
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Perform NumPy-like tensor slicing where dim is the dimension to slice along,
// start and stop determine the [start, stop) bounds of the index, and step
// describes the spacing between elements in the slice.
func (tensor *Tensor) Slice(dim, start, stop, step int64) *Tensor {
	return Slice(tensor, dim, start, stop, step)
}

// Return a new tensor which indexes the input tensor along dimension dim using
// the entries in index which is a LongTensor. The returned tensor has the same
// number of dimensions as the original tensor (input). The dimth dimension has
// the same size as the length of index; other dimensions have the same size as
// in the original tensor.
func IndexSelect(tensor *Tensor, dim int64, index *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IndexSelect(
		tensor.Pointer,
		C.int64_t(dim),
		index.Pointer,
		&output.Pointer,
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Return a new tensor which indexes the input tensor along dimension dim using
// the entries in index which is a LongTensor. The returned tensor has the same
// number of dimensions as the original tensor (input). The dimth dimension has
// the same size as the length of index; other dimensions have the same size as
// in the original tensor.
func (tensor *Tensor) IndexSelect(dim int64, index *Tensor) *Tensor {
	return IndexSelect(tensor, dim, index)
}

// ---------------------------------------------------------------------------
// MARK: Reduction Ops
// ---------------------------------------------------------------------------

// Reduce a tensor to its minimum index.
func Argmin(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Argmin(&output.Pointer, tensor.Pointer)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Reduce a tensor to its minimum index.
func (tensor *Tensor) Argmin() *Tensor {
	return Argmin(tensor)
}

// Reduce a tensor to its minimum index along the given dimension.
func ArgminByDim(tensor *Tensor, dim int, keep_dims bool) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_ArgminByDim(
		&output.Pointer,
		tensor.Pointer,
		C.int64_t(dim),
		C.bool(keep_dims),
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Reduce a tensor to its minimum index along the given dimension.
func (tensor *Tensor) ArgminByDim(dim int, keep_dims bool) *Tensor {
	return ArgminByDim(tensor, dim, keep_dims)
}

// Reduce a tensor to its maximum index.
func Argmax(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Argmax(&output.Pointer, tensor.Pointer)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Reduce a tensor to its maximum index.
func (tensor *Tensor) Argmax() *Tensor {
	return Argmax(tensor)
}

// Reduce a tensor to its maximum index along the given dimension.
func ArgmaxByDim(tensor *Tensor, dim int, keep_dims bool) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_ArgmaxByDim(
		&output.Pointer,
		tensor.Pointer,
		C.int64_t(dim),
		C.bool(keep_dims),
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Reduce a tensor to its maximum index along the given dimension.
func (tensor *Tensor) ArgmaxByDim(dim int, keep_dims bool) *Tensor {
	return ArgmaxByDim(tensor, dim, keep_dims)
}

// TODO: amax
// TODO: amin
// TODO: aminmax

// Check if all values in the tensor evaluate to true.
func All(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_All(&output.Pointer, tensor.Pointer)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Check if all values in the tensor evaluate to true.
func (tensor *Tensor) All() *Tensor {
	return All(tensor)
}

// Check if all values in the tensor evaluate to true along the given dimension.
func AllByDim(tensor *Tensor, dim int, keep_dims bool) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_AllByDim(
		&output.Pointer,
		tensor.Pointer,
		C.int64_t(dim),
		C.bool(keep_dims),
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Check if all values in the tensor evaluate to true along the given dimension.
func (tensor *Tensor) AllByDim(dim int, keep_dims bool) *Tensor {
	return AllByDim(tensor, dim, keep_dims)
}

// Check if any values in the tensor evaluate to true.
func Any(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Any(&output.Pointer, tensor.Pointer)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Check if any values in the tensor evaluate to true.
func (tensor *Tensor) Any() *Tensor {
	return Any(tensor)
}

// Check if any values in the tensor evaluate to true along the given dimension.
func AnyByDim(tensor *Tensor, dim int, keep_dims bool) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_AnyByDim(
		&output.Pointer,
		tensor.Pointer,
		C.int64_t(dim),
		C.bool(keep_dims),
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Check if any values in the tensor evaluate to true along the given dimension.
func (tensor *Tensor) AnyByDim(dim int, keep_dims bool) *Tensor {
	return AnyByDim(tensor, dim, keep_dims)
}

// TODO: dist
// TODO: logsumexp

// Reduce a tensor to its maximum value.
func Max(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Max(&output.Pointer, tensor.Pointer)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Reduce a tensor to its maximum value.
func (tensor *Tensor) Max() *Tensor {
	return Max(tensor)
}

// Reduce a tensor to its maximum value along the given dimension.
func MaxByDim(tensor *Tensor, dim int, keep_dims bool) ValueIndexPair {
	values := &Tensor{}
	indices := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_MaxByDim(
		&values.Pointer,
		&indices.Pointer,
		tensor.Pointer,
		C.int64_t(dim),
		C.bool(keep_dims),
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(values, (*Tensor).free)
	runtime.SetFinalizer(indices, (*Tensor).free)
	return ValueIndexPair{values, indices}
}

// Reduce a tensor to its maximum value along the given dimension.
func (tensor *Tensor) MaxByDim(dim int, keep_dims bool) ValueIndexPair {
	return MaxByDim(tensor, dim, keep_dims)
}

// Reduce a tensor to its minimum value.
func Min(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Min(&output.Pointer, tensor.Pointer)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Reduce a tensor to its minimum value.
func (tensor *Tensor) Min() *Tensor {
	return Min(tensor)
}

// Reduce a tensor to its minimum value along the given dimension.
func MinByDim(tensor *Tensor, dim int, keep_dims bool) ValueIndexPair {
	values := &Tensor{}
	indices := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_MinByDim(
		&values.Pointer,
		&indices.Pointer,
		tensor.Pointer,
		C.int64_t(dim),
		C.bool(keep_dims),
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(values, (*Tensor).free)
	runtime.SetFinalizer(indices, (*Tensor).free)
	return ValueIndexPair{values, indices}
}

// Reduce a tensor to its minimum value along the given dimension.
func (tensor *Tensor) MinByDim(dim int, keep_dims bool) ValueIndexPair {
	return MinByDim(tensor, dim, keep_dims)
}

// Reduce a tensor to its mean value.
func Mean(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Mean(&output.Pointer, tensor.Pointer)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Reduce a tensor to its mean value.
func (tensor *Tensor) Mean() *Tensor {
	return Mean(tensor)
}

// Reduce a tensor to its mean value along the given dimension.
func MeanByDim(tensor *Tensor, dim int, keep_dims bool) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_MeanByDim(
		&output.Pointer,
		tensor.Pointer,
		C.int64_t(dim),
		C.bool(keep_dims),
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Reduce a tensor to its mean value along the given dimension.
func (tensor *Tensor) MeanByDim(dim int, keep_dims bool) *Tensor {
	return MeanByDim(tensor, dim, keep_dims)
}

// TODO: nanmean

// Reduce a tensor to its median value.
func Median(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Median(&output.Pointer, tensor.Pointer)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Reduce a tensor to its median value.
func (tensor *Tensor) Median() *Tensor {
	return Median(tensor)
}

// Reduce a tensor to its median value along the given dimension.
func MedianByDim(tensor *Tensor, dim int, keep_dims bool) ValueIndexPair {
	values := &Tensor{}
	indices := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_MedianByDim(
		&values.Pointer,
		&indices.Pointer,
		tensor.Pointer,
		C.int64_t(dim),
		C.bool(keep_dims),
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(values, (*Tensor).free)
	runtime.SetFinalizer(indices, (*Tensor).free)
	return ValueIndexPair{values, indices}
}

// Reduce a tensor to its median value along the given dimension.
func (tensor *Tensor) MedianByDim(dim int, keep_dims bool) ValueIndexPair {
	return MedianByDim(tensor, dim, keep_dims)
}

// TODO: nanmedian
// TODO: mode
// TODO: norm
// TODO: nansum
// TODO: prod
// TODO: quantile
// TODO: nanquantile

// Reduce a tensor to its standard deviation.
func Std(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Std(&output.Pointer, tensor.Pointer)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Reduce a tensor to its standard deviation.
func (tensor *Tensor) Std() *Tensor {
	return Std(tensor)
}

// Reduce a tensor to its standard deviation along the given dimension.
func StdByDim(tensor *Tensor, dim int, unbiased bool, keep_dims bool) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_StdByDim(
		&output.Pointer,
		tensor.Pointer,
		C.int64_t(dim),
		C.bool(unbiased),
		C.bool(keep_dims),
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Reduce a tensor to its standard deviation along the given dimension.
func (tensor *Tensor) StdByDim(dim int, unbiased bool, keep_dims bool) *Tensor {
	return StdByDim(tensor, dim, unbiased, keep_dims)
}

// Reduce a tensor to its mean value and standard deviation.
func StdMean(tensor *Tensor) (*Tensor, *Tensor) {
	std := &Tensor{}
	mean := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_StdMean(
		&std.Pointer,
		&mean.Pointer,
		tensor.Pointer,
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(std, (*Tensor).free)
	runtime.SetFinalizer(mean, (*Tensor).free)
	return std, mean
}

// Reduce a tensor to its mean value and standard deviation.
func (tensor *Tensor) StdMean() (*Tensor, *Tensor) {
	return StdMean(tensor)
}

// Reduce a tensor to its mean value and standard deviation along given dimension.
func StdMeanByDim(tensor *Tensor, dim int, unbiased, keep_dims bool) (*Tensor, *Tensor) {
	std := &Tensor{}
	mean := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_StdMeanByDim(
		&std.Pointer,
		&mean.Pointer,
		tensor.Pointer,
		C.int64_t(dim),
		C.bool(unbiased),
		C.bool(keep_dims),
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(std, (*Tensor).free)
	runtime.SetFinalizer(mean, (*Tensor).free)
	return std, mean
}

// Reduce a tensor to its mean value and standard deviation along given dimension.
func (tensor *Tensor) StdMeanByDim(dim int, unbiased, keep_dims bool) (*Tensor, *Tensor) {
	return StdMeanByDim(tensor, dim, unbiased, keep_dims)
}

// Reduce a tensor to its sum.
func Sum(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Sum(&output.Pointer, tensor.Pointer)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Reduce a tensor to its sum.
func (tensor *Tensor) Sum() *Tensor {
	return Sum(tensor)
}

// Reduce a tensor to its sum along the given dimension.
func SumByDim(tensor *Tensor, dim int, keep_dims bool) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_SumByDim(
		&output.Pointer,
		tensor.Pointer,
		C.int64_t(dim),
		C.bool(keep_dims),
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Reduce a tensor to its sum along the given dimension.
func (tensor *Tensor) SumByDim(dim int, keep_dims bool) *Tensor {
	return SumByDim(tensor, dim, keep_dims)
}

// TODO: unique
// TODO: unique_consecutive

// Reduce a tensor to its variance.
func Var(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Var(&output.Pointer, tensor.Pointer)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Reduce a tensor to its variance.
func (tensor *Tensor) Var() *Tensor {
	return Var(tensor)
}

// Reduce a tensor to its variance along the given dimension.
func VarByDim(tensor *Tensor, dim int, unbiased, keep_dims bool) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_VarByDim(
		&output.Pointer,
		tensor.Pointer,
		C.int64_t(dim),
		C.bool(unbiased),
		C.bool(keep_dims),
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Reduce a tensor to its variance along the given dimension.
func (tensor *Tensor) VarByDim(dim int, unbiased, keep_dims bool) *Tensor {
	return VarByDim(tensor, dim, unbiased, keep_dims)
}

// Reduce a tensor to its mean value and variance.
func VarMean(tensor *Tensor) (*Tensor, *Tensor) {
	variance := &Tensor{}
	mean := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_VarMean(
		&variance.Pointer,
		&mean.Pointer,
		tensor.Pointer,
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(variance, (*Tensor).free)
	runtime.SetFinalizer(mean, (*Tensor).free)
	return variance, mean
}

// Reduce a tensor to its mean value and variance.
func (tensor *Tensor) VarMean() (*Tensor, *Tensor) {
	return VarMean(tensor)
}

// Reduce a tensor to its mean value and variance along given dimension.
func VarMeanByDim(tensor *Tensor, dim int, unbiased, keep_dims bool) (*Tensor, *Tensor) {
	variance := &Tensor{}
	mean := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_VarMeanByDim(
		&variance.Pointer,
		&mean.Pointer,
		tensor.Pointer,
		C.int64_t(dim),
		C.bool(unbiased),
		C.bool(keep_dims),
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(variance, (*Tensor).free)
	runtime.SetFinalizer(mean, (*Tensor).free)
	return variance, mean
}

// Reduce a tensor to its mean value and variance along given dimension.
func (tensor *Tensor) VarMeanByDim(dim int, unbiased, keep_dims bool) (*Tensor, *Tensor) {
	return VarMeanByDim(tensor, dim, unbiased, keep_dims)
}

// ---------------------------------------------------------------------------
// MARK: Comparison Ops
// ---------------------------------------------------------------------------

// Return true if the two tensors are approximately equal to one another with
// given relative and absolute tolerances.
//
// This function checks if all `tensor` and `other` satisfy the condition:
//
// |tensor − other| <= atol + rtol * |other|
//
func AllClose(tensor, other *Tensor, rtol, atol float64) bool {
	var output C.bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_AllClose(
		&output,
		tensor.Pointer,
		other.Pointer,
		C.double(rtol),
		C.double(atol),
	)))
    runtime.KeepAlive(tensor)
    runtime.KeepAlive(other)
	return bool(output)
}

// Return true if the tensor is approximately equal to another with given
// relative and absolute tolerances.
//
// This function checks if all `tensor` and `other` satisfy the condition:
//
// |tensor − other| <= atol + rtol * |other|
//
func (tensor *Tensor) AllClose(other *Tensor, rtol, atol float64) bool {
	return AllClose(tensor, other, rtol, atol)
}

// Create a new tensor describing the element-wise approximate equality of two
// tensors with given relative and absolute tolerances.
//
// This function checks if all `tensor` and `other` satisfy the condition:
//
// |tensor − other| <= atol + rtol * |other|
//
func IsClose(tensor, other *Tensor, rtol, atol float64) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IsClose(
		&output.Pointer,
		tensor.Pointer,
		other.Pointer,
		C.double(rtol),
		C.double(atol),
	)))
    runtime.KeepAlive(tensor)
    runtime.KeepAlive(other)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Create a new tensor describing the element-wise approximate equality of the
// tensor to another with given relative and absolute tolerances.
//
// This function checks if all `tensor` and `other` satisfy the condition:
//
// |tensor − other| <= atol + rtol * |other|
//
func (tensor *Tensor) IsClose(other *Tensor, rtol, atol float64) *Tensor {
	return IsClose(tensor, other, rtol, atol)
}

// TODO: Argsort

// Create a new tensor comparing the element-wise equality of two tensors.
func Eq(tensor, other *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Eq(
		&output.Pointer,
		tensor.Pointer,
		other.Pointer,
	)))
    runtime.KeepAlive(tensor)
    runtime.KeepAlive(other)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Create a new tensor comparing element-wise equality of the tensor to another.
func (tensor *Tensor) Eq(other *Tensor) *Tensor {
	return Eq(tensor, other)
}

// Return true if the two tensors are precisely equal element-wise.
func Equal(tensor, other *Tensor) bool {
	var output bool
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Equal(
		(*C.bool)(&output),
		tensor.Pointer,
		other.Pointer,
	)))
    runtime.KeepAlive(tensor)
    runtime.KeepAlive(other)
	return output
}

// Return true if the tensor is precisely equal to other element-wise.
func (tensor *Tensor) Equal(other *Tensor) bool {
	return Equal(tensor, other)
}


// Compute tensor >= other element-wise.
func GreaterEqual(tensor, other *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_GreaterEqual(
		&output.Pointer,
		tensor.Pointer,
		other.Pointer,
	)))
    runtime.KeepAlive(tensor)
    runtime.KeepAlive(other)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Compute tensor >= other element-wise.
func (tensor *Tensor) GreaterEqual(other *Tensor) *Tensor {
	return GreaterEqual(tensor, other)
}

// Compute tensor > other element-wise.
func Greater(tensor, other *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Greater(
		&output.Pointer,
		tensor.Pointer,
		other.Pointer,
	)))
    runtime.KeepAlive(tensor)
    runtime.KeepAlive(other)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Compute tensor > other element-wise.
func (tensor *Tensor) Greater(other *Tensor) *Tensor {
	return Greater(tensor, other)
}

// Compute tensor <= other element-wise.
func LessEqual(tensor, other *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_LessEqual(
		&output.Pointer,
		tensor.Pointer,
		other.Pointer,
	)))
    runtime.KeepAlive(tensor)
    runtime.KeepAlive(other)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Compute tensor <= other element-wise.
func (tensor *Tensor) LessEqual(other *Tensor) *Tensor {
	return LessEqual(tensor, other)
}

// Compute tensor < other element-wise.
func Less(tensor, other *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Less(
		&output.Pointer,
		tensor.Pointer,
		other.Pointer,
	)))
    runtime.KeepAlive(tensor)
    runtime.KeepAlive(other)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Compute tensor < other element-wise.
func (tensor *Tensor) Less(other *Tensor) *Tensor {
	return Less(tensor, other)
}

// Computes the element-wise maximum of input and other.
func Maximum(tensor, other *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Maximum(
		&output.Pointer,
		tensor.Pointer,
		other.Pointer,
	)))
    runtime.KeepAlive(tensor)
    runtime.KeepAlive(other)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Computes the element-wise maximum of input and other.
func (tensor *Tensor) Maximum(other *Tensor) *Tensor {
	return Maximum(tensor, other)
}

// Computes the element-wise minimum of input and other.
func Minimum(tensor, other *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Minimum(
		&output.Pointer,
		tensor.Pointer,
		other.Pointer,
	)))
    runtime.KeepAlive(tensor)
    runtime.KeepAlive(other)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Computes the element-wise minimum of input and other.
func (tensor *Tensor) Minimum(other *Tensor) *Tensor {
	return Minimum(tensor, other)
}

// Compute tensor != other element-wise.
func NotEqual(tensor, other *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_NotEqual(
		&output.Pointer,
		tensor.Pointer,
		other.Pointer,
	)))
    runtime.KeepAlive(tensor)
    runtime.KeepAlive(other)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Compute tensor != other element-wise.
func (tensor *Tensor) NotEqual(other *Tensor) *Tensor {
	return NotEqual(tensor, other)
}

// Test if each element of tensor is in other. Returns a boolean tensor of the
// same shape as tensor that is True for elements in other and False otherwise.
func IsIn(tensor, other *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IsIn(
		&output.Pointer,
		tensor.Pointer,
		other.Pointer,
	)))
    runtime.KeepAlive(tensor)
    runtime.KeepAlive(other)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Test if each element of tensor is in other. Returns a boolean tensor of the
// same shape as tensor that is True for elements in other and False otherwise.
func (tensor *Tensor) IsIn(other *Tensor) *Tensor {
	return IsIn(tensor, other)
}

// Test if each element of the tensor is finite or not.
func IsFinite(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IsFinite(&output.Pointer, tensor.Pointer)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Test if each element of the tensor is finite or not.
func (tensor *Tensor) IsFinite() *Tensor {
	return IsFinite(tensor)
}

// Test if each element of the tensor is infinite (positive or negative) or not.
func IsInf(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IsInf(&output.Pointer, tensor.Pointer)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Test if each element of the tensor is infinite (positive or negative) or not.
func (tensor *Tensor) IsInf() *Tensor {
	return IsInf(tensor)
}

// Test if each element of the tensor is positive infinity or not.
func IsPosInf(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IsPosInf(&output.Pointer, tensor.Pointer)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Test if each element of the tensor is positive infinity or not.
func (tensor *Tensor) IsPosInf() *Tensor {
	return IsPosInf(tensor)
}

// Test if each element of the tensor is negative infinity or not.
func IsNegInf(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IsNegInf(&output.Pointer, tensor.Pointer)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Test if each element of the tensor is negative infinity or not.
func (tensor *Tensor) IsNegInf() *Tensor {
	return IsNegInf(tensor)
}

// Returns a new tensor with boolean elements representing if each element of
// input is NaN or not. Complex values are considered NaN when either their
// real and/or imaginary part is NaN.
func IsNaN(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IsNan(&output.Pointer, tensor.Pointer)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Returns a new tensor with boolean elements representing if each element of
// input is NaN or not. Complex values are considered NaN when either their
// real and/or imaginary part is NaN.
func (tensor *Tensor) IsNaN() *Tensor {
	return IsNaN(tensor)
}

// Return a new tensor with boolean elements representing if each element of
// input is real-valued or not. All real-valued types are considered real.
// Complex values are considered real when their imaginary part is 0.
func IsReal(tensor *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_IsReal(&output.Pointer, tensor.Pointer)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Return a new tensor with boolean elements representing if each element of
// input is real-valued or not. All real-valued types are considered real.
// Complex values are considered real when their imaginary part is 0.
func (tensor *Tensor) IsReal() *Tensor {
	return IsReal(tensor)
}

// TODO: KthValue

// Return the k largest elements of the given input tensor along a given
// dimension. If largest is False then the k smallest elements are returned.
// A struct of (Values, Indices) is returned with the values and indices
// of the largest k elements of each row of the input tensor in the given
// dimension dim. The boolean option sorted if True, will make sure that the
// returned k elements are themselves sorted.
func TopK(tensor *Tensor, k, dim int64, largest, sorted bool) ValueIndexPair {
	values := &Tensor{}
	indices := &Tensor{}
	l := 0
	if largest {
		l = 1
	}
	s := 0
	if sorted {
		s = 1
	}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_TopK(
		&values.Pointer,
		&indices.Pointer,
		tensor.Pointer,
		C.int64_t(k),
		C.int64_t(dim),
		C.int8_t(l),
		C.int8_t(s),
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(values, (*Tensor).free)
	runtime.SetFinalizer(indices, (*Tensor).free)
	return ValueIndexPair{values, indices}
}

// Return the k largest elements of the given input tensor along a given
// dimension. If largest is False then the k smallest elements are returned.
// A struct of (Values, Indices) is returned with the values and indices
// of the largest k elements of each row of the input tensor in the given
// dimension dim. The boolean option sorted if True, will make sure that the
// returned k elements are themselves sorted.
func (tensor *Tensor) TopK(k, dim int64, largest, sorted bool) ValueIndexPair {
	return TopK(tensor, k, dim, largest, sorted)
}

// Sort the elements of the input tensor along a given dimension by value. If
// descending is True then the elements are sorted in descending order by
// value. A struct of (Values, Indices) is returned, where the values are the
// sorted values and indices are the indices of the elements in the original
// input tensor.
func Sort(tensor *Tensor, dim int64, descending bool) ValueIndexPair {
	values := &Tensor{}
	indices := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_Sort(
		&values.Pointer,
		&indices.Pointer,
		tensor.Pointer,
		C.int64_t(dim),
		C.bool(descending),
	)))
    runtime.KeepAlive(tensor)
	runtime.SetFinalizer(values, (*Tensor).free)
	runtime.SetFinalizer(indices, (*Tensor).free)
	return ValueIndexPair{values, indices}
}

// Sort the elements of the input tensor along a given dimension by value. If
// descending is True then the elements are sorted in descending order by
// value. A struct of (Values, Indices) is returned, where the values are the
// sorted values and indices are the indices of the elements in the original
// input tensor.
func (tensor *Tensor) Sort(dim int64, descending bool) ValueIndexPair {
	return Sort(tensor, dim, descending)
}

// ---------------------------------------------------------------------------
// MARK: Spectral Operations
// ---------------------------------------------------------------------------

// TODO: stft
// TODO: istft
// TODO: bartlett_window
// TODO: blackman_window
// TODO: hamming_window
// TODO: hann_window
// TODO: kaiser_window

// ---------------------------------------------------------------------------
// MARK: Other Operations
// ---------------------------------------------------------------------------

// TODO: atleast_1d
// TODO: atleast_2d
// TODO: atleast_3d
// TODO: bincount
// TODO: block_diag
// TODO: broadcast_tensors
// TODO: broadcast_to
// TODO: broadcast_shapes
// TODO: bucketize
// TODO: cartesian_prod
// TODO: cdist
// TODO: clone
// TODO: combinations
// TODO: corrcoef
// TODO: cov
// TODO: cross
// TODO: cummax
// TODO: cummin
// TODO: cumprod
// TODO: cumsum
// TODO: diag
// TODO: diag_embed
// TODO: diagflat
// TODO: diagonal
// TODO: diff
// TODO: einsum
// TODO: flatten
// TODO: flip
// TODO: fliplr
// TODO: flipud
// TODO: kron
// TODO: rot90
// TODO: gcd
// TODO: histc
// TODO: histogram
// TODO: histogramdd
// TODO: meshgrid
// TODO: lcm
// TODO: logcumsumexp
// TODO: ravel
// TODO: renorm
// TODO: repeat_interleave
// TODO: roll
// TODO: searchsorted
// TODO: tensordot
// TODO: trace
// TODO: tril
// TODO: tril_indices
// TODO: triu
// TODO: triu_indices
// TODO: unflatten
// TODO: vander
// TODO: view_as_real
// TODO: view_as_complex
// TODO: resolve_conj
// TODO: resolve_neg

// ---------------------------------------------------------------------------
// MARK: BLAS and LAPACK Operations
// ---------------------------------------------------------------------------

// TODO: addbmm
// TODO: addmm
// TODO: addmv
// TODO: addr
// TODO: baddbmm
// TODO: bmm
// TODO: chain_matmul
// TODO: cholesky
// TODO: cholesky_inverse
// TODO: cholesky_solve
// TODO: dot
// TODO: geqrf
// TODO: ger
// TODO: inner
// TODO: inverse
// TODO: det
// TODO: logdet
// TODO: slogdet
// TODO: lu
// TODO: lu_solve
// TODO: lu_unpack
// TODO: matmul
// TODO: matrix_power
// TODO: matrix_exp

// Perform a matrix multiplication of the matrices A and B. If A is an (n×m)
// tensor, B is an (m×p) tensor, the output will be an (n×p) tensor.
func MM(a, b *Tensor) *Tensor {
	output := &Tensor{}
	internal.PanicOnCException(unsafe.Pointer(C.Torch_MM(&output.Pointer, a.Pointer, b.Pointer)))
    runtime.KeepAlive(a)
    runtime.KeepAlive(b)
	runtime.SetFinalizer(output, (*Tensor).free)
	return output
}

// Perform a matrix multiplication of the matrices A and B. If A is an (n×m)
// tensor, B is an (m×p) tensor, the output will be an (n×p) tensor.
func (tensor *Tensor) MM(other *Tensor) *Tensor {
	return MM(tensor, other)
}

// TODO: mv
// TODO: orgqr
// TODO: ormqr
// TODO: outer
// TODO: pinverse
// TODO: qr
// TODO: svd
// TODO: svd_lowrank
// TODO: pca_lowrank
// TODO: symeig
// TODO: lobpcg
// TODO: trapz
// TODO: trapezoid
// TODO: cumulative_trapezoid
// TODO: triangular_solve
// TODO: vdot
