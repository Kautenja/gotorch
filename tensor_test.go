// Test cases for tensor.go
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

package torch_test

import (
    "math"
    "errors"
    "crypto/md5"
    "fmt"
    "os"
    "reflect"
    "testing"
    "unsafe"
    "path/filepath"
    "github.com/stretchr/testify/assert"
    "github.com/x448/float16"
    torch "github.com/Kautenja/gotorch"
)

// MARK: TensorFromBlob

func Test_Torch_TensorFromBlob(t *testing.T) {
    data := [2][3]float32{{1.0, 1.1, 1.2}, {2, 3, 4}}
    tensor := torch.TensorFromBlob(unsafe.Pointer(&data), torch.Float, []int64{2, 3})
    assert.Equal(t, []int64{2, 3}, tensor.Shape())
}

func Test_Torch_TensorFromBlob_WrapsMemory(t *testing.T) {
    data := [1]float32{1.0}
    tensor := torch.TensorFromBlob(unsafe.Pointer(&data), torch.Float, []int64{1})
    assert.Equal(t, []int64{1}, tensor.Shape())
    assert.Equal(t, float32(1), tensor.Item().(float32))
    data[0] = 2
    assert.Equal(t, float32(2), tensor.Item().(float32))
}

func Test_Torch_TensorFromBlob_PanicsOnInvalidSize(t *testing.T) {
    data := [1]float32{1.0}
    assert.PanicsWithValue(t, "Trying to create tensor with negative dimension -1: [-1]", func() {
        torch.TensorFromBlob(unsafe.Pointer(&data), torch.Float, []int64{-1})
    })
}

// MARK: NewTensorFromBlob

func Test_Torch_NewTensorFromBlob(t *testing.T) {
    data := [2][3]float32{{1.0, 1.1, 1.2}, {2, 3, 4}}
    tensor := torch.NewTensorFromBlob(unsafe.Pointer(&data), torch.Float, []int64{2, 3})
    assert.Equal(t, []int64{2, 3}, tensor.Shape())
}

func Test_Torch_NewTensorFromBlob_CopiesMemory(t *testing.T) {
    data := [1]float32{1.0}
    tensor := torch.NewTensorFromBlob(unsafe.Pointer(&data), torch.Float, []int64{1})
    assert.Equal(t, []int64{1}, tensor.Shape())
    assert.Equal(t, float32(1), tensor.Item().(float32))
    data[0] = 2
    assert.Equal(t, float32(1), tensor.Item().(float32))
}

func Test_Torch_NewTensorFromBlob_PanicsOnInvalidSize(t *testing.T) {
    data := [1]float32{1.0}
    assert.PanicsWithValue(t, "Trying to create tensor with negative dimension -1: [-1]", func() {
        torch.NewTensorFromBlob(unsafe.Pointer(&data), torch.Float, []int64{-1})
    })
}

// MARK: NewTensor

// func Test_Torch_NewTensor_WithEmptyData(t *testing.T) {
//     tensor := torch.NewTensor([]float32{})
// }

// TODO: this test makes invalid assumptions and sometimes fails
// func Test_Torch_NewTensor_WithRaggedData(t *testing.T) {
//     tensor := torch.NewTensor([][]float32{{0, 1}, {1}, {}})
//     expected := torch.NewTensor([][]float32{{0, 1}, {1, 0}, {0, 0}})
//     assert.True(t, tensor.AllClose(expected, 1e-8, 1e-5), "Got %v, expected %v", tensor, expected)
// }

func Test_Torch_NewTensor_WithComplex64DataType(t *testing.T) {
    assert.NotPanics(t, func() {
        torch.NewTensor([]complex64{})
    })
}

func Test_Torch_NewTensor_WithComplex128DataType(t *testing.T) {
    assert.NotPanics(t, func() {
        torch.NewTensor([]complex128{})
    })
}

func Test_Torch_NewTensor_PanicsOnNonSliceInput(t *testing.T) {
    assert.PanicsWithValue(t, "Expected slice but got data of type string", func() {
        _ = torch.NewTensor("foo")
    })
}

func Test_Torch_NewTensor_PanicsOnStringDataType(t *testing.T) {
    assert.PanicsWithValue(t, "Unrecognized dtype kind string", func() {
        _ = torch.NewTensor([]string{})
    })
}

func Test_Torch_NewTensor_PanicsOnIntDataType(t *testing.T) {
    assert.PanicsWithValue(t, "Unrecognized dtype kind int", func() {
        _ = torch.NewTensor([]int{})
    })
}

func Test_Torch_NewTensor_PanicsOnUInt32DataType(t *testing.T) {
    assert.PanicsWithValue(t, "Unrecognized dtype kind uint32", func() {
        _ = torch.NewTensor([]uint32{})
    })
}

func Test_Torch_NewTensor_PanicsOnUInt64DataType(t *testing.T) {
    assert.PanicsWithValue(t, "Unrecognized dtype kind uint64", func() {
        _ = torch.NewTensor([]uint64{})
    })
}

func Test_Torch_NewTensor_PanicsOnUIntPtrDataType(t *testing.T) {
    assert.PanicsWithValue(t, "Unrecognized dtype kind uintptr", func() {
        _ = torch.NewTensor([]uintptr{})
    })
}

// MARK: Clone

func Test_Tensor_Clone(t *testing.T) {
    tensor := torch.NewTensor([]float32{1})
    clone := tensor.Clone()
    assert.True(t, torch.Equal(tensor, clone))
    // Set the data of the input to new values; the clone should not change.
    other := torch.NewTensor([]float32{0})
    tensor.SetData(other)
    assert.False(t, torch.Equal(tensor, clone))
}

// MARK: String

func Test_Tensor_StringDouble(t *testing.T) {
    tensor := torch.NewTensor([]float64{})
    assert.Equal(t, "[ CPUDoubleType{0} ]", tensor.String())
}

func Test_Tensor_StringFloat(t *testing.T) {
    tensor := torch.NewTensor([]float32{})
    assert.Equal(t, "[ CPUFloatType{0} ]", tensor.String())
}

func Test_Tensor_StringHalf(t *testing.T) {
    tensor := torch.NewTensor([]uint16{})
    assert.Equal(t, "[ CPUHalfType{0} ]", tensor.String())
}

func Test_Tensor_StringLong(t *testing.T) {
    tensor := torch.NewTensor([]int64{})
    assert.Equal(t, "[ CPULongType{0} ]", tensor.String())
}

func Test_Tensor_StringInt(t *testing.T) {
    tensor := torch.NewTensor([]int32{})
    assert.Equal(t, "[ CPUIntType{0} ]", tensor.String())
}

func Test_Tensor_StringShort(t *testing.T) {
    tensor := torch.NewTensor([]int16{})
    assert.Equal(t, "[ CPUShortType{0} ]", tensor.String())
}

func Test_Tensor_StringChar(t *testing.T) {
    tensor := torch.NewTensor([]int8{})
    assert.Equal(t, "[ CPUCharType{0} ]", tensor.String())
}

func Test_Tensor_StringByte(t *testing.T) {
    tensor := torch.NewTensor([]byte{})
    assert.Equal(t, "[ CPUByteType{0} ]", tensor.String())
}

func Test_Tensor_StringBool(t *testing.T) {
    tensor := torch.NewTensor([]bool{})
    assert.Equal(t, "[ CPUBoolType{0} ]", tensor.String())
}

func Test_Tensor_StringVector(t *testing.T) {
    tensor := torch.NewTensor([]float32{1, 1})
    assert.Equal(t, ` 1
 1
[ CPUFloatType{2} ]`, tensor.String())
}

func Test_Tensor_StringColumnVector(t *testing.T) {
    tensor := torch.NewTensor([][]float32{{1}, {1}})
    assert.Equal(t, ` 1
 1
[ CPUFloatType{2,1} ]`, tensor.String())
}

func Test_Tensor_StringRowVector(t *testing.T) {
    tensor := torch.NewTensor([][]float32{{1, 1}})
    assert.Equal(t, ` 1  1
[ CPUFloatType{1,2} ]`, tensor.String())
}

func Test_Tensor_StringMatrix(t *testing.T) {
    tensor := torch.NewTensor([][]float32{{1, 1, 1}, {1, 1, 1}})
    assert.Equal(t, ` 1  1  1
 1  1  1
[ CPUFloatType{2,3} ]`, tensor.String())
}

// MARK: Save & Load

func Test_Tensor_SaveReturnsErrorOnInvalidPath(t *testing.T) {
    tensor := torch.NewTensor([][]float32{{1.0, 2.0}, {3.0, 4.0}})
    save_err := tensor.Save("./foo/bar/baz.pt")
    assert.NotNil(t, save_err)
    assert.Equal(t, "File ./foo/bar/baz.pt cannot be opened.", save_err.Error())
}

func Test_Tensor_LoadReturnsErrorOnInvalidPath(t *testing.T) {
    // create and open a temporary directory to load files from.
    temp_dir, err := os.MkdirTemp("/tmp/", "TestTensorLoadReturnsErrorOnInvalidPath")
    defer os.RemoveAll(temp_dir)  // defer clean up until the end of the test
    if err != nil {
        t.Log("Failed to create temporary directory")
        t.Fail()
        return
    }
    // Attempt to load a file that doesn't exist.
    output_path := filepath.Join(temp_dir, "foo.pt")
    tensor, err := torch.Load(output_path)
    assert.NotNil(t, err)
    assert.Nil(t, tensor.T)
    assert.Contains(t, err.Error(), "open file failed because of errno 2 on fopen:")
}

func Test_Tensor_SaveLoad(t *testing.T) {
    // create and open a temporary directory to save/load files to/from
    temp_dir, err := os.MkdirTemp("/tmp/", "TestSaveAndLoad")
    defer os.RemoveAll(temp_dir)  // defer clean up until the end of the test
    if err != nil {
        t.Log("Failed to create temporary directory")
        t.Fail()
        return
    }
    // Save the tensor to the temporary directory
    tensor := torch.NewTensor([][]float32{{1.0, 2.0}, {3.0, 4.0}})
    output_path := filepath.Join(temp_dir, "foo.pt")
    save_err := tensor.Save(output_path)
    assert.Nil(t, save_err)
    // Check that the data exists on the filesystem
    if _, err := os.Stat(output_path); errors.Is(err, os.ErrNotExist) {
        t.Log(fmt.Sprintf("Failed to save tensor to filesystem: %s", err.Error()))
        t.Fail()
        return
    }
    // Load the data from the filesystem into a new tensor
    deserialized_tensor, load_err := torch.Load(output_path)
    assert.Nil(t, load_err)
    assert.NotNil(t, deserialized_tensor.T)
    assert.True(t, torch.Equal(tensor, deserialized_tensor))
}

// Encode & Decode

func Test_Tensor_EncodeDecode(t *testing.T) {
    tensor := torch.NewTensor([][]float32{
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
    })
    bytes, err := tensor.Encode()
    assert.NoError(t, err)
    // The ground-truth length and MD5 checksum come from the C++ program
    // example/pickle.
    assert.Equal(t, 747, len(bytes))
    assert.Equal(t, fmt.Sprintf("%x", md5.Sum(bytes)), "dd65752601bf4d4ca19ae903baf96799")
    // The decoded tensor should be the same.
    decoded_tensor, err := torch.Decode(bytes)
    assert.NoError(t, err)
    assert.True(t, torch.Equal(tensor, decoded_tensor))
}

func Test_Tensor_DecodeShouldReturnErrorOnInvalidBuffer(t *testing.T) {
    bytes := []byte{1, 2, 3}
    tensor, err := torch.Decode(bytes)
    assert.Nil(t, tensor.T)
    assert.NotNil(t, err)
}

// MARK: Dtype

func Test_Tensor_DtypeDouble(t *testing.T) {
    tensor := torch.NewTensor([]float64{1})
    assert.Equal(t, torch.Double, tensor.Dtype())
}

func Test_Tensor_DtypeFloat(t *testing.T) {
    tensor := torch.NewTensor([]float32{1})
    assert.Equal(t, torch.Float, tensor.Dtype())
}

func Test_Tensor_DtypeHalf(t *testing.T) {
    tensor := torch.NewTensor([]uint16{float16.Fromfloat32(1).Bits()})
    assert.Equal(t, torch.Half, tensor.Dtype())
}

func Test_Tensor_DtypeLong(t *testing.T) {
    tensor := torch.NewTensor([]int64{1})
    assert.Equal(t, torch.Long, tensor.Dtype())
}

func Test_Tensor_DtypeInt(t *testing.T) {
    tensor := torch.NewTensor([]int32{1})
    assert.Equal(t, torch.Int, tensor.Dtype())
}

func Test_Tensor_DtypeShort(t *testing.T) {
    tensor := torch.NewTensor([]int16{1})
    assert.Equal(t, torch.Short, tensor.Dtype())
}

func Test_Tensor_DtypeChar(t *testing.T) {
    tensor := torch.NewTensor([]int8{1})
    assert.Equal(t, torch.Char, tensor.Dtype())
}

func Test_Tensor_DtypeByte(t *testing.T) {
    tensor := torch.NewTensor([]byte{1})
    assert.Equal(t, torch.Byte, tensor.Dtype())
}

func Test_Tensor_DtypeBool(t *testing.T) {
    tensor := torch.NewTensor([]bool{true})
    assert.Equal(t, torch.Bool, tensor.Dtype())
}

func Test_Tensor_DtypeComplexFloat(t *testing.T) {
    tensor := torch.NewTensor([]complex64{complex(1, 0.5)})
    assert.Equal(t, torch.ComplexFloat, tensor.Dtype())
}

func Test_Tensor_DtypeComplexDouble(t *testing.T) {
    tensor := torch.NewTensor([]complex128{complex(1, 0.5)})
    assert.Equal(t, torch.ComplexDouble, tensor.Dtype())
}

// MARK: Dim

func Test_Tensor_Dim(t *testing.T) {
    assert.Equal(t, int64(1), torch.Rand([]int64{1}, torch.NewTensorOptions()).Dim())
    assert.Equal(t, int64(2), torch.Rand([]int64{1, 1}, torch.NewTensorOptions()).Dim())
    assert.Equal(t, int64(3), torch.Rand([]int64{1, 1, 1}, torch.NewTensorOptions()).Dim())
}

// MARK: Shape

func Test_Tensor_ShapeEmpty(t *testing.T) {
    tensor := torch.NewTensor([]float32{})
    assert.Equal(t, []int64{0}, tensor.Shape())
}

func Test_Tensor_ShapeItem(t *testing.T) {
    tensor := torch.NewTensor([]float32{1})
    assert.Equal(t, []int64{1}, tensor.Shape())
}

func Test_Tensor_ShapeVector(t *testing.T) {
    tensor := torch.NewTensor([]float32{1, 1})
    assert.Equal(t, []int64{2}, tensor.Shape())
}

func Test_Tensor_ShapeColumnVector(t *testing.T) {
    tensor := torch.NewTensor([][]float32{{1}, {1}})
    assert.Equal(t, []int64{2, 1}, tensor.Shape())
}

func Test_Tensor_ShapeRowVector(t *testing.T) {
    tensor := torch.NewTensor([][]float32{{1, 1}})
    assert.Equal(t, []int64{1, 2}, tensor.Shape())
}

func Test_Tensor_ShapeMatrix(t *testing.T) {
    tensor := torch.NewTensor([][]float32{{1, 1, 1}, {1, 1, 1}})
    assert.Equal(t, []int64{2, 3}, tensor.Shape())
}

func Test_Tensor_Shape3DTensor(t *testing.T) {
    tensor := torch.NewTensor([][][]float32{
        {
            {1, 1, 1, 1},
            {1, 1, 1, 1},
        },
        {
            {1, 1, 1, 1},
            {1, 1, 1, 1},
        },
        {
            {1, 1, 1, 1},
            {1, 1, 1, 1},
        },
    })
    assert.Equal(t, []int64{3, 2, 4}, tensor.Shape())
}

// MARK: View

// >>> a = torch.randn(1, 2, 3, 4)
// >>> a.size()
// torch.Size([1, 2, 3, 4])
// >>> b = a.transpose(1, 2)  # Swaps 2nd and 3rd dimension
// >>> b.size()
// torch.Size([1, 3, 2, 4])
// >>> c = a.view(1, 3, 2, 4)  # Does not change tensor layout in memory
// >>> c.size()
// torch.Size([1, 3, 2, 4])
// >>> torch.equal(b, c)
// False
func Test_Tensor_View(t *testing.T) {
    tensor := torch.Rand([]int64{4, 4}, torch.NewTensorOptions())
    y := tensor.View(16)
    assert.Equal(t, []int64{16}, y.Shape())
    z := tensor.View(-1, 8)
    assert.Equal(t, []int64{2, 8}, z.Shape())
    a := torch.Rand([]int64{1, 2, 3, 4}, torch.NewTensorOptions())
    b := a.Transpose(1, 2)
    c := a.View(1, 3, 2, 4)
    assert.False(t, torch.Equal(b, c))
}

// https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch
// >>> z = torch.zeros(3, 2)
// >>> y = z.t()
// >>> y.size()
// torch.Size([2, 3])
// >>> y.view(6)
// Traceback (most recent call last):
//   File "<stdin>", line 1, in <module>
// RuntimeError: invalid argument 2: view size is not compatible with input tensor's
// size and stride (at least one dimension spans across two contiguous subspaces).
// Call .contiguous() before .view().
func Test_Tensor_ViewDoesImposeContiguityConstraint(t *testing.T) {
    z := torch.Zeros([]int64{3, 2}, torch.NewTensorOptions())
    y := z.Transpose(0, 1)
    assert.PanicsWithValue(t, "view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.", func() {
        y.View(6)
    })
}

// >>> target = torch.zeros(2, 2)
// >>> inputs = torch.arange(4)
// >>> inputs.view_as(target)
// tensor([[0, 1],
//         [2, 3]])
func Test_Tensor_ViewAs(t *testing.T) {
    target := torch.Zeros([]int64{2, 2}, torch.NewTensorOptions())
    inputs := torch.Arange(0, 4, 1, torch.NewTensorOptions())
    output := inputs.ViewAs(target)
    expected := torch.NewTensor([][]float32{
        {0, 1},
        {2, 3},
    })
    assert.True(t, output.Equal(expected))
}

// https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch
// >>> z = torch.zeros(3, 2)
// >>> y = z.t()
// >>> y.size()
// torch.Size([2, 3])
// >>> x = torch.zeros(6)
// >>> y.view_as(x)
// Traceback (most recent call last):
//   File "<stdin>", line 1, in <module>
// RuntimeError: invalid argument 2: view size is not compatible with input tensor's
// size and stride (at least one dimension spans across two contiguous subspaces).
// Call .contiguous() before .view().
func Test_Tensor_ViewAsDoesImposeContiguityConstraint(t *testing.T) {
    z := torch.Zeros([]int64{3, 2}, torch.NewTensorOptions())
    y := z.Transpose(0, 1)
    x := torch.Zeros([]int64{6}, torch.NewTensorOptions())
    assert.PanicsWithValue(t, "view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.", func() {
        y.ViewAs(x)
    })
}

// MARK: Reshape

func Test_Tensor_Reshape(t *testing.T) {
    tensor := torch.Rand([]int64{4, 4}, torch.NewTensorOptions())
    y := torch.Reshape(tensor, 16)
    assert.Equal(t, []int64{16}, y.Shape())
    z := torch.Reshape(tensor, -1, 8)
    assert.Equal(t, []int64{2, 8}, z.Shape())
}

// https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch
// >>> z = torch.zeros(3, 2)
// >>> y = z.t()
// >>> y.size()
// torch.Size([2, 3])
// >>> y.reshape(6)
// tensor([0., 0., 0., 0., 0., 0.])
func Test_Tensor_ReshapeDoesNotImposeContiguityConstraint(t *testing.T) {
    z := torch.Zeros([]int64{3, 2}, torch.NewTensorOptions())
    y := z.Transpose(0, 1)
    var k torch.Tensor
    assert.NotPanics(t, func() { k = y.Reshape(6) })
    assert.Equal(t, []int64{6}, k.Shape())
}

// >>> target = torch.zeros(2, 2)
// >>> inputs = torch.arange(4)
// >>> inputs.view_as(target)
// tensor([[0, 1],
//         [2, 3]])
func Test_Tensor_ReshapeAs(t *testing.T) {
    target := torch.Zeros([]int64{2, 2}, torch.NewTensorOptions())
    inputs := torch.Arange(0, 4, 1, torch.NewTensorOptions())
    output := inputs.ReshapeAs(target)
    expected := torch.NewTensor([][]float32{
        {0, 1},
        {2, 3},
    })
    assert.True(t, output.Equal(expected))
}

// https://stackoverflow.com/questions/49643225/whats-the-difference-between-reshape-and-view-in-pytorch
// >>> z = torch.zeros(3, 2)
// >>> y = z.t()
// >>> y.size()
// torch.Size([2, 3])
// >>> x = torch.zeros(6)
// >>> y.reshape_as(x)
// tensor([0., 0., 0., 0., 0., 0.])
func Test_Tensor_ReshapeAsDoesNotImposeContiguityConstraint(t *testing.T) {
    z := torch.Zeros([]int64{3, 2}, torch.NewTensorOptions())
    y := z.Transpose(0, 1)
    x := torch.Zeros([]int64{6}, torch.NewTensorOptions())
    var k torch.Tensor
    assert.NotPanics(t, func() { k = y.ReshapeAs(x) })
    assert.Equal(t, []int64{6}, k.Shape())
}

// MARK: Expand

// >>> s = torch.tensor([1,2])
// >>> s.expand(2, 2)
// tensor([[1, 2],
//         [1, 2]])
func Test_Tensor_Expand(t *testing.T) {
    tensor := torch.NewTensor([][]float32{{0, 1}})
    output := tensor.Expand(2, 2)
    assert.Equal(t, []int64{2, 2}, output.Shape())
    expected := torch.NewTensor([][]float32{{0, 1}, {0, 1}})
    assert.True(t, output.Equal(expected))
}

// >>> s = torch.tensor([1,2])
// >>> t = torch.tensor([[1,2],[3,4]])
// >>> s.expand_as(t)
// tensor([[1, 2],
//         [1, 2]])
func Test_Tensor_ExpandAs(t *testing.T) {
    a := torch.NewTensor([]int8{'a', 'b'})
    b := torch.NewTensor([][]int8{{1, 2}, {3, 4}})
    c := a.ExpandAs(b)
    g := " 97  98\n 97  98\n[ CPUCharType{2,2} ]"
    assert.Equal(t, g, c.String())
}

// MARK: SetData

// Set data should in-place overwrite the tensor data. For blobs this
// implies that the data WILL NOT be overwritten like with `copy_`
func Test_Tensor_SetData(t *testing.T) {
    data := [1]float32{2.0}  // Use 2, which is above the range of Rand
    tensor := torch.TensorFromBlob(unsafe.Pointer(&data), torch.Float, []int64{1})
    other := torch.Rand([]int64{1}, torch.NewTensorOptions())
    tensor.SetData(other)
    assert.Equal(t, tensor.Item().(float32), other.Item().(float32))
    assert.NotEqual(t, other.Item().(float32), data[0])
    assert.Equal(t, float32(2.0), data[0])
}

// MARK: Copy_

// Copy should perform an in-place copy of another tensor. This means that
// if we create the tensor from a blob, it will copy into the Golang memory.
func Test_Tensor_Copy_(t *testing.T) {
    data := [1]float32{2.0}  // Use 2, which is above the range of Rand
    tensor := torch.TensorFromBlob(unsafe.Pointer(&data), torch.Float, []int64{1})
    other := torch.Rand([]int64{1}, torch.NewTensorOptions())
    tensor.Copy_(other)
    assert.Equal(t, tensor.Item().(float32), other.Item().(float32))
    assert.Equal(t, other.Item().(float32), data[0])
    assert.NotEqual(t, float32(2.0), data[0])
}

// MARK: CastTo

func Test_Tensor_CastTo(t *testing.T) {
    a := torch.NewTensor([]int64{1, 2})
    b := a.CastTo(torch.Float)
    assert.Equal(t, torch.Float, b.Dtype())
}

// MARK: CopyTo

func Test_Tensor_CopyTo(t *testing.T) {
    a := torch.NewTensor([]int64{1, 2})
    device := torch.NewDevice("cpu")
    b := a.CopyTo(device)
    assert.True(t, torch.Equal(a, b))
}

// MARK: To

func Test_Tensor_To(t *testing.T) {
    a := torch.NewTensor([]int64{1, 2})
    b := a.To(torch.NewDevice("cpu"), torch.Float)
    assert.Equal(t, torch.Float, b.Dtype())
}

// func Test_Tensor_CUDA(t *testing.T) {
//  a := assert.New(t)
//  device := getDefaultDevice()
//  input := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
//  if !torch.IsCUDAAvailable() {
//      a.Panics(func() {
//          input.CUDA(device, false)
//      })
//      a.Panics(func() {
//          input.CUDA(device, true)
//      })
//      return
//  }

//  b := input.CUDA(device, false)
//  a.Equal(" 1  2\n 3  4\n[ CUDAFloatType{2,2} ]", b.String())

//  c := input.CUDA(device, true)
//  torch.GetCurrentCUDAStream(device).Synchronize()
//  a.Equal(" 1  2\n 3  4\n[ CUDAFloatType{2,2} ]", c.String())
// }

// func Test_Tensor_PinMemory(t *testing.T) {
//     a := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
//     b := a.PinMemory()
//     if torch.IsCUDAAvailable() {
//         assert.Equal(t, " 1  2\n 3  4\n[ CUDAFloatType{2,2} ]", b.String())
//     } else {
//         assert.Equal(t, " 1  2\n 3  4\n[ CPUFloatType{2,2} ]", b.String())
//     }
// }

// MARK: SetGrad/RequiresGrad

func Test_Tensor_SetRequiresGrad(t *testing.T) {
    tensor := torch.NewTensor([]float32{1})
    assert.False(t, tensor.RequiresGrad())
    tensor.SetRequiresGrad(true)
    assert.True(t, tensor.RequiresGrad())
    tensor.SetRequiresGrad(false)
    assert.False(t, tensor.RequiresGrad())
}

// MARK: Backward/Grad

// >>> predictions = torch.tensor([[0, 1], [2, 1]], dtype=torch.float, requires_grad=True)
// >>> targets = torch.zeros(2, 2, dtype=torch.float, requires_grad=True)
// >>> predictions.sub(targets).square().mean().backward()
// >>> targets.grad
// tensor([[-0.0000, -0.5000],
//         [-1.0000, -0.5000]])
// >>> predictions.grad
// tensor([[0.0000, 0.5000],
//         [1.0000, 0.5000]])
func Test_Tensor_Backward(t *testing.T) {
    predictions := torch.NewTensor([][]float32{{0.0, 1.0}, {2.0, 1.0}})
    predictions.SetRequiresGrad(true)
    targets := torch.Zeros([]int64{2, 2}, torch.NewTensorOptions().RequiresGrad(true))
    predictions.Sub(targets, 1.0).Square().Mean().Backward()
    expected_targets := torch.NewTensor([][]float32{{-0.0, -0.5}, {-1.0, -0.5}})
    assert.True(t, targets.Grad().Equal(expected_targets))
    expected_predictions := torch.NewTensor([][]float32{{0.0, 0.5}, {1.0, 0.5}})
    assert.True(t, predictions.Grad().Equal(expected_predictions))
}

// MARK: Detach

func Test_Tensor_Detach(t *testing.T) {
    tensor := torch.NewTensor([]float32{1})
    tensor.SetRequiresGrad(true)
    assert.True(t, tensor.RequiresGrad())
    tensor = tensor.Detach()
    assert.False(t, tensor.RequiresGrad())
}

// MARK: Index

// func Test_Tensor_Index(t *testing.T) {
//     a := torch.NewTensor([][]float32{{1, 2}, {3, 4}})
//     assert.Equal(t, float32(1), a.Index(0, 0).Item().(float32))
//     assert.Equal(t, float32(2), a.Index(0, 1).Item().(float32))
//     assert.Equal(t, float32(3), a.Index(1, 0).Item().(float32))
//     assert.Equal(t, float32(4), a.Index(1, 1).Item().(float32))
//     assert.Panics(t, func() { a.Index(0).Item() })
//     assert.Panics(t, func() { a.Index(0, 0, 0).Item() })
// }

func Test_Tensor_Index(t *testing.T) {
    tensor := torch.NewTensor([]float32{1.0, 4.0, 8.0, 5.0, 9.0})
    index := torch.NewTensor([]bool{true, false, false, true, true})
    expected := torch.NewTensor([]float32{1.0, 5.0, 9.0})
    assert.True(t, tensor.Index(index).Equal(expected))
}

func Test_Tensor_Index_PanicsOnInvalidIndex(t *testing.T) {
    tensor := torch.NewTensor([]float32{1.0, 4.0, 8.0, 5.0, 9.0})
    index := torch.NewTensor([]bool{true, false, false, true, true, false})
    expected := "The shape of the mask [6] at index 0 does not match the shape of the indexed tensor [5] at index 0"
    assert.PanicsWithValue(t, expected, func() { tensor.Index(index) })
}

// MARK: Item

func Test_Tensor_ItemByte(t *testing.T) {
    // Zero (minimum value)
    x := torch.NewTensor([]byte{0})
    y := x.Item()
    assert.Equal(t, byte(0), y)
    assert.NotEqual(t, int8(0), y)
    assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Uint8)
    assert.Equal(t, byte(0), x.Item().(byte))
    // Positive reals
    x = torch.NewTensor([]byte{1})
    y = x.Item()
    assert.Equal(t, byte(1), y)
    assert.NotEqual(t, int8(1), y)
    assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Uint8)
    assert.Equal(t, byte(1), x.Item().(byte))
    // Maximum value
    x = torch.NewTensor([]byte{math.MaxUint8})
    y = x.Item()
    assert.Equal(t, byte(math.MaxUint8), y)
    assert.Equal(t, byte(math.MaxUint8), x.Item().(byte))
}

func Test_Tensor_ItemBool(t *testing.T) {
    // false
    x := torch.NewTensor([]bool{false})
    y := x.Item()
    assert.Equal(t, false, y)
    assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Bool)
    assert.False(t, x.Item().(bool))
    // true
    x = torch.NewTensor([]bool{true})
    y = x.Item()
    assert.Equal(t, true, y)
    assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Bool)
    assert.True(t, x.Item().(bool))
}

func Test_Tensor_ItemInt8(t *testing.T) {
    // Zero
    x := torch.NewTensor([]int8{0})
    y := x.Item()
    assert.Equal(t, int8(0), y)
    assert.NotEqual(t, byte(0), y)
    assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int8)
    assert.Equal(t, int8(0), x.Item().(int8))
    // Positive reals
    x = torch.NewTensor([]int8{1})
    y = x.Item()
    assert.Equal(t, int8(1), y)
    assert.NotEqual(t, byte(1), y)
    assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int8)
    assert.Equal(t, int8(1), x.Item().(int8))
    // Negative reals
    x = torch.NewTensor([]int8{-1})
    y = x.Item()
    assert.Equal(t, int8(-1), y)
    assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int8)
    assert.Equal(t, int8(-1), x.Item().(int8))
    // Maximum value
    x = torch.NewTensor([]int8{math.MaxInt8})
    y = x.Item()
    assert.Equal(t, int8(math.MaxInt8), y)
    assert.Equal(t, int8(math.MaxInt8), x.Item().(int8))
    // Minimum value
    x = torch.NewTensor([]int8{math.MinInt8})
    y = x.Item()
    assert.Equal(t, int8(math.MinInt8), y)
    assert.Equal(t, int8(math.MinInt8), x.Item().(int8))
}

func Test_Tensor_ItemInt16(t *testing.T) {
    // Zero
    x := torch.NewTensor([]int16{0})
    y := x.Item()
    assert.Equal(t, int16(0), y)
    assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int16)
    assert.Equal(t, int16(0), x.Item().(int16))
    // Positive reals
    x = torch.NewTensor([]int16{1})
    y = x.Item()
    assert.Equal(t, int16(1), y)
    assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int16)
    assert.Equal(t, int16(1), x.Item().(int16))
    // Negative reals
    x = torch.NewTensor([]int16{-1})
    y = x.Item()
    assert.Equal(t, int16(-1), y)
    assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int16)
    assert.Equal(t, int16(-1), x.Item().(int16))
    // Maximum value
    x = torch.NewTensor([]int16{math.MaxInt16})
    y = x.Item()
    assert.Equal(t, int16(math.MaxInt16), y)
    assert.Equal(t, int16(math.MaxInt16), x.Item().(int16))
    // Minimum value
    x = torch.NewTensor([]int16{math.MinInt16})
    y = x.Item()
    assert.Equal(t, int16(math.MinInt16), y)
    assert.Equal(t, int16(math.MinInt16), x.Item().(int16))
}

func Test_Tensor_ItemInt32(t *testing.T) {
    // Zero
    x := torch.NewTensor([]int32{0})
    y := x.Item()
    assert.Equal(t, int32(0), y)
    assert.NotEqual(t, int64(0), y)
    assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int32)
    assert.Equal(t, int32(0), x.Item().(int32))
    // Positive reals
    x = torch.NewTensor([]int32{1})
    y = x.Item()
    assert.Equal(t, int32(1), y)
    assert.NotEqual(t, int64(1), y)
    assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int32)
    assert.Equal(t, int32(1), x.Item().(int32))
    // Negative reals
    x = torch.NewTensor([]int32{-1})
    y = x.Item()
    assert.Equal(t, int32(-1), y)
    assert.NotEqual(t, int64(-1), y)
    assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int32)
    assert.Equal(t, int32(-1), x.Item().(int32))
    // Maximum value
    x = torch.NewTensor([]int32{math.MaxInt32})
    y = x.Item()
    assert.Equal(t, int32(math.MaxInt32), y)
    assert.Equal(t, int32(math.MaxInt32), x.Item().(int32))
    // Minimum value
    x = torch.NewTensor([]int32{math.MinInt32})
    y = x.Item()
    assert.Equal(t, int32(math.MinInt32), y)
    assert.Equal(t, int32(math.MinInt32), x.Item().(int32))
}

func Test_Tensor_ItemInt64(t *testing.T) {
    // Zero
    x := torch.NewTensor([]int64{0})
    y := x.Item()
    assert.Equal(t, int64(0), y)
    assert.NotEqual(t, int32(0), y)
    assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int64)
    assert.Equal(t, int64(0), x.Item().(int64))
    // Positive reals
    x = torch.NewTensor([]int64{1})
    y = x.Item()
    assert.Equal(t, int64(1), y)
    assert.NotEqual(t, int32(1), y)
    assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int64)
    assert.Equal(t, int64(1), x.Item().(int64))
    // Negative reals
    x = torch.NewTensor([]int64{-1})
    y = x.Item()
    assert.Equal(t, int64(-1), y)
    assert.NotEqual(t, int32(-1), y)
    assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Int64)
    assert.Equal(t, int64(-1), x.Item().(int64))
    // Maximum value
    x = torch.NewTensor([]int64{math.MaxInt64})
    y = x.Item()
    assert.Equal(t, int64(math.MaxInt64), y)
    assert.Equal(t, int64(math.MaxInt64), x.Item().(int64))
    // Minimum value
    x = torch.NewTensor([]int64{math.MinInt64})
    y = x.Item()
    assert.Equal(t, int64(math.MinInt64), y)
    assert.Equal(t, int64(math.MinInt64), x.Item().(int64))
}

func Test_Tensor_ItemFloat16(t *testing.T) {
    // Zero
    x := torch.NewTensor([]uint16{float16.Fromfloat32(0).Bits()})
    y := x.Item()
    assert.Equal(t, float32(0), y)
    assert.Equal(t, float32(0), y.(float32))
    // Positive reals
    x = torch.NewTensor([]uint16{float16.Fromfloat32(1).Bits()})
    y = x.Item()
    assert.Equal(t, float32(1), y)
    assert.Equal(t, float32(1), y.(float32))
    // Negative reals
    x = torch.NewTensor([]uint16{float16.Fromfloat32(-1).Bits()})
    y = x.Item()
    assert.Equal(t, float32(-1), y)
    assert.Equal(t, float32(-1), y.(float32))
    // Positive rational
    x = torch.NewTensor([]uint16{float16.Fromfloat32(0.5).Bits()})
    y = x.Item()
    assert.Equal(t, float32(0.5), y)
    assert.Equal(t, float32(0.5), y.(float32))
    // Negative rational
    x = torch.NewTensor([]uint16{float16.Fromfloat32(-0.5).Bits()})
    y = x.Item()
    assert.Equal(t, float32(-0.5), y)
    assert.Equal(t, float32(-0.5), y.(float32))
    // Maximum value
    x = torch.NewTensor([]uint16{float16.Fromfloat32(65504).Bits()})
    y = x.Item()
    assert.Equal(t, float32(65504), y)
    // Minimum value
    x = torch.NewTensor([]uint16{float16.Fromfloat32(-65504).Bits()})
    y = x.Item()
    assert.Equal(t, float32(-65504), y)
}

func Test_Tensor_ItemFloat32(t *testing.T) {
    // Zero
    x := torch.NewTensor([]float32{0.0})
    y := x.Item()
    assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Float32)
    assert.Equal(t, float32(0.0), y)
    assert.Equal(t, float32(0.0), y.(float32))
    // Positive reals
    x = torch.NewTensor([]float32{1.0})
    y = x.Item()
    assert.Equal(t, float32(1.0), y)
    assert.Equal(t, float32(1.0), y.(float32))
    // Negative reals
    x = torch.NewTensor([]float32{-1.0})
    y = x.Item()
    assert.Equal(t, float32(-1.0), y)
    assert.Equal(t, float32(-1.0), y.(float32))
    // Positive rational
    x = torch.NewTensor([]float32{0.5})
    y = x.Item()
    assert.Equal(t, float32(0.5), y)
    assert.Equal(t, float32(0.5), y.(float32))
    // Negative rational
    x = torch.NewTensor([]float32{-0.5})
    y = x.Item()
    assert.Equal(t, float32(-0.5), y)
    assert.Equal(t, float32(-0.5), y.(float32))
    // https://learn.microsoft.com/en-us/cpp/cpp/floating-limits?view=msvc-170
    // Maximum value
    x = torch.NewTensor([]float32{3.402823466e+38})
    y = x.Item()
    assert.Equal(t, float32(3.402823466e+38), y)
    // Minimum value
    x = torch.NewTensor([]float32{1.175494351e-38})
    y = x.Item()
    assert.Equal(t, float32(1.175494351e-38), y)
}

func Test_Tensor_ItemFloat64(t *testing.T) {
    // Zero
    x := torch.NewTensor([]float64{0.0})
    y := x.Item()
    assert.Equal(t, reflect.TypeOf(y).Kind(), reflect.Float64)
    assert.Equal(t, float64(0), y)
    assert.Equal(t, float64(0.0), y.(float64))
    // Positive reals
    x = torch.NewTensor([]float64{1.0})
    y = x.Item()
    assert.Equal(t, float64(1.0), y)
    assert.Equal(t, float64(1.0), y.(float64))
    // Negative reals
    x = torch.NewTensor([]float64{-1.0})
    y = x.Item()
    assert.Equal(t, float64(-1.0), y)
    assert.Equal(t, float64(-1.0), y.(float64))
    // Positive reals
    x = torch.NewTensor([]float64{0.5})
    y = x.Item()
    assert.Equal(t, float64(0.5), y)
    assert.Equal(t, float64(0.5), y.(float64))
    // Negative reals
    x = torch.NewTensor([]float64{-0.5})
    y = x.Item()
    assert.Equal(t, float64(-0.5), y)
    assert.Equal(t, float64(-0.5), y.(float64))
    // https://learn.microsoft.com/en-us/cpp/cpp/floating-limits?view=msvc-170
    // Maximum value
    x = torch.NewTensor([]float64{1.7976931348623158e+308})
    y = x.Item()
    assert.Equal(t, float64(1.7976931348623158e+308), y)
    // Minimum value
    x = torch.NewTensor([]float64{2.2250738585072014e-308})
    y = x.Item()
    assert.Equal(t, float64(2.2250738585072014e-308), y)
}

func Test_Tensor_ItemPanicsOnVectorFloatData(t *testing.T) {
    x := torch.NewTensor([]float32{1, 1})
    assert.PanicsWithValue(t, "a Tensor with 2 elements cannot be converted to Scalar", func() {
        x.Item()
    })
}

func Test_Tensor_ItemPanicsOnEmptyFloatData(t *testing.T) {
    x := torch.NewTensor([]float32{})
    assert.PanicsWithValue(t, "a Tensor with 0 elements cannot be converted to Scalar", func() {
        x.Item()
    })
}

func Test_Tensor_ItemPanicsOnVectorIntData(t *testing.T) {
    x := torch.NewTensor([]int32{1, 1})
    assert.PanicsWithValue(t, "a Tensor with 2 elements cannot be converted to Scalar", func() {
        x.Item()
    })
}

func Test_Tensor_ItemPanicsOnEmptyIntData(t *testing.T) {
    x := torch.NewTensor([]int32{})
    assert.PanicsWithValue(t, "a Tensor with 0 elements cannot be converted to Scalar", func() {
        x.Item()
    })
}

func Test_Tensor_ItemPanicsOnUnsuportedBFloat16Dtype(t *testing.T) {
    x := torch.NewTensor([]uint16{1})
    x = x.CastTo(torch.BFloat16)
    assert.PanicsWithValue(t, fmt.Sprintf("Dtype %d is not supported by Item", torch.BFloat16), func() {
        x.Item()
    })
}
