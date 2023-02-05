// test cases for jit.go
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

package jit_test

import (
	"testing"
	"github.com/stretchr/testify/assert"
	"errors"
	"fmt"
	"os"
	"runtime"
	"path/filepath"
	"strings"
	"github.com/Kautenja/gotorch"
	"github.com/Kautenja/gotorch/jit"
)

// MARK: Load

func TestJitLoadThrowsErrorOnInvalidPath(t *testing.T) {
	module, err := jit.Load("../data/nonexistent.pt", torch.NewDevice("cpu"))
	assert.Nil(t, module.T)
	assert.NotNil(t, err)
}

func TestJitLoadTracedModule(t *testing.T) {
	module, err := jit.Load("../data/trace_identity.pt", torch.NewDevice("cpu"))
	assert.Nil(t, err)
	assert.NotNil(t, module.T)
}

func TestJitLoadScriptedModule(t *testing.T) {
	module, err := jit.Load("../data/script_identity.pt", torch.NewDevice("cpu"))
	assert.Nil(t, err)
	assert.NotNil(t, module.T)
}

// This test checks that the finalizer runs when the jit module is de-allocated.
func TestJitModuleGarbageCollection(t *testing.T) {
	_, _ = jit.Load("../data/trace_identity.pt", torch.NewDevice("cpu"))
	runtime.GC()
}

// MARK: Save

func TestJitModuleSaveThrowsErrorOnInvalidPath(t *testing.T) {
	module, err := jit.Load("../data/trace_identity.pt", torch.NewDevice("cpu"))
	assert.Nil(t, err)
	assert.NotNil(t, module.T)
	save_err := module.Save("./foo/bar/baz.pt")
	assert.NotNil(t, save_err)
	assert.Equal(t,
		"File ./foo/bar/baz.pt cannot be opened.",
		strings.Split(save_err.Error(), "\n")[0])
}

func TestJitModuleSaveDoesSaveModule(t *testing.T) {
	// create and open a temporary directory to save/load files to/from
	temp_dir, err := os.MkdirTemp("/tmp/", "TestJitModuleSavesModule")
	defer os.RemoveAll(temp_dir)  // defer clean up until the end of the test
	if err != nil {
		t.Log("Failed to create temporary directory")
		t.Fail()
		return
	}
	// Load a module
	module, err := jit.Load("../data/trace_identity.pt", torch.NewDevice("cpu"))
	assert.Nil(t, err)
	assert.NotNil(t, module.T)
	// Save the module to the temporary directory
	output_path := filepath.Join(temp_dir, "foo.pt")
	save_err := module.Save(output_path)
	assert.Nil(t, save_err)
	// Check that the data exists on the filesystem
	if _, err := os.Stat(output_path); errors.Is(err, os.ErrNotExist) {
		t.Log(fmt.Sprintf("Failed to save module to filesystem: %s", err.Error()))
		t.Fail()
		return
	}
	// Load the data from the filesystem into a new module
	deserialized_module, load_err := jit.Load(output_path, torch.NewDevice("cpu"))
	assert.Nil(t, load_err)
	assert.NotNil(t, deserialized_module.T)
}

// MARK: String

func TestJitModuleString(t *testing.T) {
	module, _ := jit.Load("../data/trace_identity.pt", torch.NewDevice("cpu"))
	// It's too strict to check for total equality, just look for the expected module.
	assert.Contains(t, module.String(), "torch.nn.modules.linear.Identity")
}

// MARK: Train/IsTraining

func TestJitModuleTrain(t *testing.T) {
	module, _ := jit.Load("../data/trace_identity.pt", torch.NewDevice("cpu"))
	module.Train(false)
	assert.False(t, module.IsTraining())
	// enable
	module.Train(true)
	assert.True(t, module.IsTraining())
	// disable
	module.Train(false)
	assert.False(t, module.IsTraining())
}

func TestJitModuleTrainReturnsSelf(t *testing.T) {
	module, _ := jit.Load("../data/trace_identity.pt", torch.NewDevice("cpu"))
	assert.Equal(t, module.T, module.Train(false).T)
}

// MARK: Eval

func TestJitModuleEval(t *testing.T) {
	module, _ := jit.Load("../data/trace_identity.pt", torch.NewDevice("cpu"))
	module.Train(true)
	assert.True(t, module.IsTraining())
	// disable train mode
	module.Eval()
	assert.False(t, module.IsTraining())
	// repeat calls should not change state
	module.Eval()
	assert.False(t, module.IsTraining())
}

func TestJitModuleEvalReturnsSelf(t *testing.T) {
	module, _ := jit.Load("../data/trace_identity.pt", torch.NewDevice("cpu"))
	assert.Equal(t, module.T, module.Eval().T)
}

// // MARK: SetOptimized/IsOptimized

// func TestJitModuleSetOptimized(t *testing.T) {
//  module, _ := jit.Load("../data/trace_identity.pt", torch.NewDevice("cpu"))
//  module.SetOptimized(false)
//  assert.False(t, module.IsOptimized())
//  // enable
//  module.SetOptimized(true)
//  assert.True(t, module.IsOptimized())
//  // disable
//  module.SetOptimized(false)
//  assert.False(t, module.IsOptimized())
// }

// func TestJitModuleSetOptimizedReturnsSelf(t *testing.T) {
//  module, _ := jit.Load("../data/trace_identity.pt", torch.NewDevice("cpu"))
//  assert.Equal(t, module.T, module.SetOptimized(false).T)
// }

// MARK: CastTo

func TestJitModuleCastTo(t *testing.T) {
	module, err := jit.Load("../data/trace_linear.pt", torch.NewDevice("cpu"))
	if !assert.Nil(t, err) { return }
	module = module.CastTo(torch.Double)
	tensor := torch.Rand([]int64{1, 1}, torch.NewTensorOptions()).CastTo(torch.Double)
	ivalues := []torch.IValue{torch.NewIValue(tensor)}
	assert.NotPanics(t, func() { module.Forward(ivalues) })
	outputs := module.Forward(ivalues)
	assert.True(t, outputs.IsTensor())
	assert.Equal(t, outputs.ToTensor().Dtype(), torch.Double)
}

func TestJitModuleCastToReturnsSelf(t *testing.T) {
	module, _ := jit.Load("../data/trace_identity.pt", torch.NewDevice("cpu"))
	assert.Equal(t, module.T, module.CastTo(torch.Double).T)
}

// MARK: CopyTo

func TestJitModuleCopyTo(t *testing.T) {
	module, err := jit.Load("../data/trace_identity.pt", torch.NewDevice("cpu"))
	if !assert.Nil(t, err) { return }
	assert.NotPanics(t, func() {
		module = module.CopyTo(torch.NewDevice("cpu"))
	})
}

func TestJitModuleCopyToReturnsSelf(t *testing.T) {
	module, _ := jit.Load("../data/trace_identity.pt", torch.NewDevice("cpu"))
	assert.Equal(t, module.T, module.CopyTo(torch.NewDevice("cpu")).T)
}

// TODO: Torch_Jit_Module_Copy
// TODO: Torch_Jit_Module_DeepCopy
// TODO: Torch_Jit_Module_Clone

// MARK: Forward

func TestJitModuleForwardTracedModule(t *testing.T) {
	module, _ := jit.Load("../data/trace_identity.pt", torch.NewDevice("cpu"))
	tensor := torch.NewTensor([][]float32{{1}})
	output := module.Forward([]torch.IValue{torch.NewIValue(tensor)})
	assert.True(t, output.IsTensor())
	assert.True(t, torch.Equal(output.ToTensor(), tensor))
}

func TestJitModuleForwardTracedModuleInGoRoutine(t *testing.T) {
	module, _ := jit.Load("../data/trace_identity.pt", torch.NewDevice("cpu"))
	channel := make(chan torch.Tensor)
	go func() {
		tensor := torch.NewTensor([][]float32{{1}})
		channel <- module.Forward([]torch.IValue{torch.NewIValue(tensor)}).ToTensor()
	}()
	output := <-channel
	expected := torch.NewTensor([][]float32{{1}})
	assert.True(t, torch.Equal(output, expected))
}

func TestJitModuleForwardScriptedModule(t *testing.T) {
	module, _ := jit.Load("../data/script_identity.pt", torch.NewDevice("cpu"))
	tensor := torch.NewTensor([][]float32{{1}})
	output := module.Forward([]torch.IValue{torch.NewIValue(tensor)})
	assert.True(t, output.IsTensor())
	assert.True(t, torch.Equal(output.ToTensor(), tensor))
}

func TestJitModuleForwardScriptedModuleToInt(t *testing.T) {
	module, _ := jit.Load("../data/module_that_returns_int.pt", torch.NewDevice("cpu"))
	tensor := torch.NewTensor([][]float32{{1}})
	output := module.Forward([]torch.IValue{torch.NewIValue(tensor)})
	assert.True(t, output.IsInt())
	assert.Equal(t, 222, output.ToInt())
}

func TestJitModuleForwardScriptedModuleToDouble(t *testing.T) {
	module, _ := jit.Load("../data/module_that_returns_float.pt", torch.NewDevice("cpu"))
	tensor := torch.NewTensor([][]float32{{1}})
	output := module.Forward([]torch.IValue{torch.NewIValue(tensor)})
	assert.True(t, output.IsDouble())
	assert.Equal(t, 2.22, output.ToDouble())
}

func TestJitModuleForwardScriptedModuleToBool(t *testing.T) {
	module, _ := jit.Load("../data/module_that_returns_bool.pt", torch.NewDevice("cpu"))
	tensor := torch.NewTensor([][]float32{{1}})
	output := module.Forward([]torch.IValue{torch.NewIValue(tensor)})
	assert.True(t, output.IsBool())
	assert.True(t, output.ToBool())
}

func TestJitModuleForwardScriptedModuleToString(t *testing.T) {
	module, _ := jit.Load("../data/module_that_returns_string.pt", torch.NewDevice("cpu"))
	tensor := torch.NewTensor([][]float32{{1}})
	output := module.Forward([]torch.IValue{torch.NewIValue(tensor)})
	assert.True(t, output.IsString())
	assert.Equal(t, "foo", output.ToString())
}

func TestJitModuleForwardScriptedModuleToTensorList(t *testing.T) {
	module, _ := jit.Load("../data/module_that_returns_tensor_list.pt", torch.NewDevice("cpu"))
	tensor := torch.NewTensor([][]float32{{1}})
	output := module.Forward([]torch.IValue{torch.NewIValue(tensor)})
	assert.True(t, output.IsTensorList())
	assert.True(t, output.IsList())
	tensors := output.ToTensorList()
	assert.Equal(t, 2, len(tensors))
	assert.True(t, torch.Zeros([]int64{1}, torch.NewTensorOptions()).Equal(tensors[0]))
	assert.True(t, torch.Ones([]int64{2}, torch.NewTensorOptions()).Equal(tensors[1]))
}

func TestJitModuleForwardScriptedModuleToList(t *testing.T) {
	module, _ := jit.Load("../data/module_that_returns_list.pt", torch.NewDevice("cpu"))
	tensor := torch.NewTensor([][]float32{{1}})
	output := module.Forward([]torch.IValue{torch.NewIValue(tensor)})
	assert.False(t, output.IsTensorList())
	assert.True(t, output.IsList())
	values := output.ToList()
	assert.Equal(t, 3, len(values))
	assert.Equal(t, 6, values[0].ToInt())
	assert.Equal(t, 7, values[1].ToInt())
	assert.Equal(t, 8, values[2].ToInt())
}

func TestJitModuleForwardScriptedModuleToEmptyList(t *testing.T) {
	module, _ := jit.Load("../data/module_that_returns_empty_list.pt", torch.NewDevice("cpu"))
	tensor := torch.NewTensor([][]float32{{1}})
	output := module.Forward([]torch.IValue{torch.NewIValue(tensor)})
	assert.False(t, output.IsTensorList())
	assert.True(t, output.IsList())
	values := output.ToList()
	assert.Equal(t, 0, len(values))
}

func TestJitModuleForwardScriptedModuleToTuple(t *testing.T) {
	module, _ := jit.Load("../data/module_that_returns_tuple.pt", torch.NewDevice("cpu"))
	tensor := torch.NewTensor([][]float32{{1}})
	output := module.Forward([]torch.IValue{torch.NewIValue(tensor)})
	assert.True(t, output.IsTuple())
	values := output.ToTuple()
	assert.Equal(t, 3, len(values))
	assert.Equal(t, 9, values[0].ToInt())
	assert.Equal(t, 6, values[1].ToInt())
	assert.Equal(t, 3, values[2].ToInt())
}

func TestJitModuleForwardScriptedModuleToNone(t *testing.T) {
	module, _ := jit.Load("../data/module_that_returns_none.pt", torch.NewDevice("cpu"))
	tensor := torch.NewTensor([][]float32{{1}})
	output := module.Forward([]torch.IValue{torch.NewIValue(tensor)})
	assert.True(t, output.IsNil())
}

func TestJitModuleForwardScriptedModuleToDictFloatKeys(t *testing.T) {
	module, _ := jit.Load("../data/module_that_returns_dict_float_key.pt", torch.NewDevice("cpu"))
	tensor := torch.NewTensor([][]float32{{1}})
	output := module.Forward([]torch.IValue{torch.NewIValue(tensor)})
	assert.True(t, output.IsGenericDict())
	dict := output.ToGenericDict()
	value, ok := dict[1.23]
	assert.True(t, ok)
	assert.True(t, value.IsString())
	assert.Equal(t, "foo", value.ToString())
}

func TestJitModuleForwardScriptedModuleToDictIntKeys(t *testing.T) {
	module, _ := jit.Load("../data/module_that_returns_dict_int_key.pt", torch.NewDevice("cpu"))
	tensor := torch.NewTensor([][]float32{{1}})
	output := module.Forward([]torch.IValue{torch.NewIValue(tensor)})
	assert.True(t, output.IsGenericDict())
	dict := output.ToGenericDict()
	value, ok := dict[45]
	assert.True(t, ok)
	assert.True(t, value.IsString())
	assert.Equal(t, "foo", value.ToString())
}

func TestJitModuleForwardScriptedModuleToDictStringKeys(t *testing.T) {
	module, _ := jit.Load("../data/module_that_returns_dict_str_key.pt", torch.NewDevice("cpu"))
	tensor := torch.NewTensor([][]float32{{1}})
	output := module.Forward([]torch.IValue{torch.NewIValue(tensor)})
	assert.True(t, output.IsGenericDict())
	dict := output.ToGenericDict()
	value, ok := dict["bar"]
	assert.True(t, ok)
	assert.True(t, value.IsString())
	assert.Equal(t, "foo", value.ToString())
}

func TestJitModuleForwardScriptedModuleToDictBoolKeys(t *testing.T) {
	module, _ := jit.Load("../data/module_that_returns_dict_bool_key.pt", torch.NewDevice("cpu"))
	tensor := torch.NewTensor([][]float32{{1}})
	output := module.Forward([]torch.IValue{torch.NewIValue(tensor)})
	assert.True(t, output.IsGenericDict())
	dict := output.ToGenericDict()
	value, ok := dict[true]
	assert.True(t, ok)
	assert.True(t, value.IsString())
	assert.Equal(t, "foo", value.ToString())
}
