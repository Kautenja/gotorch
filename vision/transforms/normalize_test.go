// test cases for normalize.go
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

package vision_transforms_test

import (
	"testing"
	"github.com/stretchr/testify/assert"
	"github.com/Kautenja/gotorch"
	"github.com/Kautenja/gotorch/vision/transforms"
)

func TestNormalizeTransform(t *testing.T) {
	a := assert.New(t)
	trans := vision_transforms.Normalize([]float32{10.0}, []float32{2.3})
	t1 := torch.NewTensor([]float32{10.2, 11.3, 9.2})
	t2 := trans.Forward(t1)
	expected := torch.NewTensor([]float32{
		(10.2 - 10.0) / 2.3,
		(11.3 - 10.0) / 2.3,
		(9.2 - 10.0) / 2.3})
	a.True(torch.AllClose(t2, expected, 1e-5, 1e-8))
}

func TestNormalizeTransform3D(t *testing.T) {
	a := assert.New(t)
	trans := vision_transforms.Normalize([]float32{1.0, 2.0, 3.0}, []float32{2.3, 2.4, 2.5})
	// an image in torch should be a 3D tensor with CHW format
	t1 := torch.NewTensor([][][]float32{{{10.2}}, {{11.3}}, {{9.2}}})
	t2 := trans.Forward(t1)
	expected := torch.NewTensor([][][]float32{
		{{(10.2 - 1.0) / 2.3}},
		{{(11.3 - 2.0) / 2.4}},
		{{(9.2 - 3.0) / 2.5}}})
	a.True(torch.AllClose(t2, expected, 1e-5, 1e-8))
}

func TestNormalizeTransformPanicsOnEmptyMean(t *testing.T) {
	assert.PanicsWithValue(t, "len(mean) should be greater than 0", func() {
		vision_transforms.Normalize([]float32{}, []float32{1.0})
	})
}

func TestNormalizeTransformPanicsOnEmptyStddev(t *testing.T) {
	assert.PanicsWithValue(t, "len(stddev) should be greater than 0", func() {
		vision_transforms.Normalize([]float32{1.0}, []float32{})
	})
}

func TestNormalizeTransformPanicsOnZerosInStddev(t *testing.T) {
	assert.PanicsWithValue(t, "stddev contains zeros (pre-emptive divide-by-zero error)", func() {
		vision_transforms.Normalize([]float32{1.0, 1.0, 1.0}, []float32{1.0, 0.0, 1.0})
	})
}

func TestNormalizeTransformPanicsOnShapeInequality(t *testing.T) {
	assert.PanicsWithValue(t, "len(mean)=2 and len(stddev)=3 should be the same", func() {
		vision_transforms.Normalize([]float32{1.0, 1.0}, []float32{1.0, 0.0, 1.0})
	})
}
