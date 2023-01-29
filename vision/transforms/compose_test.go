// test cases for compose.go
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

func TestComposeIsIdentityWithNoTransformers(t *testing.T) {
    tensor := torch.Rand([]int64{1, 2, 3}, torch.NewTensorOptions())
    transformer := vision_transforms.Compose()
    assert.True(t, transformer.Forward(tensor).Equal(tensor))
}

func TestComposeAppliesTransforms(t *testing.T) {
    tensor := torch.Rand([]int64{1, 3, 4, 4}, torch.NewTensorOptions())
    transformer := vision_transforms.Compose(vision_transforms.CenterCrop(2, 2))
    cropped := tensor.Slice(2, 1, 3, 1).Slice(3, 1, 3, 1)
    assert.True(t, transformer.Forward(tensor).Equal(cropped))
}
