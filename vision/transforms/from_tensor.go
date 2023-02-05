// GoTorch port of torchvision.transforms.FromTensor
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

package vision_transforms

import (
	"image"
	"github.com/Kautenja/gotorch"
	"github.com/Kautenja/gotorch/vision/transforms/functional"
)

// A transformer that converts `torch.Tensor` to `image.Image`.
type FromTensorTransformer struct { }

// Create a new FromTensorTransformer with given parameters.
func FromTensor() *FromTensorTransformer {
	return &FromTensorTransformer{}
}

// Forward pass a tensor through the transformer to convert it to an image.
func (_ FromTensorTransformer) Forward(frame torch.Tensor) image.Image {
	return vision_transforms_functional.FromTensor(frame)
}
