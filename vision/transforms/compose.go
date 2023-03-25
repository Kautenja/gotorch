// GoTorch port of torchvision.transforms.Compose
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
	"github.com/Kautenja/gotorch"
)

// An abstract transformer that performs operations on tensors.
type ITransformer interface {
	Forward(*torch.Tensor) *torch.Tensor
}

// A composition of many transformers in a sequential structure.
type ComposeTransformer struct {
	Transforms []ITransformer
}

// Create a new sequential pipeline of transformers.
func Compose(transforms ...ITransformer) *ComposeTransformer {
	return &ComposeTransformer{Transforms: transforms}
}

// Pass the tensor through the sequential transformation pipeline.
func (composition ComposeTransformer) Forward(tensor *torch.Tensor) *torch.Tensor {
	for _, transform := range composition.Transforms {
		tensor = transform.Forward(tensor)
	}
	return tensor
}
