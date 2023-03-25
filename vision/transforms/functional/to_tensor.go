// GoTorch port of torchvision.transforms.functional.to_tensor.
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

package vision_transforms_functional

import (
	"fmt"
	"unsafe"
	"image"
	"image/draw"
	"github.com/Kautenja/gotorch"
)

// Convert an image.Image to a torch Tensor.
func ToTensor(frame image.Image) *torch.Tensor {
	window := frame.Bounds()
	height := int64(window.Dy())
	width := int64(window.Dx())
	var tensor *torch.Tensor
	switch typedImage := frame.(type) {
	case *image.Uniform:
		panic(fmt.Sprintf("ToTensor not implemented for image of type Uniform"))
	case *image.RGBA:  // Image is already in RGB format.
		tensor = torch.TensorFromBlob(unsafe.Pointer(&typedImage.Pix[0]), torch.Byte, []int64{height, width, 4})
	default:  // Draw the image in RGB format.
		output := image.NewRGBA(window)
		draw.Draw(output, output.Bounds(), frame, window.Min, draw.Src)
		tensor = torch.TensorFromBlob(unsafe.Pointer(&output.Pix[0]), torch.Byte, []int64{height, width, 4})
	}
	return tensor.
		CastTo(torch.Float).                 // char -> float
		Div(torch.FullLike(tensor, 255.0)).  // [0., 255.] -> [0., 1.]
		Permute(2, 0, 1).                    // HWC -> CHW
		Slice(0, 0, 3, 1)                    // RGBA -> RGB
}
