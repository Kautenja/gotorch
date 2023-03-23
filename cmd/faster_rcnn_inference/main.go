package main

import (
	"fmt"
	"log"
	"os"
	"image"
	png "image/png"
	_ "image/jpeg"
	_ "image/gif"
	// _ "golang.org/x/image/tiff"
	// _ "golang.org/x/image/bmp"
	// _ "golang.org/x/image/webp"
	torch "github.com/Kautenja/gotorch"
	jit "github.com/Kautenja/gotorch/jit"
	T "github.com/Kautenja/gotorch/vision/transforms/functional"
)

func main() {
	if len(os.Args) < 3 {
		log.Fatal("Usage: go run main.go <model.pt> <image.png>")
		return
	}

	coco_labels := []string{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe", "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"}

	modelPath := os.Args[1]
	imagePath := os.Args[2]

	// Disable autograd for this inference context
	torch.SetGradEnabled(false)
	device := torch.NewDevice("cpu")

	// Load the model
	model, err := jit.Load(modelPath, device)
	if err != nil {
		log.Fatal(err)
		return
	}

	// Load the image from the file-system.
	imageFile, err := os.Open(imagePath)
	defer imageFile.Close()
	if err != nil {
		log.Fatal(err)
		return
	}
	// Attempt to decode the image data
	imageData, _, err := image.Decode(imageFile)
	if err != nil {
		log.Fatal(err)
		return
	}

	// Copy the pixel data into a tensor representation on the CPU.
	tensor := T.ToTensor(imageData).CopyTo(device)
	// Forward pass the tensor and extract the predictions from the IValue.
	predictions := model.Forward([]torch.IValue{torch.NewIValue([]torch.Tensor{tensor})}).ToTuple()[1].ToList()[0].ToGenericDict()

	// Select the scores, boxes, and labels, and filter predictions with scores
	// that are above the threshold.
	scores := predictions["scores"].ToTensor()
	is_object := scores.GreaterEqual(torch.FullLike(scores, 0.7))
	scores = scores.Index(is_object)
	boxes := predictions["boxes"].ToTensor().Index(is_object)
	labels := predictions["labels"].ToTensor().Index(is_object)

	// Print the string label of the first box
	fmt.Println(coco_labels[labels.Slice(0, 0, 1, 1).Item().(int64) - 1], scores.Slice(0, 0, 1, 1).Item().(float32))

	// Convert the box tensor to a Go slice in (xmin,ymin,xmax,ymax) format.
	box := boxes.CastTo(torch.Long).ToSlice().([]int64)
	fmt.Println(box)

	// Crop out the region of interest using the bounding box
	xmin, ymin, xmax, ymax := box[0], box[1], box[2], box[3]
	tensor = tensor.Slice(1, ymin, ymax, 1)
	tensor = tensor.Slice(2, xmin, xmax, 1)

	// Write the region of interest back out as an image.
	out, err := os.Create("roi.png")
	if err != nil {
		log.Fatal(err)
		return
	}
	defer out.Close()
	png.Encode(out, T.FromTensor(tensor))
}
