"""TorchVision Faster RCNN inference."""
import argparse
import logging
from tqdm import tqdm
import cv2
import numpy as np
import torch
# Even though torchvision is not used in this script, it must be imported in
# order for the TorchScript back-end to locate the torchvision C++ ops.
import torchvision as tv


# These are the MS-COCO labels for mapping integer valued outputs to strings.
# The model will output the label index + 1 (so subtract 1 from model outputs
# before indexing into this array.)
LABELS = np.array(['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'hair brush'])


logger = logging.getLogger('faster_rcnn')
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    filemode='w',
)


# Parse the command line arguments.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('model',
    type=str,
    help='The path to the model to use for inference.',
)
parser.add_argument('--threshold', '-t',
    type=float,
    help='The threshold to use for detection.',
    default=0.7,
)
parser.add_argument('--face_limit', '-f',
    type=int,
    help='The face limit to prevent excessive compute.',
)
parser.add_argument('--input_shape', '-i',
    type=int,
    nargs=2,
    help='The input shape of the image inputs in WxH format.',
    default=(128, 128),
)
parser.add_argument('--device', '-D',
    type=str,
    help='The torch device to use for inference with the model.',
    default='cpu',
)
parser.add_argument('--precision', '-P',
    type=str,
    help='The precision of the numerical system to use.',
    default='float',
    choices={'half', 'float', 'double'},
)
parser.add_argument('--capture_device', '-C',
    type=str,
    help='The ID of the camera device / path to video file to stream video from.',
    default='0',
)
parser.add_argument('--flip', '-F',
    help='Whether to horizontally flip the frames from the camera.',
    action='store_true',
)
parser.add_argument('--warmup_frames', '-W',
    type=int,
    help='The number of frames to throw away while the capture device warms up.',
    default=5,
)
parser.add_argument('--no_gui', '-ng',
    help='Whether to run the script in server mode with no GUI.',
    action='store_true',
)
parser.add_argument('--output', '-o',
    type=str,
    help='The path to write the output video to.',
)
args = parser.parse_args()


# Load the traced TorchScript model.
logger.info('Loading TorchScript model from %s', repr(args.model))
logger.info('Using accelerator: %s', args.device)
logger.info('Processing with numerical precision: %s', args.precision)
DEVICE = torch.device(args.device)
DTYPE = getattr(torch, args.precision)
with torch.autograd.inference_mode():
    model = torch.jit.load(args.model, map_location=DEVICE).eval()


# Read the first frame to get information about the screen dimensions and
# sanity check the presence of a capture device on the host machine.
capture_device = int(args.capture_device) if args.capture_device.isdigit() else args.capture_device
logger.info('Initializing frame capture from device %s', repr(capture_device))
video_input = cv2.VideoCapture(capture_device)
# Capture a single frame to ensure the device / file is healthy.
ret, frame = video_input.read()
if not ret:
    raise RuntimeError('Failed to initialize video capture')
# Throw away some frames to allow the device to warm up (hardware devices only.)
if isinstance(capture_device, int):  # OpenCV indexed device driver
    for _ in tqdm(range(args.warmup_frames), desc="Warming up camera"):
        video_input.read()


# Create the video output stream.
video_output = None
if args.output is not None:
    logger.info('Saving output frames to %s', repr(args.output))
    # define the video output. It relies on the height and width of the capture,
    # which is not known at this point in the execution
    H, W, C = frame.shape
    video_output = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (W, H))


# Run the main inference loop indefinitely.
logger.info('Starting inference loop')
progress = tqdm()
try:
    while True:
        # Fetch the next frame from the camera.
        ret, frame = video_input.read()
        if args.flip:
            frame = cv2.flip(frame, 1)
        if not ret:  # If this is a video, ret will be false when the stream ends
            break

        # Forward pass the image through the network.
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        with torch.autograd.inference_mode():
            tensor = torch.tensor(img, device=DEVICE, dtype=DTYPE).div(255.0).permute(2, 0, 1)
            # The models expect lists of CHW tensors to support arbitrary HW
            _, (predictions, ) = model([tensor])

        # Get the scores and for filtering out low-confidence detection results.
        scores = predictions['scores']
        is_object = scores > args.threshold
        # Get the scores, boxes, and labels of the most confident predictions.
        scores = scores[is_object]
        boxes = predictions['boxes'][is_object]
        labels = LABELS[predictions['labels'][is_object].sub(1).numpy()]
        labels = [f'P[{label}] = {100 * score:.2f}%' for (label, score) in zip(labels, scores.tolist())]

        # Draw the bounding boxes onto the input frame.
        frame = tv.utils.draw_bounding_boxes(
            torch.tensor(frame[..., ::-1].copy()).permute(2, 0, 1).byte(), boxes,
            labels=labels,
            width=3,
        ).permute(1, 2, 0).numpy()[..., ::-1].copy()

        if not args.no_gui: # Render the image on the display for visualization
            cv2.imshow('TorchVision Faster R-CNN Demonstration', frame)
        # Write the frame to the output buffer
        if video_output is not None:
            video_output.write(frame)

        # Update the progress bar for this iteration
        progress.update(1)

        # Detect a press to the escape key to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # detect "Q" or "Escape"
            break
except KeyboardInterrupt:  # catch keyboard interrupts from terminal and ignore
    pass


# Close the TQDM progress bar.
progress.close()
# Release the capture and storage device at the end of the execution
video_input.release()
if video_output is not None:
    video_output.release()
# Destroy all the windows
cv2.destroyAllWindows()
