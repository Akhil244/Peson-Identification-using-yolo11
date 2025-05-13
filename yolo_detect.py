import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- Argument Parsing ----------------

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model (e.g. "my_model.pt")')
parser.add_argument('--source', required=True, help='Input source: image/video file, folder, or webcam index (e.g. "0")')
parser.add_argument('--thresh', type=float, default=0.5, help='Confidence threshold (default: 0.5)')
parser.add_argument('--resolution', default=None, help='Display resolution in WxH (e.g. "640x480")')
parser.add_argument('--record', action='store_true', help='Record output video (requires resolution)')

args = parser.parse_args()

# ---------------- Setup ----------------

model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

img_ext_list = ['.jpg', '.jpeg', '.png', '.bmp']
vid_ext_list = ['.avi', '.mp4', '.mov', '.mkv']

# Check model file
if not os.path.exists(model_path):
    print('ERROR: Model file not found.')
    sys.exit(1)

# Load YOLO model
model = YOLO(model_path, task='detect')
labels = model.names

# Detect input source type
if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext.lower() in img_ext_list:
        source_type = 'image'
    elif ext.lower() in vid_ext_list:
        source_type = 'video'
    else:
        print(f'Unsupported file extension: {ext}')
        sys.exit(1)
elif img_source.isdigit():
    source_type = 'usb'
    usb_idx = int(img_source)
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(1)

# Parse resolution
resize = False
if user_res:
    try:
        resW, resH = map(int, user_res.lower().split('x'))
        resize = True
    except:
        print('Invalid resolution format. Use format WxH, e.g. 640x480')
        sys.exit(1)

# Set up recording
if record:
    if source_type not in ['video', 'usb']:
        print('Recording only works for video or webcam.')
        sys.exit(1)
    if not resize:
        print('Resolution must be set to record.')
        sys.exit(1)
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

# ---------------- Input Initialization ----------------

if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(os.path.join(img_source, '*')) if os.path.splitext(f)[1].lower() in img_ext_list]
elif source_type in ['video', 'usb']:
    cap = cv2.VideoCapture(img_source if source_type == 'video' else usb_idx)
    if resize:
        cap.set(3, resW)
        cap.set(4, resH)

# ---------------- Inference Loop ----------------

img_count = 0
frame_rate_buffer = []
fps_avg_len = 200
avg_frame_rate = 0

while True:
    t_start = time.perf_counter()

    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list):
            print("All images processed.")
            break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    else:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Video stream ended or failed.")
            break

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes

    object_count = 0
    for i in range(len(detections)):
        box = detections[i]
        xyxy = box.xyxy.cpu().numpy().squeeze().astype(int)
        conf = box.conf.item()
        cls = int(box.cls.item())
        if conf < min_thresh:
            continue
        color = (0, 255, 0)
        label = f'{labels[cls]}: {conf:.2f}'
        cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
        cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        object_count += 1

    # FPS + Count display
    if source_type in ['video', 'usb']:
        cv2.putText(frame, f'FPS: {avg_frame_rate:.2f}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
    cv2.putText(frame, f'Objects: {object_count}', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    cv2.imshow("YOLO Detection", frame)
    if record:
        recorder.write(frame)

    key = cv2.waitKey(1 if source_type in ['video', 'usb'] else 0)
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.waitKey()
    elif key == ord('p'):
        cv2.imwrite('capture.png', frame)

    # Update FPS
    t_end = time.perf_counter()
    fps = 1 / (t_end - t_start)
    frame_rate_buffer.append(fps)
    if len(frame_rate_buffer) > fps_avg_len:
        frame_rate_buffer.pop(0)
    avg_frame_rate = np.mean(frame_rate_buffer)

# ---------------- Cleanup ----------------

if source_type in ['video', 'usb']:
    cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()
print(f'Average FPS: {avg_frame_rate:.2f}')
