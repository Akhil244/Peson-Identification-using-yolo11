# 🧠 Person Identification System

A real-time **Person Recognition System** built using **YOLOv11** trained on a **custom dataset**. The system detects and identifies known individuals from a webcam feed. If the person is not recognized, they are flagged as "Unknown".

![Demo Output](https://github.com/user-attachments/assets/fb3024fe-688a-415c-aa3d-23cb42f3cc76)

> **Fig:** Real-time object detection result using YOLOv11.

---

## 🚀 Features

- Real-time person detection and identification
- Built using **YOLOv11** (custom trained)
- Webcam integration (`--source 0`)
- Detects multiple known individuals
- Flags unrecognized persons

To Run the main file

```bash
python yolo_detect.py --model my_model.pt --source 0 --resolution 640x480
