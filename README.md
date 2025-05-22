# 🧠 Person Identification System

A real-time **Person Recognition System** built using **YOLOv11** trained on a **custom dataset**. The system detects and identifies known individuals from a webcam feed. If the person is not recognized, they are flagged as "Unknown".

![image](https://github.com/user-attachments/assets/7f17c37d-5625-4638-bca2-a4d9fa8d2edd)


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
