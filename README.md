Accident Detection System using YOLOv8 and OpenCV

This project is a real-time accident detection system built using Python, YOLOv8, and OpenCV. The goal is to automatically identify accidents (like a vehicle hitting a person) from video footage, capture a screenshot of the incident, fetch the current location using IP-based geolocation, and send an email alert with both the image and the location.


Project Overview

The system is designed to:
- Detect accidents in CCTV or any video input
- Capture a screenshot when an accident is detected
- Get the current location using IP geolocation
- Send an email alert with the screenshot and location



Technologies Used

- Python
- YOLOv8 (Ultralytics)
- OpenCV
- Shapely
- smtplib (for email)
- requests (for geolocation)



How It Works

1. The video is processed frame-by-frame using OpenCV.
2. YOLOv8 detects objects like people and vehicles.
3. If a person and a vehicle overlap beyond a threshold, it's marked as an accident.
4. A screenshot of the frame is saved.
5. The location is fetched using the public IP.
6. An email is sent with the screenshot and location details.



How to Run

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```


2. Place your video (e.g., `sample_video.mp4`) in the project directory.
3. Update your email credentials and recipients in `yolo_detect.py`.
4. Run the main script:

```bash
python main.py
```
Demo

Watch the accident detection system in action:

🔗 [Click to watch the demo video](https://drive.google.com/file/d/1Jc-iYa-MGYkh9K_sOtl_kZygT7R_zEFx/view?usp=sharing)
