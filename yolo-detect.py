import cv2
import time
import smtplib
import requests
import numpy as np
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from shapely.geometry import Polygon
from ultralytics import YOLO


try:
    model = YOLO('yolov8n.pt')  # Replace with 'best.pt' if using custom model
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    exit()

sender_email = os.getenv("SENDER_EMAIL")
sender_password = os.getenv("SENDER_PASSWORD")
receiver_email = ["example1@gamil.com"]

def get_location():
    """Gets approximate location based on IP."""
    try:
        ip = requests.get('https://api64.ipify.org?format=json').json()['ip']
        location = requests.get(f'http://ip-api.com/json/{ip}').json()
        if location['status'] == 'success':
            return f"Latitude: {location['lat']}, Longitude: {location['lon']}, City: {location['city']}, Country: {location['country']}"
        else:
            return "Location not found."
    except Exception as e:
        print(f"Error getting location: {e}")
        return "Error getting location."

def send_email_with_screenshot(screenshot_path, location):
    """Sends an email with screenshot and location."""
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = ", ".join(receiver_email)
    msg['Subject'] = "Accident Detected - Screenshot and Location"

    body = f"An accident has been detected.\nLocation: {location}\n\nSee attached screenshot."
    msg.attach(MIMEText(body, 'plain'))

    with open(screenshot_path, "rb") as file:
        img = MIMEImage(file.read())
        img.add_header('Content-Disposition', 'attachment', filename="accident_screenshot.png")
        msg.attach(img)

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender_email, sender_password)
        for email in receiver_email:
            server.sendmail(sender_email, email, msg.as_string())
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")

def process_video(video_path, output_path):
    """Runs the accident detection on a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    accident_occurred = False
    accident_start = None
    screenshot_taken = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        current_time = time.time()
        detected_now = False

        if len(results[0].boxes) >= 2:
            for i in range(len(results[0].boxes)):
                for j in range(i + 1, len(results[0].boxes)):
                    obj1 = results[0].boxes[i]
                    obj2 = results[0].boxes[j]
                    cls1 = model.names[int(obj1.cls)]
                    cls2 = model.names[int(obj2.cls)]

                    if ('person' in [cls1, cls2]) and (cls1 in ['car', 'motorcycle', 'truck', 'bus'] or cls2 in ['car', 'motorcycle', 'truck', 'bus']):
                        x1_1, y1_1, x2_1, y2_1 = obj1.xyxy[0]
                        x1_2, y1_2, x2_2, y2_2 = obj2.xyxy[0]

                        poly1 = Polygon([(x1_1, y1_1), (x2_1, y1_1), (x2_1, y2_1), (x1_1, y2_1)])
                        poly2 = Polygon([(x1_2, y1_2), (x2_2, y1_2), (x2_2, y2_2), (x1_2, y2_2)])

                        iou = poly1.intersection(poly2).area / poly1.union(poly2).area
                        if iou > 0.37:
                            detected_now = True
                            if not accident_occurred:
                                accident_occurred = True
                                accident_start = current_time
                                screenshot_taken = False

        if accident_occurred and not detected_now:
            if current_time - accident_start > 5:
                accident_occurred = False
                screenshot_taken = False

        if detected_now and not screenshot_taken:
            screenshot_path = "accident_screenshot.png"
            cv2.imwrite(screenshot_path, frame)
            print(f"Accident detected. Screenshot saved at {screenshot_path}.")
            location = get_location()
            send_email_with_screenshot(screenshot_path, location)
            screenshot_taken = True

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf.item()
            cls = int(box.cls)
            label = f'{model.names[cls]} {conf:.2f}'
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        text = 'Accident Occurred!' if accident_occurred else 'No Accident Detected'
        color = (0, 0, 255) if accident_occurred else (0, 255, 0)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
