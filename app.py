from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
from ultralytics import YOLO

app = FastAPI()

# Enable CORS
origins = [
    "*",  # Adjust this to your React app URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the models
model = YOLO('./Models/yolov8n.pt')
model_license_plate = YOLO('./Models/license_plate_detector.pt')

def generate_frames():
    cap = cv2.VideoCapture(0)  # Use the webcam as the video source

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Run YOLOv8 on the captured frame to detect vehicles
        results_vehicle = model.predict(source=frame, save=False, show=False, classes=[2, 3, 5, 7])
        img = results_vehicle[0].orig_img

        # Loop through each detected vehicle
        for result in results_vehicle:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for vehicles

                # Crop the vehicle region from the frame
                vehicle_img = img[y1:y2, x1:x2]

                # Run the license plate detection model on the cropped vehicle
                license_plate_results = model_license_plate.predict(source=vehicle_img, save=False, show=False, conf=0.25)

                if license_plate_results[0].boxes:
                    for lp_box in license_plate_results[0].boxes:
                        lp_x1, lp_y1, lp_x2, lp_y2 = map(int, lp_box.xyxy[0].tolist())
                        cv2.rectangle(frame, (x1 + lp_x1, y1 + lp_y1), (x1 + lp_x2, y1 + lp_y2), (0, 0, 255), 2)  # Red box for license plates

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame to the response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get('/video_feed')
def video_feed():
    return StreamingResponse(generate_frames(), media_type='multipart/x-mixed-replace; boundary=frame')

