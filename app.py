from flask import Flask, Response, request, render_template
import cv2
import os
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO model
model = YOLO("yolov8n.pt")  # YOLOv8 nano model

# Define only ambulance class ID (COCO dataset ID)
AMBULANCE_CLASS_ID = 47  # Ambulance

# Store uploaded video path
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
video_path = None

def detect_vehicles(video_path):
    cap = cv2.VideoCapture(video_path)
    ambulance_detected = False
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Perform YOLO detection
        results = model(frame)
        new_frame = frame.copy()
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if class_id == AMBULANCE_CLASS_ID and confidence > 0.5:
                    ambulance_detected = True
                    cv2.rectangle(new_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for ambulance
                    cv2.putText(new_frame, f"Ambulance ({confidence:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # If no ambulance is detected, highlight all vehicles in blue
        if not ambulance_detected:
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(new_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for other vehicles
            # Add a big red dot on the top left corner
            cv2.circle(new_frame, (30, 30), 20, (0, 0, 255), -1)  # Red dot
        
        # Encode frame for streaming
        _, buffer = cv2.imencode('.jpg', new_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return '''
    <h2>Upload a Video for YOLOv8 Detection</h2>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="video">
        <input type="submit" value="Upload and Process">
    </form>
    '''

@app.route('/upload', methods=['POST'])
def upload_video():
    global video_path
    if 'video' not in request.files:
        return "No file uploaded!", 400

    file = request.files['video']
    if file.filename == '':
        return "No selected file!", 400

    video_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(video_path)

    return f"File uploaded successfully! <a href='/video_feed'>View Processed Video</a>"

@app.route('/video_feed')
def video_feed():
    if video_path is None:
        return "No video uploaded yet!", 400
    return Response(detect_vehicles(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
