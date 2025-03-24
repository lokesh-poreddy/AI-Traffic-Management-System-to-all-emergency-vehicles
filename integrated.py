import cv2
import torch
import time
import threading
import requests
from ultralytics import YOLO
from flask import Flask, request, jsonify

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")  

# Traffic light state
traffic_light = {"red": True, "green": False}

# Sample location (Can be dynamic based on GPS)
LOCATION = {"lat": 12.9716, "long": 77.5946}

# Emergency alert API URL (Local server)
ALERT_API_URL = "http://127.0.0.1:5000/alert"

# Flask app for handling alerts
app = Flask(__name__)

# 🚨 Function to send emergency alerts (Runs in a separate thread)
def send_alert(vehicle_type):
    alert_data = {"vehicle": vehicle_type, "location": LOCATION, "priority": "high"}
    
    def alert_thread():
        try:
            response = requests.post(ALERT_API_URL, json=alert_data, timeout=5)
            response.raise_for_status()
            print(f"🚨 Alert Sent: {response.json()}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Alert Failed: {e}")
    
    threading.Thread(target=alert_thread, daemon=True).start()

# 🚦 Function to switch traffic lights
def switch_light(state):
    if state == "green" and not traffic_light["green"]:
        print("🔵 GREEN LIGHT: Emergency vehicle detected. Allow passage!")
        traffic_light["red"] = False
        traffic_light["green"] = True
    elif state == "red" and not traffic_light["red"]:
        print("🔴 RED LIGHT: No emergency vehicle detected.")
        traffic_light["red"] = True
        traffic_light["green"] = False

# 🎥 Vehicle detection function
def detect_vehicles():
    cap = cv2.VideoCapture("/Users/poreddylokeshreddy/Downloads/vid.mp4")  # Use 0 for webcam

    if not cap.isOpened():
        print("❌ ERROR: Could not open video source!")
        return

    print("📷 Video Stream Started...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ ERROR: Could not read frame!")
            break

        # YOLO Inference
        results = model.predict(frame, conf=0.3, verbose=False)
        emergency_detected = False

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])  
                confidence = float(box.conf[0])

                # 🚑 Check for emergency vehicles (Class IDs: 5 = Bus, 7 = Truck)
                if class_id in [5, 7]:
                    emergency_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Emergency Vehicle", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 🚦 Update traffic light state
        if emergency_detected:
            switch_light("green")
            send_alert("Emergency Vehicle")
        else:
            switch_light("red")

        # Display Traffic Feed
        cv2.imshow("Traffic Feed", frame)

        # 🛑 Exit on 'q' key
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("🚦 Traffic Detection Stopped!")

# 🔥 Flask API to receive alerts
@app.route('/alert', methods=['POST'])
def emergency_alert():
    data = request.json
    vehicle_type = data.get("vehicle")
    location = data.get("location")

    if not vehicle_type or not location:
        return jsonify({"error": "Missing vehicle type or location"}), 400

    print(f"🚑 ALERT RECEIVED: {vehicle_type} at {location}")
    return jsonify({"message": "Alert processed successfully!"})

# 🏃 Run Flask API in a separate thread
def run_api():
    app.run(debug=True, port=5000, use_reloader=False)

# 🎬 Main Execution
if __name__ == "__main__":
    threading.Thread(target=run_api, daemon=True).start()  
    detect_vehicles()
