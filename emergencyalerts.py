from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

# Function to send an alert to hospitals/fire stations
def send_alert(vehicle_type, location):
    alert_message = {
        "vehicle": vehicle_type,
        "location": location,
        "priority": "high"
    }
    
    # Simulate sending alert (Replace with actual API endpoint)
    emergency_response_api = "https://hospital-fire-dept-alerts.com/api/notify"
    response = requests.post(emergency_response_api, json=alert_message)

    return response.status_code

# API Endpoint to receive emergency detection alerts
@app.route('/alert', methods=['POST'])
def emergency_alert():
    data = request.json
    vehicle_type = data.get("vehicle")
    location = data.get("location")

    if not vehicle_type or not location:
        return jsonify({"error": "Missing vehicle type or location"}), 400

    status = send_alert(vehicle_type, location)
    return jsonify({"message": "Alert sent successfully!", "status": status})

if __name__ == "__main__":
    app.run(debug=True, port=5000)