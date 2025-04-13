from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
import cv2
import numpy as np
from threading import Thread
import queue
import time
import os
import json
from vidgear.gears import CamGear
from datetime import datetime
import threading

app = Flask(__name__)
app.secret_key = 'crowd_monitoring_secret_key'  # For session management

# Global variables
video_thread = None
frame_queue = queue.Queue(maxsize=10)
frame_count = 0
alerts = {
    'tirumala': []
}
object_counts = {
    'tirumala': {}
}
stream = None
crowd_threshold = 50  # Default threshold for crowd alerts
registered_users = []

# Load registered users from JSON file if it exists
def load_users():
    global registered_users
    if os.path.exists('users.json'):
        try:
            with open('users.json', 'r') as f:
                registered_users = json.load(f)
        except:
            registered_users = []

# Save registered users to JSON file
def save_users():
    with open('users.json', 'w') as f:
        json.dump(registered_users, f, indent=4)

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def process_video():
    global frame_count, stream, object_counts, alerts, crowd_threshold
    
    try:
        # Initialize video stream from YouTube
        stream = CamGear(source='https://youtu.be/YzcawvDGe4Y?si=kX3xDtpo-3tjd8rv', 
                        stream_mode=True, 
                        logging=True).start()
        
        while True:
            try:
                frame = stream.read()
                if frame is None:
                    print("No frame received, restarting stream")
                    break
                
                # Update frame count
                frame_count += 1
                
                # Process every 6th frame for better performance
                if frame_count % 6 != 0:
                    continue
                
                # Resize frame for better display
                frame = cv2.resize(frame, (1020, 600))
                
                # Detecting objects
                height, width, _ = frame.shape
                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                
                # Detecting objects
                net.setInput(blob)
                outs = net.forward(output_layers)
                
                # Information to display on screen
                class_ids = []
                confidences = []
                boxes = []
                
                # Showing information on the screen
                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            # Object detected
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            
                            # Rectangle coordinates
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)
                
                # Apply non-max suppression
                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                
                # Reset object counts for this frame
                current_counts = {}
                
                # Draw boxes and count objects
                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        label = str(classes[class_ids[i]])
                        confidence = confidences[i]
                        
                        # Count objects
                        if label not in current_counts:
                            current_counts[label] = 0
                        current_counts[label] += 1
                        
                        # Draw box and label
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Update object counts for Tirumala
                object_counts['tirumala'] = current_counts
                
                # Check for crowd density and generate alerts
                total_people = current_counts.get('person', 0)
                if total_people > crowd_threshold:  # Use the global threshold
                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    alert_msg = {
                        "time": current_time,
                        "message": f"High crowd density detected: {total_people} people (threshold: {crowd_threshold})",
                        "event_details": {
                            "location": "Tirumala Temple",
                            "crowd_count": total_people,
                            "threshold": crowd_threshold,
                            "severity": "High" if total_people > crowd_threshold * 1.5 else "Medium"
                        },
                        "safety_instructions": [
                            "Please remain calm and follow the instructions of temple staff",
                            "Move towards the nearest exit in an orderly manner",
                            "Avoid pushing or rushing",
                            "If you feel unwell, inform the nearest staff member",
                            "Stay with your group and do not get separated"
                        ]
                    }
                    alerts['tirumala'].append(alert_msg)
                
                # Display object counts on frame
                y = 30
                for obj, count in current_counts.items():
                    text = f"{obj}: {count}"
                    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    y += 30
                
                # Display threshold on frame
                cv2.putText(frame, f"Threshold: {crowd_threshold}", (10, y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Put frame in queue
                if not frame_queue.full():
                    frame_queue.put(frame)
                else:
                    try:
                        frame_queue.get_nowait()  # Remove old frame
                        frame_queue.put(frame)    # Put new frame
                    except queue.Empty:
                        pass
                
                time.sleep(0.1)  # Add small delay to prevent high CPU usage
                
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error in video stream: {str(e)}")
    finally:
        if stream:
            stream.stop()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/alert_page')
def alert_page():
    # Check if user is registered
    if 'user_id' not in session:
        return redirect(url_for('crowd_alert'))
    return render_template('alert_page.html')

@app.route('/crowd_alert')
def crowd_alert():
    return render_template('crowd_alert.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.1)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_counts/<location>')
def get_counts(location):
    if location in object_counts:
        return jsonify({'success': True, 'counts': object_counts[location]})
    return jsonify({'success': False, 'error': 'Invalid location'})

@app.route('/get_alerts/<location>')
def get_alerts(location):
    if location in alerts:
        return jsonify({'success': True, 'alerts': alerts[location]})
    return jsonify({'success': False, 'error': 'Invalid location'})

@app.route('/send_warning/<location>', methods=['POST'])
def send_warning(location):
    if location in alerts:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_msg = {
            "time": current_time,
            "message": f"Manual warning sent for {location} at {current_time}",
            "event_details": {
                "location": "Tirumala Temple",
                "type": "Manual Warning",
                "severity": "Medium"
            },
            "safety_instructions": [
                "Please remain calm and follow the instructions of temple staff",
                "Move towards the nearest exit in an orderly manner",
                "Avoid pushing or rushing",
                "If you feel unwell, inform the nearest staff member",
                "Stay with your group and do not get separated"
            ]
        }
        alerts[location].append(alert_msg)
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Invalid location'})

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    global crowd_threshold
    data = request.json
    if 'threshold' in data:
        old_threshold = crowd_threshold
        crowd_threshold = int(data['threshold'])
        
        # Check current counts and generate alert if needed
        for location in object_counts:
            current_count = object_counts[location].get('person', 0)
            if current_count > crowd_threshold:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                alert_msg = {
                    "time": current_time,
                    "message": f"Alert: Current crowd ({current_count} people) exceeds new threshold ({crowd_threshold})",
                    "event_details": {
                        "location": "Tirumala Temple",
                        "crowd_count": current_count,
                        "old_threshold": old_threshold,
                        "new_threshold": crowd_threshold,
                        "severity": "High" if current_count > crowd_threshold * 1.5 else "Medium"
                    },
                    "safety_instructions": [
                        "Please remain calm and follow the instructions of temple staff",
                        "Move towards the nearest exit in an orderly manner",
                        "Avoid pushing or rushing",
                        "If you feel unwell, inform the nearest staff member",
                        "Stay with your group and do not get separated"
                    ]
                }
                alerts[location].append(alert_msg)
        
        return jsonify({'success': True, 'threshold': crowd_threshold})
    return jsonify({'success': False, 'error': 'Invalid threshold value'})

@app.route('/register_user', methods=['POST'])
def register_user():
    data = request.json
    
    # Generate a unique user ID
    user_id = len(registered_users) + 1
    
    # Add registration timestamp
    data['registered_at'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data['user_id'] = user_id
    
    # Add to registered users
    registered_users.append(data)
    
    # Save to JSON file
    save_users()
    
    # Set session
    session['user_id'] = user_id
    session['user_name'] = data.get('name', '')
    
    return jsonify({'success': True, 'user_id': user_id})

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    session.pop('user_name', None)
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Load registered users
    load_users()
    
    # Start video processing in a separate thread
    video_thread = threading.Thread(target=process_video)
    video_thread.daemon = True
    video_thread.start()
    
    # Create static directory if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Run Flask app with SSL
    app.run(host='0.0.0.0', port=3000, debug=True, ssl_context=('ssl/cert.pem', 'ssl/key.pem')) 