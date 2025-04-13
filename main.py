import cv2
import numpy as np
from vidgear.gears import CamGear    

# Initialize video stream from YouTube
stream = CamGear(source='https://www.youtube.com/watch?v=your_video_id', stream_mode=True, logging=True).start()

# Load the pre-trained model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize object count
count = 0
object_counts = {}

while True:
    # Read frame
    frame = stream.read()
    
    # Check if frame is empty
    if frame is None:
        break
        
    count += 1
    # Process every 6th frame for better performance
    if count % 6 != 0:
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
    
    # Draw boxes and count objects
    object_counts = {}
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            
            # Count objects
            if label not in object_counts:
                object_counts[label] = 0
            object_counts[label] += 1
            
            # Draw box and label
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Display object counts on frame
    y = 30
    for obj, count in object_counts.items():
        text = f"{obj}: {count}"
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 30
    
    # Display the frame
    cv2.imshow("Object Detection", frame)
    
    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Clean up
stream.stop()
cv2.destroyAllWindows()

