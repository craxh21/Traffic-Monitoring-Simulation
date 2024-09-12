# main.py
from flask import Flask, request, render_template
import os
import cv2
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

# Function to detect vehicles in an image
def detect_vehicles(image_path):
    # Load YOLO model
    net = cv2.dnn.readNet('models/yolov3.weights', 'models/yolov3.cfg')
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]#Returns the indexes of the output layers that we need for detecting objects.

    # Load image
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # Pre-process the image for YOLO
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)#Converts the image into a blob (binary large object) that can be processed by the YOLO model.
    net.setInput(blob)#Sets the pre-processed image (blob) as input
    outs = net.forward(output_layers)#Runs a forward pass through the network to get the output predictions.

    # Initialize counters for vehicle types
    vehicle_counts = {
        "car": 0,
        "truck": 0,
        "motorcycle": 0,
        "bus": 0,
        "total": 0
    }

    # Process the output of YOLO
    for out in outs:
        for detection in out:# Each prediction consists of:
            scores = detection[5:]# Confidence scores for each of the 80 classes in the COCO dataset.
            class_id = np.argmax(scores)# Identifies the class with the highest score (most likely object).
            confidence = scores[class_id]# he confidence score of the detected object for the identified class

            # Filter for confidence
            if confidence > 0.5:  # Adjust threshold as needed
                # Assuming class IDs are mapped according to COCO names
                if class_id == 2:  # Car
                    vehicle_counts["car"] += 1
                elif class_id == 5:  # Bus
                    vehicle_counts["bus"] += 1
                elif class_id == 7:  # Truck
                    vehicle_counts["truck"] += 1
                elif class_id == 3:  # Motorcycle
                    vehicle_counts["motorcycle"] += 1
                
                vehicle_counts["total"] += 1

    return vehicle_counts

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    images_and_counts = []
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            
            # Run the detection function
            vehicle_counts = detect_vehicles(image_path)

            # Append the results
            images_and_counts.append((file.filename, vehicle_counts))

    return render_template('index.html', images_and_counts=images_and_counts)

if __name__ == '__main__':
    app.run(debug=True)