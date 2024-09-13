from flask import Flask, request, render_template
import os
from ultralytics import YOLO
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/images'

# Load the YOLOv8 model
model = YOLO('models/yolov8n.pt')  # Using the YOLOv8 nano model. Replace with other model if needed.

# Function to detect vehicles in an image
def detect_vehicles(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Perform inference with YOLOv8
    results = model(image)

    # Initialize counters for vehicle types
    vehicle_counts = {
        "car": 0,
        "truck": 0,
        "motorcycle": 0,
        "bus": 0,
        "total": 0
    }

    # Process the results to count vehicle types
    for result in results[0].boxes.data:  # Adjust based on YOLOv8 output format
        class_id = int(result[5])  # Class ID
        confidence = result[4]  # Confidence score

        # Filter by confidence threshold
        if confidence > 0.2:  #adjust as your need
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
