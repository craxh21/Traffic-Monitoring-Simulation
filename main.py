import cv2  # Import OpenCV for computer vision tasks
import pandas as pd  # Import pandas for data manipulation
from ultralytics import YOLO
from tracker import Tracker

model = YOLO('models/yolov8s.pt')  # Using YOLO model for small objects

cap = cv2.VideoCapture('static/videos/traffic2.mp4')  # Start capturing video from the specified file

# Read the class names from the COCO dataset
with open("coco.txt", "r") as my_file:  # Open the file containing class names
    data = my_file.read()  # Read the content of the file
class_list = data.split("\n")  # Split the content into a list of class names based on newline characters

count = 0
tracker = Tracker()  # Create an instance of the Tracker class to keep track of detected objects

# Define regions as tuples of (top_left, bottom_right)
regions = [
    ((100, 100), (400, 300)),  # Define Region 1
    ((450, 100), (800, 300)),  # Region 2
    ((100, 350), (400, 550)),  # Region 3
    ((450, 350), (800, 550))    # Region 4
]

# Initialize vehicle ID tracking for each region
region_vehicle_ids = [set() for _ in range(len(regions))]  # Create a list of sets to track unique vehicle IDs in each region
current_region_counts = [0] * len(regions)  # Store the counts of vehicles in each region

# Function to calculate the region having the max vehicles
def index_of_max_value(my_list):
    if len(my_list) == 0:  # Check if the list is empty
        return None  # Return None if the list is empty
    max_index = 0  # Assume the first element is the maximum
    for i in range(1, len(my_list)):
        if my_list[i] > my_list[max_index]:  # Compare with current max value
            max_index = i  # Update max_index if a larger value is found
    return max_index

def is_inside_region(center, region):
    """Check if the center of a vehicle is inside the defined region."""
    (x1, y1), (x2, y2) = region  # Unpack the coordinates of the region
    return x1 <= center[0] <= x2 and y1 <= center[1] <= y2  # Return True if the center is within the region's bounds

def generate_frames():
    global current_region_counts  # Make sure to reference the global counts
    while True:  # Start an infinite loop to process frames
        ret, frame = cap.read()  # Read a frame from the video capture
        if not ret:  # If the frame was not read correctly (end of video)
            break  # Exit the loop

        global count
        count += 1  # Increment the frame count
        if count % 3 != 0:  # Process every third frame to reduce computational load
            continue  # Skip the current iteration if it's not a frame to process

        frame = cv2.resize(frame, (1020, 600))  # Resize the frame for consistent processing

        results = model.predict(frame)  # Use the YOLO model to predict objects in the current frame
        a = results[0].boxes.data  # Extract the bounding box data from the model's results
        px = pd.DataFrame(a).astype("float")  # Convert the bounding box data to a pandas DataFrame and ensure the data type is float

        bbox_list = []  # Initialize a list to store bounding boxes
        for index, row in px.iterrows():  # Iterate through each detected bounding box
            x1 = int(row[0])  # Get the top-left x-coordinate
            y1 = int(row[1])  # Get the top-left y-coordinate
            x2 = int(row[2])  # Get the bottom-right x-coordinate
            y2 = int(row[3])  # Get the bottom-right y-coordinate
            d = int(row[5])  # Get the class ID (index) for the detected object
            c = class_list[d]  # Get the class name corresponding to the class ID
            bbox_list.append([x1, y1, x2, y2])  # Append the bounding box coordinates to the list

        bbox_id = tracker.update(bbox_list)  # Update the tracker with the current bounding box list and get the tracked vehicle IDs

        # Reset region vehicle counts for the current frame
        current_region_counts = [0] * len(regions)  # Initialize current frame counts

        # Draw regions and count vehicles
        for i, region in enumerate(regions):
            top_left, bottom_right = region  # Unpack the coordinates of the current region
            cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2)  # Draw region rectangle

            # Count vehicles in this region
            for bbox in bbox_id:
                x3, y3, x4, y4, id = bbox  # Unpack the bounding box coordinates and vehicle ID

                # Calculate the center of the bounding box
                center_x = (x3 + x4) // 2
                center_y = (y3 + y4) // 2
                center = (center_x, center_y)

                # Check if the center is inside the region
                if is_inside_region(center, region):
                    current_region_counts[i] += 1  # Increment current frame count

            # Display count for each region based on current frame
            cv2.putText(frame, f'Region {i + 1} Count: {current_region_counts[i]}', 
                        (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        # Get the region with the maximum vehicle count
        max_region_index = index_of_max_value(current_region_counts)

        # Display the region with the max vehicle count at the top of the video
        if max_region_index is not None:
            cv2.putText(frame, f'Max Vehicles in Region {max_region_index + 1}', (50, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)  # Display the region with the max count

        # Draw bounding boxes for detected vehicles
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)  # Draw bounding box

        # Encode the frame in JPEG format to serve over the web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
