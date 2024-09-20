import cv2
from ultralytics import YOLO

# Function to determine if a point is on the line
def is_point_on_line(point, line_start, line_end):
    x, y = point
    lx1, ly1 = line_start
    lx2, ly2 = line_end

    # Check if the point is within the bounding box defined by the line segment
    return (min(lx1, lx2) <= x <= max(lx1, lx2) and
            min(ly1, ly2) <= y <= max(ly1, ly2))

# Function to check if the bounding box crosses the defined line
def check_line_crossing(bbox, line_start, line_end):
    # Unpack the bounding box
    x1, y1, x2, y2 = bbox

    # Get the line segment coordinates
    lx1, ly1 = line_start
    lx2, ly2 = line_end

    # Check for intersection
    line_vector = (lx2 - lx1, ly2 - ly1)
    box_corners = [(x1, y1), (x2, y2)]

    for corner in box_corners:
        if is_point_on_line(corner, line_start, line_end):
            return True

    return False

# Load the pre-trained YOLOv8 model
model = YOLO('models/yolov8n.pt')

# Path to your video
video_path = 'static/videos/traffic2.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize variables
vehicle_count = 0
line_start = (400, 600)  # Starting point of the line (x1, y1)
line_end = (1050, 200)    # Ending point of the line (x2, y2)
crossed_ids = set()      # To track crossed vehicle IDs

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("End of video.")
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    # Draw the counting line
    cv2.line(annotated_frame, line_start, line_end, (0, 255, 0), 2)

    for detection in results[0].boxes.data.tolist():
        class_id = int(detection[5])
        conf = detection[4]
        x1, y1, x2, y2 = map(int, detection[:4])

        # Draw bounding boxes
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(annotated_frame, f'{conf:.2f}', (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Count vehicles passing the line
        if class_id == 2 and conf > 0.65:  # Adjust class_id and confidence threshold

            # Check if the vehicle's bounding box crosses the line
            if check_line_crossing((x1, y1, x2, y2), line_start, line_end) and (class_id not in crossed_ids):
                vehicle_count += 1
                crossed_ids.add(class_id)  # Add the ID to the set to avoid recounting
            elif not check_line_crossing((x1, y1, x2, y2), line_start, line_end):
                crossed_ids.discard(class_id)  # Remove from the set when the vehicle is below the line

    # Display the vehicle count
    cv2.putText(annotated_frame, f'Vehicle Count: {vehicle_count}', 
                (int(frame.shape[1] * 0.75), 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Resize and display the frame
    resized_frame = cv2.resize(annotated_frame, (640, 480))  # Resize to a manageable size
    cv2.imshow('YOLOv8 Detection', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
