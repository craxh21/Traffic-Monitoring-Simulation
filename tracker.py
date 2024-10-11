from collections import defaultdict
import math

class Tracker:
    def __init__(self, max_distance=35, max_history=30):
        # track_history keeps track of object movement history. Each ID maps to a list of (x, y) coordinates.
        self.track_history = defaultdict(lambda: [])  # {id: [(x, y), (x, y), ...]}
        self.id_count = 0  # Unique ID counter for new objects
        self.max_distance = max_distance  # Maximum distance to associate an object with its previous position
        self.max_history = max_history  # Maximum number of past positions to store for each object

    def update(self, objects_rect):# Update the tracker with the current frame's detected objects.
        objects_bbs_ids = []  # List to store updated bounding boxes with object IDs

        for rect in objects_rect:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = rect
            
            # Calculate the center of the bounding box
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            same_object_detected = False  # Flag to check if the object matches any existing tracked object

            # Loop through existing tracked objects
            for obj_id, track in self.track_history.items():
                prev_center = track[-1]  # Get the last known center of the object
                dist = math.hypot(cx - prev_center[0], cy - prev_center[1])  # Calculate distance from previous center

                # If the distance is less than the max_distance, consider it the same object
                if dist < self.max_distance:
                    self.track_history[obj_id].append((cx, cy))  # Add new position to the object's history

                    # Ensure we keep only the last 'max_history' points for each object
                    if len(self.track_history[obj_id]) > self.max_history:
                        self.track_history[obj_id].pop(0)

                    # Append bounding box and ID to the result
                    objects_bbs_ids.append([x1, y1, x2, y2, obj_id])

                    same_object_detected = True  # Mark the object as detected
                    break  # Break the loop once a match is found

            # If no existing object matches, assign a new ID to this object
            if not same_object_detected:
                self.track_history[self.id_count].append((cx, cy))  # Create new track with the current center
                objects_bbs_ids.append([x1, y1, x2, y2, self.id_count])  # Add bounding box and new ID
                self.id_count += 1  # Increment ID counter for the next new object

        # Remove old objects that were not detected in this frame
        new_track_history = defaultdict(lambda: [])
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            new_track_history[object_id] = self.track_history[object_id]  # Keep only the detected objects

        # Update the track history with only the active objects
        self.track_history = new_track_history.copy()

        return objects_bbs_ids  # Return the updated list of bounding boxes with associated IDs
