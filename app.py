from flask import Flask, render_template, Response, jsonify
import main  # Your vehicle detection script

app = Flask(__name__)

# Home page to display video and signal status
@app.route('/')
def index():
    return render_template('index.html')

# Route for video stream
@app.route('/video_feed')
def video_feed():
    return Response(main.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# API to fetch current signal statuses
@app.route('/signal_status')
def signal_status():
    signals = {
        "region_1": "red",  # Default to red
        "region_2": "red",
        "region_3": "red",
        "region_4": "red"
    }

    # Logic to update signals based on vehicle counts
    current_counts = main.current_region_counts  # Access the current vehicle counts
    max_region_index = main.index_of_max_value(current_counts)  # Get the region with the max count

    # Update signal status based on vehicle count
    if max_region_index is not None:
        signals[f"region_{max_region_index + 1}"] = "green"  # Green for the region with the most vehicles
        for i in range(len(signals)):
            if i != max_region_index:
                signals[f"region_{i + 1}"] = "red"  # Red for others

    return jsonify(signals)

if __name__ == "__main__":
    app.run(debug=True)
