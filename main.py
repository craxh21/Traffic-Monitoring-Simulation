from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

# Set the upload folder
UPLOAD_FOLDER = 'static/videos'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle video upload
        if 'video' in request.files:
            video_file = request.files['video']
            if video_file.filename != '':
                video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_file.filename)
                video_file.save(video_path)
                return redirect(url_for('index'))

    # List all videos in the /static/videos directory
    videos = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('index.html', videos=videos)

if __name__ == '__main__':
    app.run(debug=True)
