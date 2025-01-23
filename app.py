from flask import Flask, request, render_template, redirect, url_for
import os
from PIL import Image, ImageDraw
from ultralytics import YOLO
from vehicle_classifier import classify_vehicle

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['RESULT_FOLDER'] = os.path.join('static', 'results')

# Ensure the directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load the YOLOv5 model for object detection
model = YOLO("yolov5s.pt")

# Route for the homepage (image upload)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        image = request.files.get("image")
        if image:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(image_path)
            return redirect(url_for("results", filename=image.filename))
        return "Error: No image uploaded.", 400
    return render_template("index.html")

# Route for displaying the results
@app.route("/results/<filename>")
def results(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    results_path = os.path.join(app.config['RESULT_FOLDER'], filename)

    if not os.path.exists(image_path):
        return "Error: The specified file was not found.", 404

    # Perform object detection with YOLOv5
    results = model(image_path)
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # Define the class IDs for vehicles (based on COCO dataset classes)
    vehicle_class_ids = [2, 3, 5, 7]  # 2: car, 3: motorcycle, 5: bus, 7: truck

    # Draw bounding boxes and crop vehicle images
    vehicle_images = []
    for result in results:
        for box in result.boxes:
            if int(box.cls) in vehicle_class_ids:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                cropped_image = img.crop((x1, y1, x2, y2))
                vehicle_images.append(cropped_image)

    # Save the image with bounding boxes
    img.save(results_path)

    # Classify each cropped vehicle image
    approximations = [classify_vehicle(vehicle_image) for vehicle_image in vehicle_images]

    return render_template("results.html", filename=filename, approximations=approximations)

if __name__ == "__main__":
    app.run(debug=True)
