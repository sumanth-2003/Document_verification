from flask import Flask, render_template, request, jsonify
from roboflow import Roboflow
import base64
from PIL import Image, ImageDraw
from io import BytesIO
# import pytesseract

app = Flask(__name__)

txtbbs = {}

def overlay_boxes(image, predictions):
    draw = ImageDraw.Draw(image)
    for prediction in predictions:
        width, height = image.size
        x_center, y_center, w, h = (
            prediction["x"],
            prediction["y"],
            prediction["width"],
            prediction["height"],
        )
        x, y = x_center - w / 2, y_center - h / 2  # Calculate top-left coordinates
        class_name = prediction["class"]

        # Set background color based on class
        class_colors = {
            "details": "blue",
            "qr": "green",
            "image": "black",
            "aadharno": "red",
            "goi": "purple",
            "emblem": "orange",
        }

        if(class_name == "aadharno" or class_name == "details"):
            txtbbs[class_name] = [x, y, x + w, y + h]
        # Draw thick filled rectangle as background
        draw.rectangle([x, y, x + w, y + h], outline=class_colors.get(class_name, "white"), width=2)

        # Draw class name on top-left corner in white
        draw.rectangle([x, y, x+50, y+20], fill=class_colors.get(class_name, "white"))
        draw.text((x, y), class_name, fill="white")
    print(txtbbs)
    return image

# def extract_text_from_image(image, bounding_boxes):
    
#     for box in bounding_boxes:
#         x, y, width, height = box
#         text_region = image.crop((x, y, x + width, y + height))
        
#         # Perform OCR on the text region
#         text = pytesseract.image_to_string(text_region)
#         extracted_text.append(text.strip())

#     return extracted_text


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    try:
        # Retrieve image data
        image_data_uri = request.json.get('image')

        # Extract base64-encoded part
        _, image_data_base64 = image_data_uri.split(',', 1)

        # Decode base64 image string
        image_bytes = base64.b64decode(image_data_base64)

        # Use BytesIO to create a stream from the image data
        image_stream = BytesIO(image_bytes)

        # Open the image using PIL
        image = Image.open(image_stream).convert('RGB')

        # Save the image to a file
        image.save("input_image.jpg")

        rf = Roboflow(api_key="RMzZna7r8BabI0Fz7SJV")
        project = rf.workspace().project("aadhardetection")
        model = project.version(4).model
        # jsonimage = model.predict("input_image.jpg", confidence=40, overlap=30).json();
        prediction_result = model.predict("input_image.jpg", confidence=40, overlap=30)

        # Get predictions from the JSON response
        predictions = prediction_result.json()["predictions"]

        # Overlay bounding boxes on the input image
        image_with_boxes = overlay_boxes(image.copy(), predictions)

        # Save the image with bounding boxes (optional)
        image_with_boxes.save("output_image.jpg")
        details_region = image.crop(txtbbs["details"])
        aadharno_region = image.crop(txtbbs["aadharno"])

        # Perform OCR on the cropped regions
        # details_text = pytesseract.image_to_string(details_region)
        # aadharno_text = pytesseract.image_to_string(aadharno_region)

        # Print the extracted text
        # print("Details:", details_text)
        # print("Aadhar Number:", aadharno_text)
        # Convert the resulting image to base64
        # buffered = BytesIO()
        # image_with_boxes.save(buffered, format="JPEG")
        # base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        # Add your Roboflow API integration logic here with the processed image data
        with open("output_image.jpg", "rb") as image_file:
            # Convert the image to base64
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        # try:
        #     with open("input_image.jpg", "rb") as image_file:
        #         image = Image.open(image_file)
        #         image.show()
        # except Exception as e:
        #     print(f"Error opening image: {e}")
        return jsonify({"roboflow_result": base64_image})
        # return jsonify({"roboflow_result": 1})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
