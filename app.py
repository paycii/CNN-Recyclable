from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from keras.utils import load_img, img_to_array
from flask import jsonify
import base64
from io import BytesIO

app = Flask(__name__)
model = tf.keras.models.load_model('./model/best_model.h5')

@app.route('/',methods= ['GET'])
def home():
    return render_template('home.html')
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded images
    images = request.files.getlist('images')
    # Load the trained model
    model = tf.keras.models.load_model('./model/best_model.h5')
    # Initialize an empty list for the predictions
    predictions = []
    # Loop through the images
    for image in images:

        # Save the image to a temporary file
        image.save('temp.jpg')

        #Encode to use it later
        with open('temp.jpg', 'rb') as image_file:
            # Encode the image data
            image_data = image_file.read()
            base64_string = base64.b64encode(image_data).decode('utf-8')
        # Read the image file

        image = load_img('temp.jpg', target_size=(150, 150))

        # Convert the image to a NumPy array
        image_array = img_to_array(image)

        # Add an extra dimension to the image array
        image_array = np.expand_dims(image_array, axis=0)

        # Get the prediction
        prediction = model.predict(image_array)[0]

        # Test the prediction
        if prediction > 0.5:
            prediction_text = 'Recyclable'
        else:
            prediction_text = 'Not recyclable'
        predictions.append({
            'prediction': prediction_text,
            'base64_string': base64_string,
        })
        print(predictions)
        #return the list in JSON
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)
