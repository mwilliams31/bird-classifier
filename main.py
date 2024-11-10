"""
Bird Classifier Script

This script provides a web interface for uploading bird images and classifying their species using a pre-trained TensorFlow Lite model.

Usage:
    Run this script with a web server that supports Flask and Google Cloud Run functions to provide a web interface for bird image classification.
"""

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import functions_framework

from flask import render_template_string


def get_prediction(image) -> str:
    """
    Predicts the bird species from an uploaded image using a pre-trained TensorFlow Lite model.

    Args:
        image: An uploaded image file.

    Returns:
        str: The predicted bird species label.
    
    """
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="vision-classifier-birds-v3.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Load and resize image
    img_array = np.fromstring(image.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_ANYCOLOR)
    new_img = cv2.resize(img, (224, 224))   # model requires 224x224 input

    # Prepare input data
    interpreter.set_tensor(input_details[0]["index"], [new_img])

    # Run inference
    interpreter.invoke()

    # Get output data
    output_data = interpreter.get_tensor(output_details[0]["index"])

    # Get the highest scoring index
    max_index = np.argmax(output_data)

    # Load labels and get the predicted species
    labels = pd.read_csv("aiy_birds_V1_labelmap.csv")
    predicted_label = labels.loc[labels["id"] == max_index]["name"].values[0]

    return predicted_label


def render_upload_form() -> str:
    """
    Renders an HTML form to upload a bird image for classification.
    """

    return render_template_string("""                          
        <html>   
        <head>   
            <title>Bird Classifier: Upload File</title>   
        </head>   
        <body>
            <h1>Bird Classifier</h1> 
            <p>Upload a picture of a bird to classify its species:</p>                         
            <form action = "/upload" method = "POST" enctype="multipart/form-data">   
                <input type="file" name="file" />   
                <input type="submit" value="Upload">   
            </form>   
            <p>If you don't have an image available, you can download one from <a href="https://www.kaggle.com/datasets/umairshahpirzada/birds-20-species-image-classification" target="_blank">this image classification dataset.</a></ 
        </body>   
        </html>                            
""")


@functions_framework.http
def classify_bird(request):
    """
    Classifies the species of a bird from an uploaded image.
    
    Args:
        request: The HTTP request containing the uploaded image.
    
    Returns:
        str: The classification result.
    """

    # Default action: render the upload form; otherwise, classify the uploaded image.
    if request.method == "POST" and request.path == "/upload":
        predicted_species = get_prediction(request.files["file"])
        
        return render_template_string("""                       
            <html> 
            <head> 
                <title>Classification Result</title> 
            </head> 
            <body>
                <h1>Bird Classifier</h1> 
                <p>File uploaded successfully!</p> 
                <p>The predicted species of your uploaded image is <b>{{ species }}</b>.</p> 
            </body> 
            </html>
        """, species=predicted_species)

    return render_upload_form()
