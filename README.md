# Bird Classifier

This script provides a web interface for uploading bird images and classifying their species using a pre-trained TensorFlow Lite model [published by Google](https://github.com/tensorflow/tfhub.dev/blob/master/assets/docs/google/models/aiy/vision/classifier/birds_V1/1.md).

The URL for the (rudimentary) web interface is available at https://classify-bird-y5mfbrotfq-ue.a.run.app/. The site was deployed using [Google Cloud Run functions](https://cloud.google.com/functions/docs/concepts/overview), which means it may take some time to load initially due to [cold starts](https://cloud.google.com/functions/docs/concepts/execution-environment#cold-starts).

## Components
The webapp consists of the following notable files:
* `main.py` - source code that contains the entry point function for the Google Cloud Run function
* `aiy_birds_V1_labelmap.csv` - mapping of ID values to bird species
* `vision-classifier-birds-v3.tflite` - pretrained ML model
