import json
import os
import time

import numpy as np
import redis
import settings
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

# TODO
# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(host=settings.REDIS_IP, 
                 port=settings.REDIS_PORT, 
                 db=settings.REDIS_DB_ID)

# TODO
# Load your ML model and assign to variable `model`
# See https://drive.google.com/file/d/1ADuBSE4z2ZVIdn66YDSwxKv-58U7WEOn/view?usp=sharing
# for more information about how to use this model.
model = ResNet50(include_top=True, weights="imagenet")


def predict(image_name):
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    class_name = None
    pred_probability = None

    # Load image
    img_path = image.load_img(os.path.join(settings.UPLOAD_FOLDER, image_name), target_size=(224, 224))
    
    # Convert to array
    img_array = image.img_to_array(img_path)

    # Expand dimensions to match the model's expected input (batch of images)
    img_array_batch = np.expand_dims(img_array, axis=0)

    # Scale the input image to the range used in the trained network
    img_array_batch = preprocess_input(img_array_batch)

    # Get predictions from the model using batch of images
    predictions = model.predict(img_array_batch)

    # Decode predictions
    pred = decode_predictions(predictions, top=1)

    class_name = pred[0][0][1]
    pred_probability = round(pred[0][0][2], 4)

    return class_name, pred_probability


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:
        # Inside this loop you should add the code to:
        #   1. Take a new job from Redis
        queue = db.brpop(settings.REDIS_QUEUE)[1]
        queue = json.loads(queue.decode("utf-8"))
 
        #   1.1 Get ID and image name from the job
        job_id = queue["id"]
        job_image_name = queue["image"]

        #   2. Run your ML model on the given data
        class_name, pred_probability = predict(job_image_name)

        #   3. Store model prediction in a dict with the following shape:
        #      {
        #         "prediction": str,
        #         "score": float,
        #      }
        prediction = {
            "prediction": class_name,
            "score": pred_probability
        }

        #   4. Store the results on Redis using the original job ID as the key
        #      so the API can match the results it gets to the original job
        #      sent
        db.set(job_id, json.dumps(prediction))

        # Hint: You should be able to successfully implement the communication
        #       code with Redis making use of functions `brpop()` and `set()`.

        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
