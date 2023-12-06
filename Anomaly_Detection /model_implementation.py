"""
Anomaly Detection using Autoencoders
Jinlun Zhang, 220668810, MMAI5500 A3

Main task: Analyze a short video and detect the frames
where something unusual happens.

Note: This file only contains the code implementing the autoencoder.
It Loads the model, and contains a function that takes a normalized frame
and returns True or False depending on whether the frame is anomalous or not.
To train the autoencoder from scratch, please
refer to the file 'mmai5500_a3.py'
"""

from keras.models import load_model

# This is the relative file path used in the code,
# assuming you have the video in the current working directory
file_name = './assignment3_video.avi'
image_folder = './image_folder'
model_path = 'autoencoder_model.keras'

# Load the trained keras autoencoder model
autoencoder = load_model(model_path)
print("Model loaded.")


# Defines the function that predicts whether a new frame is anomalous or not
# The threshold that will flag an outlying frame is subjectively set at 0.02
# according to the plot of the MAE losses of all the video frames
def predict(frame):
    """
    Argument
    --------
    frame   : Video frame with shape == (44, 60, 3) and dtype == float.
              Also, this frame must already be normalized bewteen 0 to 1.

    Return
    anomaly : A boolean indicating whether the frame is an anomaly or not.
    ------
    """

    # Flatten the images to a single vector
    frame = frame.reshape((1, -1))
    loss = autoencoder.evaluate(frame, frame, verbose=0)
    # When the MAE loss is larger than the empirically defined threshold,
    # flag anomaly frame
    anomaly = loss >= 0.02
    return anomaly
