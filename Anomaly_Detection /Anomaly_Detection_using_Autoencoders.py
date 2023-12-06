"""
Anomaly Detection using Autoencoders
Jinlun Zhang, 220668810, MMAI5500 A3

Main task: Analyze a short video and detect the frames
where something unusual happens.

Note: To only see the codes that load the model and utilize it without training
Please refer to the file 'mmai5500_a3_model_implement.py'
"""

from keras.models import load_model
from PIL import Image
from glob import glob
import numpy as np
import keras
from keras import layers
import os
import cv2
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# This is the relative file path used in the code,
# assuming you have the video in the current working directory
file_name = './assignment3_video.avi'
image_folder = './image_folder'
model_path = 'autoencoder_model.keras'

# Define the function to convert a input video into image frames stored in
# a local folder


def convert_video_to_images(img_folder, filename='assignment3_video.avi'):
    """
    Converts the video file (assignment3_video.avi) to JPEG images.
    Once the video has been converted to images, then this function doesn't
    need to be run again.

    Arguments
    ---------
    filename    : (string) file name (absolute or relative path) of video file.
    img_folder  : (string) folder where the video frames will be stored as
    JPEG images.
    """

    # Make the img_folder if it doesn't exist.'
    try:
        if not os.path.exists(img_folder):
            os.makedirs(img_folder)
    except OSError:
        print('Error')

    # Make sure that the abscense/prescence of path
    # separator doesn't throw an error.
    # Instantiate the video object.
    img_folder = f'{img_folder.rstrip(os.path.sep)}{os.path.sep}'
    video = cv2.VideoCapture(filename)

    # Check if the video is opened successfully
    if not video.isOpened():
        print("Error opening video file")

    i = 0

    # Get the key frames
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            im_fname = f'{img_folder}frame{i:0>4}.jpg'
            print('Captured...', im_fname)
            cv2.imwrite(im_fname, frame)
            i += 1
        else:
            break

    # close the video file
    video.release()
    cv2.destroyAllWindows()

    if i:
        print(f'Video converted\n{i} images written to {img_folder}')


# Define the function to load the extracted image files
def load_images(img_dir, im_width=60, im_height=44):
    """
    Reads, resizes, and normalizes the extracted image frames from a folder.

    The images are returned in 2 formats.
    One as a Numpy array of flattened images,
    where the images with the 3-d shape (im_width, im_height, num_channels)
    are reshaped into the 1-d shape (im_width x im_height x num_channels)),
    and a list with the images with their original number of dimensions
    suitable for display.

    Arguments
    ---------
    img_dir   : (string) the directory where the images are stored.
    im_width  : (int) The desired width of the image.
                      The default value works well.
    im_height : (int) The desired height of the image.
                      The default value works well.

    Returns
    X : (numpy.array) An array of the flattened images.
    images : (list) A list of the resized unflattened images.
    """

    images = []
    fnames = glob(f'{img_dir}{os.path.sep}frame*.jpg')
    fnames.sort()

    for fname in fnames:
        im = Image.open(fname)
        # resize the image to im_width and im_height.
        im_array = np.array(im.resize((im_width, im_height)))
        # Convert uint8 to decimal and normalize to 0 - 1.
        images.append(im_array.astype(np.float32) / 255.)
        # Close the PIL image once converted and stored.
        im.close()

    # Flatten the images to a single vector
    X = np.array(images).reshape(-1, np.prod(images[0].shape))
    return X, images


# extract the frames from the video and save the frames in a local folder
convert_video_to_images(image_folder)

# load the extracted image files from the folder created above
X, images = load_images(image_folder)


# Implement an autoencoder

# This is the size of our encoded representations
encoding_dim = 32

# This is our input image
input_img = keras.Input(shape=(X.shape[1],))
# "encoded" is the encoded representation of the input
encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(X.shape[1], activation='sigmoid')(decoded)

# This model maps an input to its reconstruction
autoencoder = keras.Model(input_img, decoded)

# configure the autoencoder
autoencoder.compile(optimizer='adam', loss='mae')

# Split the data into training and testing sets
X_train, X_test = train_test_split(X, test_size=.1, random_state=22)

# Model Training
autoencoder.fit(X_train, X_train,
                epochs=30,
                batch_size=32,
                shuffle=True,
                validation_data=(X_test, X_test))


# Evaluate the loss for each frame and plot the loss to find the threshold
# Initialize a list to store the losses
losses = []

# Loop through each frame in image
for frame in images:
    # Reshape the frame as required by the model
    frame = frame.reshape((1, -1))

    # Evaluate the loss and append it to the list
    loss = autoencoder.evaluate(frame, frame, verbose=0)
    losses.append(loss)

# Plotting the losses
plt.figure(figsize=(10, 6))
sns.lineplot(data=losses)
plt.title('Loss for Each Frame')
plt.xlabel('Frame Number')
plt.ylabel('Loss')

# Add a horizontal line at y=0.02 to indicate my subjective threshold
plt.axhline(y=0.02, color='r', linestyle='--')
plt.show()


# Save the model
autoencoder.save('autoencoder_model.keras')
print("Model saved.")


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
