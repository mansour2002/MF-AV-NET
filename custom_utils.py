"""
Artery Vein Network (AV-Net) utility functions.

This file provides utility functions for image processing and data loading, namely:
- load_multichannel_image: Custom image reader for multi-channel inputs.
- SSIMLoss: A custom loss function based on Structural Similarity Index.

"""

import numpy as np
from skimage import io 


import tensorflow as tf
from tensorflow.keras import backend as K

def iou_coef_MC_for_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the Intersection Over Union (IoU) coefficient for a batch of images.
    This version is designed to be used within a loss function.

    Args:
        y_true: The ground truth labels (binary segmentation masks).
        y_pred: The predicted labels (binary segmentation masks, typically from a sigmoid output).

    Returns:
        A scalar tensor representing the mean IoU coefficient across the batch.
    """
    smooth = 1e-6  # A small smoothing factor to prevent division by zero.
                   
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = K.sum(y_true * y_pred, axis=[1, 2])
    union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2]) - intersection

    iou = K.mean((intersection + smooth) / (union + smooth))
    return iou

def iou_loss_MC(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the IoU (Jaccard) loss for a batch of images.
    This loss function aims to maximize the IoU coefficient during training.

    Args:
        y_true: The ground truth labels (binary segmentation masks).
        y_pred: The predicted labels (binary segmentation masks, typically from a sigmoid output).

    Returns:
        A scalar tensor representing the IoU loss, which is 1 - IoU coefficient.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    return 1.0 - iou_coef_MC_for_loss(y_true, y_pred)


def load_multichannel_image(df, im_shape, path_list, column_idx):
    """
    Custom multi-channel image reader.

    Loads images specified in the dataframe from multiple directories
    and stacks them into a single multi-channel NumPy array.

    Args:
        df: Pandas DataFrame, containing image names with extensions (e.g., "image001.png").
        im_shape: Tuple (height, width), the desired dimensions for the images.
        path_list: List of Path objects, for directories containing different input channels.
        column_idx: Integer, the specific column index in the dataframe
                    from which to read the image filenames.

    Returns:
        numpy.ndarray: A multi-dimension array of shape
                       (len(df), im_shape[0], im_shape[1], len(path_list)).
    """
    images_data = []
    num_channels = len(path_list)
    target_height, target_width = im_shape

    for _, item in df.iterrows():
        # Initialize an empty array for the current multi-channel image
        current_image = np.zeros([target_height, target_width, num_channels], dtype=np.float32)
        
        # Get the image filename from the specified column
        image_filename = item[column_idx]

        for j, img_path_dir in enumerate(path_list):
            full_image_path = img_path_dir / image_filename
            
            try:
                temp_image = io.imread(full_image_path)
                
                # Normalize image data to [0, 1]
                temp_image = temp_image.astype(np.float32) / 255.0
                
                # Handle potential grayscale images (2D) for consistent 3D slicing
                if temp_image.ndim == 2:
                    temp_image = np.expand_dims(temp_image, axis=-1)
                
                # Ensure image fits or resize if necessary, or pad
                # This assumes padding with zeros, or that images are correctly sized.
                # If resizing is needed, skimage.transform.resize would be appropriate.
                h, w = temp_image.shape[:2]
                current_image[0:h, 0:w, j] = temp_image[:, :, 0] # Assuming single channel from input images

            except FileNotFoundError:
                print(f"Warning: Image not found at {full_image_path}. Skipping or filling with zeros.")
                # You might want to log this or handle it more robustly
            except Exception as e:
                print(f"Error loading image {full_image_path}: {e}")
                # Handle other loading errors
                
        images_data.append(current_image)
    
    # Convert list of images to a single numpy array
    images_array = np.array(images_data)
    print(f'--- Images loaded ---')
    print(f'\tShape: {images_array.shape}')
    
    return images_array