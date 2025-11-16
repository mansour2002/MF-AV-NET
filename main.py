"""
Artery Vein Network (AV-Net) training script.

This script contains the code to train AV-Net using command-line arguments for hyperparameters.
It demonstrates a 3-channel input model. Minor modifications are needed for other input types.
"""

import os
import argparse
from pathlib import Path

import pandas as pd

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from model import avnet_model
from custom_utils import load_multichannel_image, iou_loss_MC # Ensure iou_loss_MC is correctly imported

def configure_gpus():
    """Configures TensorFlow to use available GPUs with memory growth."""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print(f"Num GPUs Available: {len(gpus)}")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def main(args):
    """
    Main function to load data, build, compile, and train the AV-Net model.

    Args:
        args: An argparse.Namespace object containing parsed command-line arguments.
    """
    configure_gpus()

    # Model Hyperparameters
    IMAGE_HEIGHT = args.image_height
    IMAGE_WIDTH = args.image_width
    N_CHANNELS = args.n_channels
    LEARNING_RATE = args.learning_rate
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    TRAIN_SPLIT_RATIO = args.train_split_ratio
    DATASET_BASE_PATH = Path(args.dataset_base_path)
    RESULTS_DIR = Path(args.results_dir)
    TRAIN_CSV = args.train_csv

    # Load the AVNet model
    model = avnet_model(
        blocks=[6, 12, 24, 16],
        height=IMAGE_HEIGHT,
        width=IMAGE_WIDTH,
        n_channels=N_CHANNELS
    )
    model.summary()

    # Import DenseNet121 pre-trained weights
    densenet_model = DenseNet121(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    )

    weights = [layer.get_weights() for layer in densenet_model.layers[5:427]]
    for layer, weight in zip(model.layers[5:427], weights):
        layer.set_weights(weight)

    # Compile the Model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=iou_loss_MC, # Using the specified IoU loss
        metrics=["acc"]
    )

    # Setup directories for results
    os.makedirs(RESULTS_DIR, exist_ok=True)

    model_file_format = os.path.join(RESULTS_DIR, "avnet_model.{epoch:03d}.hdf5")
    checkpointer = ModelCheckpoint(
        filepath=model_file_format,
        monitor='val_loss',
        save_weights_only=True,
        save_best_only=True,
        verbose=1
    )

    # Import the dataset
    df = pd.read_csv(TRAIN_CSV)

    # Split data into training and validation sets
    n_total_samples = len(df)
    n_train_samples = int(n_total_samples * TRAIN_SPLIT_RATIO)

    # Using .iloc for splitting, and .sample(frac=1) for shuffling
    train_df = df.iloc[:n_train_samples].sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = df.iloc[n_train_samples:].sample(frac=1, random_state=42).reset_index(drop=True)


    # Define paths for input and ground truth data
    # Adjust these paths based on your actual dataset directory structure
    path_oct = DATASET_BASE_PATH / "oct"
    path_octa = DATASET_BASE_PATH / "octa"
    path_gt = DATASET_BASE_PATH / "gt"

    input_paths = [path_oct, path_octa]
    ground_truth_paths = [path_gt]

    # Load training data
    im_shape = (IMAGE_HEIGHT, IMAGE_WIDTH)
    X_train = load_multichannel_image(train_df, im_shape, input_paths, 0)
    y_train = load_multichannel_image(train_df, im_shape, ground_truth_paths, 0)

    # Load validation data
    X_valid = load_multichannel_image(val_df, im_shape, input_paths, 0)
    y_valid = load_multichannel_image(val_df, im_shape, ground_truth_paths, 0)

    # Train the model
    print("\n--- Starting Model Training ---")
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        steps_per_epoch=n_train_samples // BATCH_SIZE, # Use n_train_samples here
        validation_data=(X_valid, y_valid),
        callbacks=[checkpointer],
        epochs=EPOCHS
    )
    print("\n--- Model Training Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the AV-Net model for artery-vein segmentation.")

    # Hyperparameters
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate for the Adam optimizer.")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to train the model.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training.")
    parser.add_argument("--image_height", type=int, default=320,
                        help="Height of the input images.")
    parser.add_argument("--image_width", type=int, default=320,
                        help="Width of the input images.")
    parser.add_argument("--n_channels", type=int, default=3,
                        help="Number of input channels for the model (e.g., 3 for OCTA+OCT).")

    # Paths and Data Splitting
    parser.add_argument("--dataset_base_path", type=str, default="dataset",
                        help="Base path to the dataset directory (e.g., 'dataset/oct', 'dataset/octa').")
    parser.add_argument("--results_dir", type=str, default="Results",
                        help="Directory to save training results and model checkpoints.")
    parser.add_argument("--train_csv", type=str, default="train.csv",
                        help="Path to the CSV file containing training image names.")
    parser.add_argument("--train_split_ratio", type=float, default=0.8,
                        help="Fraction of the dataset to use for training (e.g., 0.8 for 80%% train, 20%% validation).")

    args = parser.parse_args()
    main(args)