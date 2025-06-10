"""
Artery Vein Network (AV-Net) architecture.

This file contains the network architecture of AV-Net.
All functions and dependencies are self-contained.

"""

from tensorflow.keras.layers import (
    Input, Conv2D, ZeroPadding2D, BatchNormalization,
    UpSampling2D, Concatenate, Activation, AveragePooling2D, MaxPool2D
)
from tensorflow.keras import Model
from tensorflow.keras import backend

def conv_block(x, growth_rate, name):
    """
    Building block for the Dense block.

    Args:
        x: Input tensor.
        growth_rate: Float, growth rate at dense layers.
        name: String, block label.

    Returns:
        x2: Output tensor for the block after concatenation.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=f"{name}_0_bn")(x)
    x1 = Activation("relu", name=f"{name}_0_relu")(x1)
    
    x1 = Conv2D(filters=4 * growth_rate, kernel_size=1, use_bias=False, name=f"{name}_1_conv")(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=f"{name}_1_bn")(x1)
    x1 = Activation("relu", name=f"{name}_1_relu")(x1)
    
    x1 = Conv2D(filters=growth_rate, kernel_size=3, use_bias=False, padding="same", name=f"{name}_2_conv")(x1)
    x2 = Concatenate(axis=bn_axis, name=f"{name}_concat")([x, x1])

    return x2

def dense_block(x, blocks, name):
    """
    Densely connected blocks, as used in DenseNet.

    Args:
        x: Input tensor.
        blocks: Integer, the number of conv blocks within this dense block.
        name: String, block label.

    Returns:
        x: Output tensor for the dense block.
    """
    for i in range(blocks):
        x = conv_block(x, 32, name=f"{name}_block{i + 1}")
    return x

def transition_block(x, reduction, name):
    """
    Transition Block, reducing spatial dimensions and feature count.

    Args:
        x: Input tensor.
        reduction: Float, compression rate at transition layers (e.g., 0.5).
        name: String, block label.

    Returns:
        x_pool: Output tensor after pooling.
        x_conv: Conv2D output tensor for skip-connection with the decoder.
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=f"{name}_bn")(x)
    x1 = Activation('relu', name=f"{name}_relu")(x1)
    
    # Calculate output filters based on reduction
    filters = int(backend.int_shape(x)[bn_axis] * reduction)
    x_conv = Conv2D(filters=filters, kernel_size=1, use_bias=False, name=f"{name}_conv")(x1)
    
    x_pool = AveragePooling2D(2, strides=2, name=f"{name}_pool")(x_conv)

    return x_pool, x_conv

def decoder_block(x, x_skip, growth_rate, name):
    """
    Decoder Block, combining upsampled features with skip connections.

    Args:
        x: Input tensor 1, tensor from the previous upsampled layer.
        x_skip: Input tensor 2, tensor from the corresponding layer in the encoder (skip connection).
        growth_rate: Float, growth rate for decoder block.
        name: String, block name.

    Returns:
        x_out: Output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    
    x_concat = Concatenate(axis=bn_axis, name=f"{name}_concat")([x, x_skip])
    
    x_out = Conv2D(filters=growth_rate, kernel_size=3, use_bias=False, padding="same", name=f"{name}_0_conv")(x_concat)
    x_out = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=f"{name}_0_bn")(x_out)
    x_out = Activation("relu", name=f"{name}_0_relu")(x_out)
    
    x_out = Conv2D(filters=growth_rate, kernel_size=3, use_bias=False, padding="same", name=f"{name}_1_conv")(x_out)
    x_out = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=f"{name}_1_bn")(x_out)
    x_out = Activation("relu", name=f"{name}_1_relu")(x_out)

    return x_out

def avnet_model(blocks, height, width, n_channels):
    """
    AV-Net model is an encoder-decoder network.
    - The encoder is based on the DenseNet-121 architecture.
    - The decoder is a custom decoder.

    Args:
        blocks: List of integers, number of blocks for the four dense layers in the encoder.
        height: Integer, input image height.
        width: Integer, input image width.
        n_channels: Integer, number of input image channels.

    Returns:
        Model: Keras Model of the AV-Net network.
    """
    input_size = (height, width, n_channels)
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1
    
    inputs = Input(input_size, name="Input_Layer")
    
    # Encoder
    x = ZeroPadding2D(padding=(3, 3), name="conv1_zeropad1")(inputs)
    x = Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, padding="valid", name="conv1_conv")(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="conv1_bn")(x)
    x0 = Activation("relu", name="conv1_relu")(x) # Store for skip connection
    
    x = ZeroPadding2D(padding=(1, 1), name="conv1_zeropad2")(x0)
    x = MaxPool2D(pool_size=3, strides=2, name="pool1")(x)
    
    x = dense_block(x, blocks[0], name="conv2")
    x, x1 = transition_block(x, 0.5, name="pool2") # Store x1 for skip connection
    
    x = dense_block(x, blocks[1], name="conv3")
    x, x2 = transition_block(x, 0.5, name="pool3") # Store x2 for skip connection
    
    x = dense_block(x, blocks[2], name="conv4")
    x, x3 = transition_block(x, 0.5, name="pool4") # Store x3 for skip connection
    
    x = dense_block(x, blocks[3], name="conv5")
    
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="bn")(x)
    x = Activation("relu", name='relu')(x)
    
    # Decoder
    x = UpSampling2D(size=(2, 2))(x)
    x = decoder_block(x, x3, 256, name="decode1")
    
    x = UpSampling2D(size=(2, 2))(x)
    x = decoder_block(x, x2, 128, name="decode2")
    
    x = UpSampling2D(size=(2, 2))(x)
    x = decoder_block(x, x1, 64, name="decode3")
    
    x = UpSampling2D(size=(2, 2))(x)
    x = decoder_block(x, x0, 32, name="decode4")
    
    # Final layers
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(filters=16, kernel_size=3, strides=1, use_bias=False, padding="same", name="decoder_conv1_conv")(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="decoder_conv1_bn")(x)
    x = Activation("relu", name="decoder_conv1_relu")(x)
    
    x = Conv2D(filters=16, kernel_size=3, strides=1, use_bias=False, padding="same", name="decoder_conv2_conv")(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="decoder_conv2_bn")(x)
    x = Activation("relu", name="decoder_conv2_relu")(x)
    
    x = Conv2D(filters=1, kernel_size=1, strides=1, use_bias=True, padding="same", name="final_conv")(x)
    outputs = Activation("sigmoid", name="output_classification")(x)
    
    avnet = Model(inputs, outputs, name="AVNet") # Give the model a name
    
    return avnet