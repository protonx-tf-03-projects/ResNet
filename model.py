from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, ReLU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
import tensorflow as tf

def conv_bn_rl(x, f, k=1, s=1, p='same'):
    '''
    Acronyms:
        conv: Convolution
        bn: Batch Normalization
        rl: Relu Activation Function
    Variables:
        f: Filters
        k: Kernel Size
        s: Strides
        p: Padding Type
    '''
    x = Conv2D(f, k, strides=s, padding=p)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

def stage1(input_shape):
    x_input = Input(input_shape)
    x = conv_bn_rl(x, 64, 7, 2)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    return x

def identity_block(tensor, f):
    x = conv_bn_rl(tensor, f)
    x = conv_bn_rl(x, f, 3)
    x = Conv2D(4*f, 1)(x)
    x = BatchNormalization()(x)
    x = Add()([x, tensor])
    output = ReLU()(x)
    return output
  
def conv_block(tensor, f, s):
    x = conv_bn_rl(tensor, f)
    x = conv_bn_rl(x, f, 3, s)
    x = Conv2D(4*f, 1)(x)
    x = BatchNormalization()(x)
    shortcut = Conv2D(4*f, 1, strides=s)(tensor)
    shortcut = BatchNormalization()(shortcut)
  
    x = Add()([x, shortcut])
    output = ReLU()(x)
    return output
  
def resnet_block(x, f, n, s=2):
    '''
    Variables:
        n: number of identity blocks that we want to create
    '''
    x = conv_block(x, f, s)
    for _ in range(n-1):
        x = identity_block(x, f)
    return x

def resnet50(input_shape, classes):
    input = Input(input_shape)
    x = stage1(input)
    x = resnet_block(x, 64, 3, 1)
    x = resnet_block(x, 128, 4)
    x = resnet_block(x, 256, 6)
    x = resnet_block(x, 512, 3)
    x = AveragePooling2D()(x)
    output = Dense(classes, activation='softmax')(x)
    model = Model(input, output)
    return model

def resnet101(input_shape, classes):
    input = Input(input_shape)
    x = stage1(input)
    x = resnet_block(x, 64, 3, 1)
    x = resnet_block(x, 128, 4)
    x = resnet_block(x, 256, 23)
    x = resnet_block(x, 512, 3)
    x = AveragePooling2D()(x)
    output = Dense(classes, activation='softmax')(x)
    model = Model(input, output)
    return model

def resnet152(input_shape, classes):
    input = Input(input_shape)
    x = stage1(input)
    x = resnet_block(x, 64, 3, 1)
    x = resnet_block(x, 128, 8)
    x = resnet_block(x, 256, 36)
    x = resnet_block(x, 512, 3)
    x = AveragePooling2D()(x)
    output = Dense(classes, activation='softmax')(x)
    model = Model(input, output)
    return model