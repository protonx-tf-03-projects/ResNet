from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, ReLU
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
import tensorflow as tf
def bn_rl(x):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x

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

def bn_rl_conv(x, f, k=1, s=1, p='same'):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(f, k, strides=s, padding=p)(x)
    return x
    
def stage1(x):
    x_input = Input(x)
    x = conv_bn_rl(x, 64, 7, 2)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    return x

def identity_block(x, f):
    x = conv_bn_rl(x, f)
    x = conv_bn_rl(x, f, 3)
    x = Conv2D(4*f, 1)(x)
    x = BatchNormalization()(x)
    x = Add()([x, x])
    output = ReLU()(x)
    return output
  
def conv_block(x, f, s):
    x = conv_bn_rl(x, f)
    x = conv_bn_rl(x, f, 3, s)
    x = Conv2D(4*f, 1)(x)
    x = BatchNormalization()(x)
    shortcut = Conv2D(4*f, 1, strides=s)(x)
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

def resnet34(input_shape, classes):
    inputs = Input(input_shape)
    x = stage1(input)
    x = resnet_block(x, 64, 2, 1)
    x = resnet_block(x, 128, 2)
    x = resnet_block(x, 256, 2)
    x = resnet_block(x, 512, 2)
    x = AveragePooling2D()(x)
    output = Dense(classes, activation='softmax')(x)
    model = Model(inputs, output)
    return model

model = resnet34((224,224,3), 2)