from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.initializers import glorot_uniform


class BasicBlock(Layer):
    """
        Implementation of the basic block

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    def __init__(self, f, stride):
        super(BasicBlock, self).__init__()

        self.expansion = 1
        self.Conv2a = Conv2D(f, kernel_size=(3, 3), strides=stride, padding='same',
                             kernel_initializer=glorot_uniform(seed=0))
        self.Bn2a = BatchNormalization(axis=3)

        self.Conv2b = Conv2D(f, kernel_size=(3, 3), strides=(1, 1), padding='same',
                             kernel_initializer=glorot_uniform(seed=0))
        self.Bn2b = BatchNormalization(axis=3)

    def call(self, input_tensor, *args, **kwargs):
        X = self.Conv2a(input_tensor, *args, **kwargs)
        X = self.Bn2a(X)
        X = Activation('relu')(X)

        X = self.Conv2b(X)
        X = self.Bn2b(X)

        return X

class BottleNeckBlock(Layer):
    """
        Implementation of the bottleneck

        Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path

        Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    def __init__(self, f, stride):
        super(BottleNeckBlock, self).__init__(name='')

        self.expansion = 4
        self.Conv2a = Conv2D(f, kernel_size= (1, 1), strides = stride, padding = 'valid', kernel_initializer = glorot_uniform(seed=0))
        self.Bn2a = BatchNormalization(axis = 3)

        self.Conv2b = Conv2D(f, kernel_size= (3, 3), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))
        self.Bn2b = BatchNormalization(axis = 3)

        self.Conv2c = Conv2D(f*4, kernel_size= (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))
        self.Bn2c = BatchNormalization(axis = 3)

    def call(self, input_tensor, *args, **kwargs):
        X = self.Conv2a(input_tensor, *args, **kwargs)
        X = self.Bn2a(X)
        X = Activation('relu')(X)

        X = self.Conv2b(X)
        X = self.Bn2b(X)
        X = Activation('relu')(X)

        X = self.Conv2c(X)
        X = self.Bn2c(X)

        return X

class ConvolutionalBlock(Layer):
    """
      Implementation of the convolutional block
      Arguments:

      X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
      f -- python list of integers, defining the number of filters in the CONV layers of the main path
      use_BottleNeck -- bool type, defining the type of block that we will use
      Returns:

      X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)

    """

    def __init__(self, filter, use_BottleNeck = False):
        super(ConvolutionalBlock, self).__init__(name='')

        if (use_BottleNeck):
            self.ConBlock = BottleNeckBlock(filter, stride = 2)
            self.expansion = 4
        else:
            self.ConBlock = BasicBlock(filter, stride = 2)
            self.expansion = 1
        self.downsample = Sequential()
        self.downsample.add(Conv2D(filter*self.expansion, kernel_size= (1, 1), strides = 2, padding = 'valid'))
        self.downsample.add(BatchNormalization(axis = 3))

    def call(self, input_tensor, *args, **kwargs):

        #MAIN PATH
        X = self.ConBlock(input_tensor, *args, **kwargs)

        # SHORTCUT PATH
        X_shortcut = self.downsample(input_tensor, *args, **kwargs)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X += X_shortcut
        X = Activation('relu')(X)

        return X


class BuildingBlock(Layer):
    """
        Creates a `BuildingBlock` layer instance.

        Args:
        filter: the number of filters in the convolution
        num_block: number of `BuildingBlock` in a layer
        use_DownSample: type of shortcut connection: Identity or Convolutional shortcut
        use_BottleNeck: type of block: basic or bottleneck

        Returns:
        X -- output of the BuildingBlock block, tensor of shape (n_H, n_W, n_C)
    """
    def __init__(self, filter, stride, use_BottleNeck=False, use_DownSample=False):
        super(BuildingBlock, self).__init__()

        if (use_BottleNeck):
            self.Block = BottleNeckBlock(filter, stride)
        else:
            self.Block = BasicBlock(filter, stride)

        if use_DownSample:
            self.Down = ConvolutionalBlock(filter, use_BottleNeck=use_BottleNeck)
        else:
            self.Down = 0

    def call(self, input_tensor, *args, **kwargs):
        if (self.Down != 0):
            X = self.Down(input_tensor, *args, **kwargs)
        else:
            X = self.Block(input_tensor, *args, **kwargs)
            X += input_tensor
            X = Activation('relu')(X)
        return X