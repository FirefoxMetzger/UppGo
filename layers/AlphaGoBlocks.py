from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Add, Flatten
# does AlphaGO use Conv2D or Conv3D? -- I will assume Conv2D, but might need to fix
from keras.utils import conv_utils
from keras.regularizers import l2

from layers.Block import Block


class ConvNorm(Block):
    def __init__(self, features=256, kernel=3, c=1e-4, data_format=None,**kwargs):
        super(ConvNorm, self).__init__(**kwargs)
        data_format = conv_utils.normalize_data_format(data_format)

        # initialize all layers
        self.layers = [
            Conv2D( features, 
                kernel, 
                strides=1, 
                padding="same",
                data_format=data_format,
                kernel_regularizer=l2(c),
                bias_regularizer=l2(c)),
        BatchNormalization(axis=3,scale=False)
        ]   

class Conv(Block):
    def __init__(self, features=256, kernel=3, c=1e-4, data_format=None, **kwargs):
        super(Conv, self).__init__(**kwargs)

        self.layers = [
            ConvNorm(**kwargs),
            Activation("relu")
        ]

class Residual(Block):
    # residual block
    # conv 256 filters kernel 3x3 stride 1
    # norm
    # relu
    # conv 256 kernel 3x3 stride 1
    # norm
    # skip (add input to current tensor)
    # relu
    def __init__(self, input_size, c=1e-4,**kwargs):
        super(Residual, self).__init__(**kwargs)

        self.layers = [
            Activation("linear", trainable=False),
            ConvNorm(features=input_size, c=c),
            Activation("relu"),
            ConvNorm(features=input_size, c=c),
            Add(),
            Conv2D( input_size,3,
                    strides=1,
                    padding="same",
                    kernel_regularizer=l2(c),
                    bias_regularizer=l2(c))
        ]

    def call(self, x):
        # the residual block using Keras functional API

        first = self.layers[0](x)           # linear input layer
        x =     self.layers[1](x)           # ConvNorm
        x =     self.layers[2](x)           # ReLu
        x =     self.layers[3](x)           # ConvNorm
        x =     self.layers[4]([x, first])  # Add
        x =     self.layers[5](x)           # Conv2D

        return x

    def compute_output_shape(self, input_shape):
        first = self.layers[0].compute_output_shape(input_shape)
        shape = self.layers[1].compute_output_shape(first)
        shape = self.layers[2].compute_output_shape(shape)
        shape = self.layers[3].compute_output_shape(shape)
        shape = self.layers[4].compute_output_shape([shape, first])
        shape = self.layers[5].compute_output_shape(shape)

        return shape

class ResidualTower(Block):
    def __init__(self, input_size, layers, **kwargs):
        super(ResidualTower, self).__init__(**kwargs)

        self.layers = [Residual(input_size) for _ in range(layers)]

class ValueHead(Block):
    def __init__(self, **kwargs):
        super(ValueHead, self).__init__(**kwargs)

        self.layers = [
            ConvNorm(features=1 ,kernel=1),
            Activation("relu"),
            Flatten(),
            Dense(  256, 
                    bias_regularizer=l2(1e-4),
                    kernel_regularizer=l2(1e-4)),
            Activation("relu"),
            Dense(  1,
            bias_regularizer=l2(1e-4),
            kernel_regularizer=l2(1e-4)),
            Activation("tanh")
        ]

class PolicyHead(Block):
    def __init__(self, c=1e-4, **kwargs):
        super(PolicyHead, self).__init__(**kwargs)

        self.layers = [
            ConvNorm(features=2 ,kernel=1, c=c),
            Activation("relu"),
            Flatten(),
            Dense(  362,
                    kernel_regularizer=l2(c),
                    bias_regularizer=l2(c),
                    activation="softmax")
        ]
