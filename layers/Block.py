from keras.engine.topology import Layer

class Block(Layer):
    """
        An abstract class to make it easier to define the blocks
        used by AlphaGo:Zero. Given a list of layers, it stacks them
        sequentially and aggregates them in a seperate layer. The
        way of stacking (i.e. connecting them internally) can be altered
        by defining your own call(self, x)

        In a sense it is something between a Container and a Layer.

        # Properties:
            layers: List. A list of layers to be aggregated inside the
                block.
            trainable_weights: Accumulates all trainable weights in 
                this Block and forwards them
            non_trainable_weights: See trainable_weights but for
                non trainable ones.

        # Methods:
            call(x): Connect the layers internally and with the input
                again assuming sequential order (Overwrite if needed)
            compute_output_shape(input_shape):
                The shape of the block's output.
    """

    def __init__(self,**kwargs):
        super(Block, self).__init__(**kwargs)
        self.layers = list()

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def compute_output_shape(self, input_shape):
        current_shape = input_shape
        for layer in self.layers:
            current_shape = layer.compute_output_shape(current_shape)
        
        return current_shape

    # this will enable tracking of parameters within keras
    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        weights = []
        for layer in self.layers:
            weights += layer.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.non_trainable_weights
        if not self.trainable:
            trainable_weights = []
            for layer in self.layers:
                trainable_weights += layer.trainable_weights
            return trainable_weights + weights
        return weights
