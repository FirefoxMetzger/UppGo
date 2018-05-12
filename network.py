from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Add, add, Flatten
# does AlphaGO use Conv2D or Conv3D? -- I will assume Conv2D, but might need to fix
from layers.AlphaGoBlocks import ResidualTower, ValueHead, PolicyHead, Conv

from keras.utils.vis_utils import plot_model
from keras.regularizers import l2
import keras


# input: 19x19x17
# last 8 board states for current player
# last 8 board states for opposite player
# 1 filled with current player (1=black, 0=white)
# regularization is c = 0.0001
def AlphaZero(input_tensor, residual_blocks=10):
    x = Conv()(input_tensor)
    x = ResidualTower(256, residual_blocks)(x)
    value = ValueHead(name="value")(x)
    policy = PolicyHead(name="policy")(x)

    return Model(input_tensor, [policy, value])

if __name__ == "__main__":
    # if this function is called, visualize the network
    # this is for debugging purposes mainly
    a = keras.Input(shape=[19, 19, 17])
    model = AlphaZero(a, residual_blocks=2)
    model.summary()
    plot_model(model, to_file="test.png")
