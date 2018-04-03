from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, BatchNormalization, Activation, Add, add, Flatten
# does AlphaGO use Conv2D or Conv3D? -- I will assume Conv2D, but might need to fix
from layers.AlphaGoBlocks import ResidualTower, ValueHead, PolicyHead, Conv

from keras.utils.vis_utils import plot_model
from keras.regularizers import l2


# input: 19x19x17
# last 8 board states for current player
# last 8 board states for opposite player
# 1 filled with current player (1=black, 0=white)
# regularization is c = 0.0001
tower = Sequential()
tower.add(Conv(input_shape=(19,19,17)))
# 19 residual blocks until the policy head
# my little 650 Ti can't fit 19 layers into memory I will reduce this for testing :(
tower.add(ResidualTower(256, 10)) # set this to 19 on TITAN V
tower.add(Activation("linear", trainable=False, name="head"))
residual_head = tower.get_layer(name="head").output

value = ValueHead(name="value")(residual_head)
policy = PolicyHead(name="policy")(residual_head)

model = Model(tower.input, [policy,value])

if __name__ == "__main__":
    # if this function is called, visualize the network
    # this is for debugging purposes mainly
    model.summary()
    plot_model(model, to_file="test.png")