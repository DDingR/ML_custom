import numpy as np
from src.activation import *
from src.network import *

network = TwoLayerNet(3, 4, 2)
x = np.random.rand(3)
y = network.predict(x)

print(y)
