import numpy as np

class FNN:
    def __init__(self,layer_sizes,activation = "relu", output_activation = "sigmod"):
        self.params = {}
        self.layer_sizes = layer_sizes
        