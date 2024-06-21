# implementing a neural network from scratch

import numpy as np
import math

# MLP multi layer perception

class MLP:
    
    def __init__(self, num_inputs = 3, num_hidden=[3,5], num_output=2) -> None:
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_output = num_output
        
        # structure
        layers = [self.num_inputs] + self.num_hidden + [self.num_output]
        
        
        # initiating random weights
        self.weight = []
        
        for i in range(len(layers)-1):
            # generate a random n*m matrix
            w = np.random.rand(layers[i],layers[i+1])
            
            self.weight.append(w)
            
    def fowardpropagation(self, inputs):
        activations = inputs
        
        # calculating the net sum for each layer
        for w in self.weight:
            net_inputs = np.dot(activations,w)
            # print(net_inputs)
            
            # calculate the activation i.e input for the next layer
            activations = self._sigmoid(net_inputs)
        
        return activations
    
    def _sigmoid(self,net_inputs):
        y =1/ (1+ np.exp(-net_inputs))
        return y
            
if __name__ == "__main__":
    mlp = MLP()
    inputs = np.random.rand(mlp.num_inputs)
    print(mlp.fowardpropagation(inputs))
    
    
        