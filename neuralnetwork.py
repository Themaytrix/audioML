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
        
        self.activations = []
        
        for i in range(len(layers)):
            # instantiate dummy activations to be 1D array
            activation = np.zeros(layers[i])
            self.activations.append(activation)
        
        self.derivatives = []
        
        for i in range(len(layers)-1):
            # instantiate dummy activations to be 1D array
            derivative = np.zeros(layers[i],layers[i]+1)
            self.derivatives.append(derivative)
        
            
    def fowardpropagation(self, inputs):
        activations = inputs
        self.activations[0] = inputs
        
        # calculating the net sum for each layer
        for i,w in enumerate(self.weight):
            net_inputs = np.dot(activations,w)
            # print(net_inputs)
            
            # calculate the activation i.e input for the next layer
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations
        
        return activations
    
    def backpropagate(self,error):
        # dE/dW = (y-a(i+1))*s'(h(i+1))*a_i
        # s'(h(i+1)) = s(hi+1)(1-s(h(i+1)))
        # a(i+1) = s(h(i+1))
        
        # loop in reverse order
        for i in reversed(range(self.derivatives)):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)
            
    def _sigmoid_derivative(self,x):
        return x*(1.0-x)
    
    def _sigmoid(self,net_inputs):
        y =1/ (1+ np.exp(-net_inputs))
        return y
            
if __name__ == "__main__":
    mlp = MLP()
    inputs = np.random.rand(mlp.num_inputs)
    print(mlp.fowardpropagation(inputs))
    
    
        