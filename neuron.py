import math
# implementing first neuron from scratch

# need your activation function and summation of weights

def activation(inputs,weight):
    # perform summation of inputs
    h=0
    for x,w in zip(inputs,weight):
        h += x*w
        
    # perform activation. use a sigmoid function. normalize output between -1 and 1
    
    return sigmoid(h)

def sigmoid(x):
    y = 1.0 / (1+math.exp(-x))
    return y


inputs = [0.5,0.3,0.2]
weights = [0.4,0.7,0.2]

# output neuron 
output = activation(inputs,weights)

print(output)