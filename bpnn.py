# back-propagation neural network

import numpy as np

np.random.seed(0)

# calculate a sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# derivative of the sigmoid function, in terms of output (i.e. y)
def dsigmoid(y):
    return y*(1-y)

class Unit:
    def __init__(self, input_length, ishidden=1): # ishidden=1 for hidden node
        self.delta = np.random.rand() # delta 对当前node的误差，一个标量
        self.weight = np.random.rand(input_length)
        self.bias = np.random.rand()
        self.output = 0
        self.ishidden = ishidden
    # 每一节点的净输入
    def _calc(self, input): 
        return np.dot(self.weight, input) + self.bias
    def calc(self, input): 
        z = self._calc(input)
        if self.ishidden ==1:
            self.output = sigmoid(z)
        else:
            self.output = z
        return self.output
    def update(self, input, weight_next_layer=None, delta_next_layer=None, output_theLast=None, target=None, rate=0.5): # delta_latter_layer 下一层每个node的误差，一个向量； self.input 对当前node的输入，一个向量
        if self.ishidden ==1:
            self.delta = sigmoid(self._calc(input)) * np.sum(np.dot(weight_next_layer, delta_next_layer))
        else:
            self.delta = output_theLast - target
        self.weight -= rate * self.delta * self.input
        self.bias -= rate * self.delta
    
class Layer:
    def __init__(self, input_length, output_length, ishidden=1):
        self.units = [Unit(input, input_length) for i in range(output_length)]
        #self.units(-1) = Unit(input, input_length, ishidden=0) # output layer
        #self.layer_input = input
        self.layer_output = [0] * output_length
        self.input_length = input_length
        self.ishidden = ishidden
    def calc(self, input):
        self.layer_output = [unit.calc(input) for unit in self.units]
        return self.layer_output
    def update(self, input, weights_next_layer=None, delta_next_layer=None, outputs_theLast=None, targets=None, rate=0.5):
        if self.ishidden ==1:
            for weight_next_layer, unit in zip(weights_next_layer, self.units):
                unit.update(input, weight_next_layer=weight_next_layer, delta_next_layer=delta_next_layer, output_theLast=None, target=None, rate=rate)
        else:
            for output_theLast, target, unit in list(zip(outputs_theLast, targets, self.units)):
                unit.update(input, weights_next_layer=None, delta_next_layer=None, output_theLast=output_theLast, target=target, rate=rate)
    
class BPNN:
    def __init__(self, input_length, hidden_length, output_length, number_hidden_layer=5, rate=0.5):
        self.input_length = input_length
        self.hidden_length = hidden_length
        self.output_length = output_length
        self.hidden_layer = [Layer(input_length, hidden_length, ishidden=1) if i== 0 else Layer(hidden_length, hidden_length, ishidden=1) for i in range(self.hidden_length) ]
        self.output_layer = Layer(hidden_length, output_length, ishidden=0)
    
    def calc(self, input):
        if len(input) != self.input_length:
            raise ValueError('wrong number of input')
        
        self.input = input
        for i in range(len(self.hidden_layer)):
            output = self.hidden_layer[i].calc(self.input)
            self.input = output




