'''encode: utf-8'''

import numpy as np
import random
np.random.seed(0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

class BPNNRegression:
    def __init__(self,sizes):
        self.number_layer = len(sizes)
        self.sizes = sizes
        # 初始化weights和biases
        self.weights = [np.random.randn(r,c) for r,c in zip(sizes[1:],sizes[:-1])]
        self.biases = [np.random.randn(r,1) for r in sizes[1:]]
    
    def MSGD(self, train_data, epochs, mini_batch_size, eta, error=0.1):
        number_train_data = len(train_data)
        for epoch in range(epochs):
            random.shuffle(train_data)
            mini_batchs = [train_data[k:k+mini_batch_size] for k in range(0,number_train_data,mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_WB_by_mini_batch(mini_batch,eta)
            error_epoch = self.evaluate(train_data)
            print('Epoch{0} Error{1}'.format(epoch,error_epoch))
            if error_epoch < error:
                break

    def update_WB_by_mini_batch(self, mini_batch, eta):
        number_mini_batch = len(mini_batch)
        # 创建两个矩阵来记录weights和biases的偏导（变化量），大小和其一样
        dweights = [np.zeros(weight.shape) for weight in self.weights]
        dbiases = [np.zeros(bias.shape) for bias in self.biases]
        for x,y in mini_batch:
            dws,dbs = self.backward_propagation(x,y)
            dweights = [dweight+dw for dweight,dw in zip(dweights,dws)]
            dbiases = [dbias+db for dbias,db in zip(dbiases,dbs)]
        self.weights = [weight-eta/number_mini_batch*dweight for weight,dweight in zip(self.weights,dweights)]
        self.biases = [bias-eta/number_mini_batch*dbias for bias,dbias in zip(self.biases,dbiases)]
    
    def backward_propagation(self, x, y):
        # forward_propagation
        input = x
        # 记录输出值（第0层是input）
        outputs = [input]
        # 记录净输入值
        net_inputs = []
        for weight,bias in zip(self.weights,self.biases):
            net_input = np.dot(weight,input)+bias
            output = sigmoid(net_input)
            net_inputs.append(net_input)
            outputs.append(output)
            input = output
        outputs[-1] = net_inputs[-1] # 回归最后一层不经过神经网络
        # backward_propagation
        # 创建两个矩阵来记录(对于单个样本)weights和biases的偏导（变化量），大小和其一样
        dws = [np.zeros(weight.shape) for weight in self.weights]
        dbs = [np.zeros(bias.shape) for bias in self.biases]
        # 计算最后一层的误差，weight和bias
        delta = outputs[-1]-y
        dws[-1] = np.dot(delta,outputs[-2].T)
        dbs[-1] =delta
        for layer in range(2,self.number_layer):
            z = net_inputs[-layer]
            delta = dsigmoid(z) * np.dot(self.weights[-layer+1].T,delta)
            dws[-layer] = np.dot(delta,outputs[-layer-1].T)
            dbs[-layer] = delta
        return dws,dbs
    
    def _forward(self, x):
        for weight,bias in zip(self.weights,self.biases):
            z = np.dot(weight,x)+bias
            x = sigmoid(z)
        return z

    def evaluate(self,train_data):
        train_results = [[self._forward(x),y] for x,y in train_data]
        return 0.5 * np.sum([(x-y)**2 for x,y in train_results])
    
    def predict(self, test_data):
        #test_results = [self._forward(x) for x in test_data]
        test_results = np.array([self._forward(x) for x in test_data])
        return test_results

# 测试
if __name__ == "__main__":
    x_samples = np.linspace(0,10,500)
    x_samples = [np.array([[x_sample]]) for x_sample in x_samples]
    x_samples = np.array(x_samples)
    y_samples = np.sin(x_samples)
    train_data  = [[x_sample,y_sample] for x_sample,y_sample in zip(x_samples,y_samples)]
    nn = BPNNRegression([1,6,1])
    nn.MSGD(train_data, 2500, 500, 0.5)
    y_predicts = nn.predict(x_samples)

    x_samples = np.squeeze(x_samples)
    y_samples = np.squeeze(y_samples)
    y_predicts = np.squeeze(y_predicts)
    
    import matplotlib.pyplot as plt
    plt.plot(x_samples,y_samples,x_samples,y_predicts)
    plt.show()
    