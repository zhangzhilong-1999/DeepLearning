import numpy as np

np.random.seed(0)

def sigmoid(x):
    return 1/(1+np.exp(-x))
def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

class BPNNRegression:
    def __init__(self,sizes):
        # 神经网络结构
        self.number_layer = len(sizes)
        self.sizes = sizes
        # 初始化偏差，除输入层外，其他层每个节点都生成一个bias值
        self.biases = [np.random.randn(size,1) for size in sizes[1:]]
        # 初始化每条神经元的权重
        self.weights = [np.random.randn(r,c) for r, c in zip(sizes[1:],sizes[:-1])]
    def MSGD(self, train_data, epochs, mini_batch_size, eta, error = 0.1):
        n_samples = len(train_data)
        for epoch in range(epochs):
            # 随机打乱训练集数据
            np.random.shuffle(train_data) 
            # 根据小样本大小划分子训练集集合
            mini_batchs = [train_data[k:k+mini_batch_size] for k in range(0,n_samples,mini_batch_size)]
            # 利用每次小样更新 w 和 b
            for mini_batch in mini_batchs:
                self.update_WB_by_minibatch(mini_batch,eta)
            # 迭代一次(one epoch)后结果
            error_epoch = self.evaluate(train_data)
            print('Epoch {0} Error {1}'.format(epoch, error_epoch))
            if error_epoch < error:
                break
    def update_WB_by_minibatch(self, mini_batch, eta):
        n_minibatch = len(mini_batch)
        # 创建矩阵来记录weights和biases的增量,大小和weights,biases一样
        dbiases = [np.zeros(bias.shape) for bias in self.biases]
        dweights = [np.zeros(weight.shape) for weight in self.weights]
        
        # 根据小样本中每个样本的x和y, 计算weighthe bias的偏导
        for x, y in mini_batch:
            dbs, dws = self.back_propagation(x, y)
            dbiases = [dbias+db for dbias,db in zip(dbiases,dbs)]
            dweights = [dweight+dw for dweight,dw in zip(dweights,dws)]
        self.biases = [bias-eta/n_minibatch*dbias for bias,dbias in zip(self.biases,dbiases)]
        self.weights = [weight-eta/n_minibatch*dweight for weight,dweight in zip(self.weights,dweights)]
    def back_propagation(self, x, y):
        # forward_propagation
        input = x
        # 存储每个神经元的输出值 activation[0] = x
        outputs = [input]
        # 存储每个神经元的净输入值
        net_inputs = []
        for bias,weight in zip(self.biases,self.weights):
            net_input = np.dot(weight,input)+bias
            output = sigmoid(net_input)
            # 记录净输入，输出等
            net_inputs.append(net_input)
            outputs.append(output)
            input = output
        outputs[-1] = net_inputs[-1] # 最后一层g(z)=z
        
        # backward_forGradient
        # 创建矩阵来记录weights和biases的增量,大小和weights,biases一样
        dbs = [np.zeros(bias.shape) for bias in self.biases]
        dws = [np.zeros(weight.shape) for weight in self.weights]
        # 计算最后一层的误差
        delta = outputs[-1]-y
        # 更新最后一层的bias和weight的偏导数
        dbs[-1] = delta
        dws[-1] = delta*outputs[-1]
        # backward 更新
        for layer in range(2,self.number_layer):
            # 从倒数第1层开始更新，因此需要采用-layer
            # 利用 layer + 1 层的 δ 计算 l 层的 δ
            net_input = net_inputs[-layer]
            delta = dsigmoid(net_input) * np.dot(self.weights[-(layer-1)].T,delta)
            dbs[-layer] = delta
            dws[-layer] = np.dot(delta,outputs[-(layer+1)].T)
        return dbs,dws

    def evaluate(self, train_data):
        train_results = [[self._forward_propagation(x),y] for x,y in train_data]
        return np.sum([(x-y)**2 for (x,y) in train_results])
    def _forward_propagation(self, x):
        input = x
        for bias,weight in zip(self.biases,self.weights):
            z = np.dot(weight, input) + bias
            input = sigmoid(z)
        output = z # 回归，最后一层不经过激活函数
        return output
    def predict(self, test_x_samples):
        test_results = [self._forward_propagation(x) for x in test_x_samples]
        return test_results

if __name__ == "__main__":
    x_samples = np.linspace(0,10,500)
    x_samples = [np.array([[x_sample]]) for x_sample in x_samples]
    x_samples = np.array(x_samples)
    y_samples = np.sin(x_samples)
    train_data  = [[x_sample,y_sample] for x_sample,y_sample in zip(x_samples,y_samples)]
    nn = BPNNRegression([1,4,4,1])
    nn.MSGD(train_data, 1000, 500, 0.5)
    y_predicts = nn.predict(x_samples)

    x_samples = np.squeeze(x_samples)
    y_samples = np.squeeze(y_samples)
    y_predicts = np.squeeze(y_predicts)
    
    import matplotlib.pyplot as plt
    plt.plot(x_samples,y_samples,x_samples,y_predicts)
    plt.show()
    