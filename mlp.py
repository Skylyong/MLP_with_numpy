import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# Note: you may import library functions from numpy or scipy to compute distances
from scipy.spatial.distance import cdist


# 构造数据集
def oracle(x, landmarks):
    dists = cdist(x, landmarks, metric='euclidean')
    d1 = dists.max(axis=1)
    d2 = dists.min(axis=1)
    return d1 * d2


def make_dataset_regression(size=100, complexity=2, ndim=3, return_landmarks=False):
    data_mtx = np.random.rand(size, ndim)
    landmarks = np.random.rand(complexity, ndim)
    y = oracle(data_mtx, landmarks)
    if return_landmarks:
        return data_mtx, y, landmarks
    else:
        return data_mtx, y


def make_2d_grid_dataset_regression(size, landmarks):
    x = np.linspace(0.0, 1.0, int(size ** 0.5))
    y = np.linspace(0.0, 1.0, int(size ** 0.5))
    xx, yy = np.meshgrid(x, y)
    print(xx.shape)
    z = np.dstack((xx, yy))
    data_mtx = z.reshape(-1, 2)
    y = oracle(data_mtx, landmarks)
    return data_mtx, y


def oracle_classification(X, pos_landmarks, neg_landmarks):
    pos_value = oracle(X, pos_landmarks)
    neg_value = oracle(X, neg_landmarks)
    return (pos_value <= neg_value).astype(int)


def make_dataset_classification(size=100, complexity=2, ndim=3, return_landmarks=False):
    data_mtx = np.random.rand(size, ndim)
    pos_landmarks = np.random.rand(complexity, ndim)
    neg_landmarks = np.random.rand(complexity, ndim)
    y = oracle_classification(data_mtx, pos_landmarks, neg_landmarks)
    if return_landmarks:
        return data_mtx, y, pos_landmarks, neg_landmarks
    else:
        return data_mtx, y


def make_2d_grid_dataset_classification(size, pos_landmarks, neg_landmarks):
    x = np.linspace(0.0, 1.0, int(size ** 0.5))
    y = np.linspace(0.0, 1.0, int(size ** 0.5))
    xx, yy = np.meshgrid(x, y)
    z = np.dstack((xx, yy))
    data_mtx = z.reshape(-1, 2)
    y = oracle_classification(data_mtx, pos_landmarks, neg_landmarks)
    return data_mtx, y


def plot_2d_classification(X_test, y_test, preds, pos_landmarks, neg_landmarks):
    acc = np.sum(y_test == preds) / y_test.shape[0]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(18, 7)
    ax1.set_title("Prediction")
    ax2.set_title("Truth")
    ax3.set_title(f"Comparsion acc:{acc}")

    ax1.scatter(X_test[:, 0], X_test[:, 1], c=preds, marker='o', cmap=plt.cm.coolwarm, alpha=0.2)
    ax1.scatter(pos_landmarks[:, 0], pos_landmarks[:, 1], s=200, c='b', marker='*', alpha=0.65)
    ax1.scatter(neg_landmarks[:, 0], neg_landmarks[:, 1], s=200, c='r', marker='*', alpha=0.65)

    ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', cmap=plt.cm.coolwarm, alpha=0.2)
    ax2.scatter(pos_landmarks[:, 0], pos_landmarks[:, 1], s=200, c='b', marker='*', alpha=0.65)
    ax2.scatter(neg_landmarks[:, 0], neg_landmarks[:, 1], s=200, c='r', marker='*', alpha=0.65)

    ax3.scatter(X_test[:, 0], X_test[:, 1], c=preds, marker='o', cmap=plt.cm.coolwarm, alpha=0.2)
    ax3.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', cmap=plt.cm.coolwarm, alpha=0.2)
    ax3.scatter(pos_landmarks[:, 0], pos_landmarks[:, 1], s=200, c='b', marker='*', alpha=0.65)
    ax3.scatter(neg_landmarks[:, 0], neg_landmarks[:, 1], s=200, c='r', marker='*', alpha=0.65)
    plt.show()

def plot_2d_regression(X_test, y_test, preds, landmarks):
    # YOUR CODE HERE
    # raise NotImplementedError()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 7)
    ax1.set_title("Prediction")
    ax2.set_title("Truth")
    ax1.scatter(X_test[:, 0], X_test[:, 1], c=preds, marker='o', cmap=plt.cm.coolwarm, alpha=0.2)
    ax2.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', cmap=plt.cm.coolwarm, alpha=0.2)
    ax1.scatter(landmarks[:, 0], landmarks[:, 1], s=200, c='r', marker='*', alpha=0.65)
    ax2.scatter(landmarks[:, 0], landmarks[:, 1], s=200, c='r', marker='*', alpha=0.65)
    plt.show()


# 测试 构造数据集
# pos_landmarks = np.random.rand(2, 2)
# neg_landmarks = np.random.rand(2, 2)
# data_mtx, y_test = make_2d_grid_dataset_classification(10000, pos_landmarks, neg_landmarks)
#
# plot_2d_classification(data_mtx, y_test, y_test, pos_landmarks, neg_landmarks)
#
# landmarks = np.random.rand(2, 2)
# X_test, y_test = make_2d_grid_dataset_regression(100000, landmarks)
# plot_2d_regression(X_test, y_test, y_test, landmarks)



# 实现常见的激活函数、损失函数和它们对应的导函数
def mse_loss(y, t):
    return np.mean(np.square(y - t))


def mse_loss_grad(y, t):
    return 2*(y-t)/np.prod(y.shape)



def binary_crossentropy_loss(y, t):
    y = np.clip(y, 1e-7, 1 - 1e-7)
    term_0 = (1 - t) * np.log(1 - y + 1e-7)
    term_1 = t * np.log(y + 1e-7)
    return -np.mean(term_0 + term_1, axis=0)


def binary_crossentropy_loss_grad(y, t):
   return (1 / y.shape[1]) * (-(t / y) + ((1 - t) / (1 - y)))


def linear_activation(x):
    return x


def linear_activation_grad(x):
    return np.array([1])


def logistic_activation(x):
    return 1/(1+np.exp(-x))


def logistic_activation_grad(x):
    return logistic_activation(x)*(1-logistic_activation(x))


def relu_activation(x, alpha=.05):
    return np.where(x < 0, alpha*x, x)

def relu_activation_grad(x, alpha=.05):
    return np.where(x < 0, alpha, 1)


# 实现一个包含一个隐藏层和一个输出层的全连接神经网络

def init_multilayer_perceptron(in_dim, hidden_dim, hidden_activation_func, hidden_activation_func_grad,
                               out_activation_func, out_activation_func_grad, loss, loss_grad, init_size=1e-3):
    '''
    初始化网络参数、激活函数、损失函数
    :param in_dim: 输入向量的维度
    :param hidden_dim: 隐藏层输出维度
    :param hidden_activation_func: 隐藏层激活函数
    :param hidden_activation_func_grad: 隐藏层激活函数导数
    :param out_activation_func: 输出层激活函数
    :param out_activation_func_grad: 输出层激活函数导数
    :param loss: 损失函数
    :param loss_grad: 损失函数导数
    :param init_size: 初始化参数的边界
    :return: 一个包含模型要素的list
    '''
    W_ih = np.random.uniform(-init_size, init_size, (in_dim, hidden_dim))
    b_ih = np.random.uniform(-init_size, init_size, hidden_dim).reshape(1, -1)
    W_ho = np.random.uniform(-init_size, init_size, hidden_dim).reshape(1, -1)
    b_ho = np.random.uniform(-init_size, init_size, 1).reshape(1, -1)
    return [W_ih, b_ih, W_ho, b_ho, \
            hidden_activation_func, \
            hidden_activation_func_grad, \
            out_activation_func, \
            out_activation_func_grad, \
            loss, loss_grad]


def forward_multilayer_perceptron(x, perceptron_model, return_pre_activation=False):
    '''
    前向传播算法
    :param x: 输入向量
    :param perceptron_model:  init_multilayer_perceptron()返回的模型
    :param return_pre_activation: 是否返回中间变量(反向传播的时候需要计算中间变量)
    :return: 预测值
    '''
    x = x.reshape(1, -1)
    h_ih = x.dot(perceptron_model[0])+perceptron_model[1].reshape(1, -1)
    h = perceptron_model[4](h_ih)
    a = h.dot(perceptron_model[2].reshape(-1, 1))+perceptron_model[3]
    y = perceptron_model[6](a)

    if return_pre_activation:
        return y, a, h, h_ih
    else:
        return y



def compute_gradient_multilayer_perceptron(x, t, perceptron_model):
    '''
    计算参数的梯度
    :param x: 输入向量
    :param t: 真实值
    :param perceptron_model: 模型
    :return: 各参数的梯度
    '''
    y, a, h, h_ih = forward_multilayer_perceptron(x, perceptron_model, return_pre_activation=True)
    Db_ho = perceptron_model[-1](y,t)*perceptron_model[-3](a)
    DW_ho = perceptron_model[-1](y,t)*perceptron_model[-3](a)*h
    Db_ih = perceptron_model[-1](y,t)*perceptron_model[-3](a)*perceptron_model[2]*perceptron_model[5](h_ih)
    DW_ih =  np.tile(Db_ih, (2,1))*x.reshape(-1, 1)
    return DW_ih, Db_ih, DW_ho, Db_ho


def update_multilayer_perceptron(grads, learning_rate, perceptron_model):
    '''
    利用梯度下降法更新参数
    :param grads: 隐藏层、输出层参数的梯度
    :param learning_rate: 学习率
    :param perceptron_model: 模型
    :return: 参数更新后的模型
    '''
    for i in range(4):
        # print (f'grads[i]: {grads[i]}')
        perceptron_model[i] -= learning_rate*grads[i]
    return perceptron_model


def fit_multilayer_perceptron(X_train, y_train, hidden_activation_func, hidden_activation_func_grad,
                              out_activation_func, out_activation_func_grad, loss, loss_grad, learning_rate, hidden_dim,
                              batch_size=8, max_n_iter=10000, verbose=False):
    '''
    构建模型、并用训练集对模型进行训练
    :param X_train: 训练集特征向量
    :param y_train: 训练集标签
    :param hidden_activation_func: 隐藏层激活函数
    :param hidden_activation_func_grad:  隐藏层激活函数的导数
    :param out_activation_func: 输出层激活函数
    :param out_activation_func_grad: 输出层激活函数的导数
    :param loss: 损失函数
    :param loss_grad: 损失函数的导数
    :param learning_rate: 学习率
    :param hidden_dim: 隐藏层的输出维度
    :param batch_size: 批大小
    :param max_n_iter: 训练轮次
    :param verbose: 如果为true，每隔100轮打印一次损失值
    :return: 训练好的模型
    '''
    model = init_multilayer_perceptron(X_train.shape[1], hidden_dim, hidden_activation_func, hidden_activation_func_grad,
                               out_activation_func, out_activation_func_grad, loss, loss_grad)
    for iter in range(max_n_iter):
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        for i in range(0, X_train.shape[0], batch_size):
            end = min(batch_size+i, X_train.shape[0])
            batch_indices = indices[i:end]
            Grad = []
            for k, index in enumerate(batch_indices):
                grad = compute_gradient_multilayer_perceptron(X_train[index], y_train[index], model)
                Grad.append(grad)
        avg_grad = [Grad[0][0],Grad[0][1],Grad[0][2],Grad[0][3]]
        for  g in Grad[1:]:
            for i in range(4):
                avg_grad[i] += g[i]
        for i in range(4):
            avg_grad[i] /= len(Grad)
        update_multilayer_perceptron(avg_grad, learning_rate, model)
        if verbose and (iter+1)%100 == 0:
            y = np.zeros(X_train.shape[0])
            for i in range(X_train.shape[0]):
                y[i] = forward_multilayer_perceptron(X_train[i], model)
            loss_value = loss(y, y_train)
            print(f"iter: {iter + 1}, loss: {np.average(loss_value)}")
    return model


def score_multilayer_perceptron(X_test, perceptron_model):
    '''
    计算测试集的预测值
    :param X_test: 测试集
    :param perceptron_model: 训练好的模型
    :return: 预测值
    '''
    scores = np.zeros(X_test.shape[0])
    for i in range(scores.shape[0]):
        scores[i] = forward_multilayer_perceptron(X_test[i], perceptron_model)
    return scores


def predict_multilayer_perceptron(X_test, perceptron_model):
    '''

    :param X_test:
    :param perceptron_model:
    :return:
    '''
    print (X_test)
    scores = score_multilayer_perceptron(X_test, perceptron_model)

    print (scores)
    return np.where(scores >= 0.5, 1, 0).reshape(-1)


# Test Q7
# data, y, pos_landmarks, neg_landmarks = make_dataset_classification(size=300, complexity=2, ndim=2, return_landmarks=True)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=.3)
# X_test, y_test = make_2d_grid_dataset_classification(3000, pos_landmarks, neg_landmarks)
#
# perceptron_model = fit_multilayer_perceptron(X_train, y_train,
#                                              relu_activation, relu_activation_grad,
#                                              logistic_activation, logistic_activation_grad,
#                                              binary_crossentropy_loss, binary_crossentropy_loss_grad,
#                                              learning_rate=1e-1, hidden_dim=4, batch_size=32,
#                                              max_n_iter=3000, verbose=True)
# preds = predict_multilayer_perceptron(X_test, perceptron_model)
#
# print (preds, y_test)
#
# plot_2d_classification(X_test, y_test, preds, pos_landmarks, neg_landmarks)

# regression
# data, y, landmarks = make_dataset_regression(size=300, complexity=2, ndim=2, return_landmarks=True)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=.3)
# X_test, y_test = make_2d_grid_dataset_regression(3000, landmarks)
#
# perceptron_model = fit_multilayer_perceptron(X_train, y_train,
#                                              relu_activation, relu_activation_grad,
#                                              linear_activation, linear_activation_grad,
#                                              mse_loss, mse_loss_grad,
#                                              learning_rate=1e-2, hidden_dim=4, batch_size=32,
#                                              max_n_iter=3000, verbose=True)
# preds = score_multilayer_perceptron(X_test, perceptron_model)
# plot_2d_regression(X_test, y_test, preds, landmarks)