import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

# Note: you may import library functions from numpy or scipy to compute distances
from scipy.spatial.distance import cdist


# Q1
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


# Test Q1
# pos_landmarks = np.random.rand(2, 2)
# neg_landmarks = np.random.rand(2, 2)
# data_mtx, y_test = make_2d_grid_dataset_classification(10000, pos_landmarks, neg_landmarks)
#
# plot_2d_classification(data_mtx, y_test, y_test, pos_landmarks, neg_landmarks)
#
# landmarks = np.random.rand(2, 2)
# X_test, y_test = make_2d_grid_dataset_regression(100000, landmarks)
# plot_2d_regression(X_test, y_test, y_test, landmarks)



# Q2
from sklearn.neural_network import MLPClassifier, MLPRegressor


def fit_ann_regression(X_train, y_train, param):
    model = MLPRegressor(hidden_layer_sizes=param, activation='relu', \
                         alpha=1e-3, learning_rate_init=1e-3, \
                         max_iter=3000).fit(X_train, y_train)
    return model


def predict_ann_regression(X_test, ann_model):
    return ann_model.predict(X_test)


def fit_ann_classification(X_train, y_train, param):
    model = MLPClassifier(hidden_layer_sizes=param, activation='relu', \
                          alpha=1e-3, learning_rate_init=1e-3, \
                          max_iter=3000).fit(X_train, y_train)
    return model


def predict_ann_classification(X_test, ann_model):
    return ann_model.predict(X_test)


def score_ann_classification(X_test, ann_model):
    proba = ann_model.predict_proba(X_test)
    return proba[:, 1]


# Test Q2

# Just run the following code, do not modify it
# data, y, pos_landmarks, neg_landmarks = make_dataset_classification(size=300, complexity=2, ndim=2, return_landmarks=True)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=.3)
# X_test, y_test = make_2d_grid_dataset_classification(3000, pos_landmarks, neg_landmarks)
#
# perceptron_model = fit_ann_classification(X_train, y_train, param=(4,))
# preds = predict_ann_classification(X_test, perceptron_model)
# plot_2d_classification(X_test, y_test, preds, pos_landmarks, neg_landmarks)


# Q3
def true_positive_rate(preds, targets):
    ids = np.argsort(-preds)
    sorted_targets = targets[ids]
    P = np.sum(targets == 1)
    N = np.sum(targets != 1)
    TPR = np.cumsum(sorted_targets) / P

    return TPR


def false_positive_rate(preds, targets):
    ids = np.argsort(-preds)
    sorted_targets = targets[ids]
    P = np.sum(targets == 1)
    N = np.sum(targets != 1)
    FPR = np.cumsum(1 - sorted_targets)/N
    return FPR


def compute_auc(TPR, FPR):
    y = TPR[1:]
    x = np.diff(FPR, axis=0)
    return np.sum(x * y)



def compute_scores(fit_func, score_func, param, X_train, y_train, X_test, num):
    model = fit_func(X_train, y_train, param)
    scores_list = []
    for _ in range(num):
        scores_list.append(score_func(X_test, model))
    return scores_list


def compute_tpr_fpr_range(scores_list, y_test, false_positive_rate_func, true_positive_rate_func, low_quantile,
                          high_quantile):
    # YOUR CODE HERE
    raise NotImplementedError()


def plot_roc(tpr_low, tpr_high, tpr_mid, fpr, compute_auc_func):
    plt.figure(figsize=(6, 6))
    # label1 = 'AUC ROC:%.2f' % compute_auc(TPR1, FPR1)
    plt.plot(fpr, tpr_low, c='tab:red')
    plt.scatter(fpr, tpr_low, c='tab:red')

    # label2 = 'AUC ROC:%.2f' % compute_auc(TPR2, FPR2)
    plt.plot(fpr, tpr_high, c='tab:blue')
    plt.scatter(fpr, tpr_high, c='tab:blue')

    plt.plot([0, 1], [0, 1], '--', c='gray', alpha=.4)
    plt.grid()
    plt.legend()
    plt.title('AUC ROC: %s'%compute_auc_func(tpr_mid, fpr))
    plt.show()

# Test Q3
data, y = make_dataset_classification(size=200, complexity=4, ndim=10, return_landmarks=False)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=.2)

scores_list = compute_scores(fit_ann_classification, score_ann_classification, (4,), X_train, y_train, X_test, num=10)

# print (scores_list[0].shape)
# print(scores_list[0])
# tpr_low, tpr_high, tpr_mid, fpr = compute_tpr_fpr_range(scores_list, y_test, false_positive_rate, true_positive_rate, low_quantile=0.05, high_quantile=0.95)
#
# plot_roc(tpr_low, tpr_high, tpr_mid, fpr, compute_auc)

# TPR1 = np.array([1/3, 2/3, 2/3, 1, 1])
# TPR2 = np.array([1/4, 2/4, 2/4, 1, 1])
# FPR = np.array([0. , 0. , 0.5, 0.5, 1. ])
# plot_roc(TPR1, TPR2, TPR2, FPR, compute_auc)



# Q4
def init_linear_perceptron(in_dim, init_size=1e-3):
    w = np.random.uniform(-init_size, init_size, in_dim).reshape(-1, in_dim)
    b = np.random.uniform(-init_size, init_size, 1)
    # print(w)
    # print(b)
    return [w, b]


def forward_linear_perceptron(x, perceptron_model):
    return np.sum(perceptron_model[0] * x) + perceptron_model[1]


def update_linear_perceptron(x, t, learning_rate, perceptron_model):
    perceptron_model[0] += learning_rate * x * t
    perceptron_model[1] += learning_rate * t
    return perceptron_model


def fit_linear_perceptron(X_train, y_train, learning_rate, max_n_iter=5000):
    perceptron_model = init_linear_perceptron(X_train.shape[1])
    iter = 0
    for iter in range(max_n_iter):
        index = np.random.choice(y_train.shape[0], 1)
        x, y = X_train[index], y_train[index]
        print(f"iter: {iter}, index: {index}, y: {y}")
        if y * forward_linear_perceptron(x, perceptron_model) <= 0:
            perceptron_model = update_linear_perceptron(x, y, learning_rate, perceptron_model)

    return perceptron_model


def score_linear_perceptron(X_test, perceptron_model):
    scores = np.zeros(X_test.shape[0])
    for i in range(scores.shape[0]):
        scores[i] = forward_linear_perceptron(X_test[i], perceptron_model)
    return scores


def predict_linear_perceptron(X_test, perceptron_model):
    scores = score_linear_perceptron(X_test, perceptron_model)
    return np.where(scores >= 0, 1, -1)




# test Q4
# data, y, pos_landmarks, neg_landmarks = make_dataset_classification(size=300, complexity=2, ndim=2, return_landmarks=True)
# y[y==0]=-1
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=.3)
# X_test, y_test = make_2d_grid_dataset_classification(3000, pos_landmarks, neg_landmarks)
# y_test[y_test==0]=-1
#
#
# perceptron_model = fit_linear_perceptron(X_train, y_train, learning_rate=1e-2)
# preds = predict_linear_perceptron(X_test, perceptron_model)
# scores = score_linear_perceptron(X_test, perceptron_model)
# print (scores)
# print (preds)
# print (y_test)
# plot_2d_classification(X_test, y_test, preds, pos_landmarks, neg_landmarks)
#
# test success


# Q5
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

# Test Q5



# Q6
def init_perceptron(in_dim, activation_func, activation_func_grad, loss, loss_grad, init_size=1e-3):
    w = np.random.uniform(-init_size, init_size, in_dim).reshape(1, -1)
    b = np.random.uniform(-init_size, init_size, 1)
    return [w, b, activation_func, activation_func_grad, loss, loss_grad]


def forward_perceptron(x, perceptron_model, return_pre_activation=False):
   h = np.sum(x*perceptron_model[0])+perceptron_model[1]
   y = perceptron_model[2](h)
   if return_pre_activation:
       return y,h
   else:
       return y


def update_perceptron(x, t, learning_rate, perceptron_model):
    y, h = forward_perceptron(x, perceptron_model, return_pre_activation=True)
    d_w = perceptron_model[-1](y.reshape(-1,1), t)*perceptron_model[3](h)*x
    d_b = perceptron_model[-1](y.reshape(-1,1), t)*perceptron_model[3](h)
    # print(f'd_w: {d_w}, d_b: {d_b}')
    perceptron_model[0] -= learning_rate*d_w
    perceptron_model[1] -= learning_rate * d_b.reshape(1)
    return perceptron_model

def fit_perceptron(X_train, y_train, activation_func, activation_func_grad, loss, loss_grad, learning_rate,
                   max_n_iter=1000, verbose=False):
    model = init_perceptron(X_train.shape[1], activation_func, activation_func_grad, loss, loss_grad)
    for iter in range(max_n_iter):
        # index = np.random.choice(X_train.shape[0], 1)
        for index in range(X_train.shape[0]):
            model = update_perceptron(X_train[index], y_train[index],learning_rate, model)
        if verbose and (iter+1)% 100 == 0:
            y = forward_perceptron(X_train, model)
            # print(f'y: {y}, y_train: {y_train}')
            loss_value = loss(y, y_train)
            print(f"iter: {iter+1}, loss: {loss_value}")
    return model



def score_perceptron(X_test, perceptron_model):
    scores = np.zeros(X_test.shape[0])
    for i in range(scores.shape[0]):
        scores[i] = forward_perceptron(X_test[i], perceptron_model)
    return scores

def predict_perceptron(X_test, perceptron_model):
    scores = score_perceptron(X_test, perceptron_model)
    return np.where(scores >= 0.5, 1, 0)


# Test Q 6

# data, y, pos_landmarks, neg_landmarks = make_dataset_classification(size=300, complexity=2, ndim=2, return_landmarks=True)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=.3)
# X_test, y_test = make_2d_grid_dataset_classification(3000, pos_landmarks, neg_landmarks)
#
# perceptron_model = fit_perceptron(X_train, y_train,
#                                   logistic_activation, logistic_activation_grad,
#                                   binary_crossentropy_loss, binary_crossentropy_loss_grad,
#                                   learning_rate=1e-2, verbose=False)
# preds = predict_perceptron(X_test, perceptron_model)
# # print(preds, y_test)
# # print('loss: ', perceptron_model[-2](preds, y_test))
#
#
# plot_2d_classification(X_test, y_test, preds, pos_landmarks, neg_landmarks)


# regression
# data, y, landmarks = make_dataset_regression(size=300, complexity=2, ndim=2, return_landmarks=True)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=.3)
# X_test, y_test = make_2d_grid_dataset_regression(3000, landmarks)
#
# perceptron_model = fit_perceptron(X_train, y_train,
#                                   relu_activation, relu_activation_grad,
#                                   mse_loss, mse_loss_grad,
#                                   learning_rate=1e-2, verbose=False)
# preds = score_perceptron(X_test, perceptron_model)
# plot_2d_regression(X_test, y_test, preds, landmarks)


# Q7
def init_multilayer_perceptron(in_dim, hidden_dim, hidden_activation_func, hidden_activation_func_grad,
                               out_activation_func, out_activation_func_grad, loss, loss_grad, init_size=1e-3):
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
    x = x.reshape(1, -1)
    h_ih = x.dot(perceptron_model[0])+perceptron_model[1].reshape(1, -1)
    h = perceptron_model[4](h_ih)
    a = h.dot(perceptron_model[2].reshape(-1, 1))+perceptron_model[3]
    y = perceptron_model[6](a)
    # print (f'x: {x}, y: {y}, a: {a}, h: {h}, h_ih:{h_ih}')
    if return_pre_activation:
        return y, a, h, h_ih
    else:
        return y



def compute_gradient_multilayer_perceptron(x, t, perceptron_model):
    y, a, h, h_ih = forward_multilayer_perceptron(x, perceptron_model, return_pre_activation=True)
    Db_ho = perceptron_model[-1](y,t)*perceptron_model[-3](a)
    DW_ho = perceptron_model[-1](y,t)*perceptron_model[-3](a)*h
    Db_ih = perceptron_model[-1](y,t)*perceptron_model[-3](a)*perceptron_model[2]*perceptron_model[5](h_ih)
    DW_ih =  np.tile(Db_ih, (2,1))*x.reshape(-1, 1)
    return DW_ih, Db_ih, DW_ho, Db_ho


def update_multilayer_perceptron(grads, learning_rate, perceptron_model):
    for i in range(4):
        # print (f'grads[i]: {grads[i]}')
        perceptron_model[i] -= learning_rate*grads[i]
    return perceptron_model


def fit_multilayer_perceptron(X_train, y_train, hidden_activation_func, hidden_activation_func_grad,
                              out_activation_func, out_activation_func_grad, loss, loss_grad, learning_rate, hidden_dim,
                              batch_size=8, max_n_iter=10000, verbose=False):
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
        # print (f'before model[0] : {model[0]}')
        update_multilayer_perceptron(avg_grad, learning_rate, model)
        # print (f'after model[0] : {model[0]}')
        if verbose and (iter+1)%100 == 0:
            y = np.zeros(X_train.shape[0])
            for i in range(X_train.shape[0]):
                y[i] = forward_multilayer_perceptron(X_train[i], model)
            loss_value = loss(y, y_train)
            print(f"iter: {iter + 1}, loss: {np.average(loss_value)}")
    return  model


def score_multilayer_perceptron(X_test, perceptron_model):
    scores = np.zeros(X_test.shape[0])
    for i in range(scores.shape[0]):
        scores[i] = forward_multilayer_perceptron(X_test[i], perceptron_model)
    return scores


def predict_multilayer_perceptron(X_test, perceptron_model):
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



# Q8
import itertools
from collections import Counter,defaultdict

# def train_OvO(X_train, y_train, fit_func, param):
#     n_classes = np.unique(y_train)
#     estimators = []
#     for labels in itertools.combinations(n_classes, 2):
#         index = (y_train==labels[0]) | (y_train == labels[1])
#
#         y_train_ = y_train[index]
#         X_train_ = X_train[index]
#
#         y_train_[y_train_==labels[0]] = -1
#         y_train_[y_train_ == labels[1]] = 1
#         model = fit_func(X_train_, y_train_, param)
#         estimators.append((labels[0], model, labels[1] ))
#     return estimators


def train_OvO(X_train, y_train, fit_func, param):
    estimators = []
    for labels in itertools.combinations(np.unique(y_train), 2):
        index = (y_train == labels[1]) | (y_train == labels[0])

        y_train_sub = y_train[index]
        X_train_sub = X_train[index]

        y_train_sub[y_train_sub == labels[0]] = -1
        y_train_sub[y_train_sub == labels[1]] = 1

        model = fit_func(X_train_sub, y_train_sub, param)
        estimators.append((labels[0], labels[1], model))
    return estimators


def test_OvO(X_test, predict_func, score_func, estimators):
    preds = np.zeros((len(estimators), X_test.shape[0]))
    scores = defaultdict(list)
    for idx, e in enumerate(estimators):
        score = score_func(X_test, e[2])
        scores[e[1]].append(score)
        scores[e[0]].append(1 - score)
        pred = predict_func(X_test, e[2])
        pred = np.where(pred == 1, e[1], e[0])
        preds[idx] = pred.reshape(1, -1)
    for key , items in scores.items():
        ave = np.zeros(items[0].shape[0])
        for item in items:
            ave += item
        scores[key] = ave / len(items)

    results = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        pred = preds[:, i]
        count_values = Counter(pred).most_common()
        pred_labels = [int(label) for label, count_times in count_values if count_times == count_values[0][1]]
        if len(pred_labels) > 1:
            p_scores = [scores[label][i] for label in pred_labels]
            index = p_scores.index(max(p_scores))
            results[i] = pred_labels[index]
        else:
            results[i] = pred_labels[0]
    return results

# Test Q8
from sklearn.datasets import make_classification
X,y = make_classification(n_samples=400, n_features=4, n_informative=4, n_redundant=0, n_repeated=0, n_classes=6, n_clusters_per_class=1, weights=None, flip_y=0.1, class_sep=1.0, random_state=5)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)

estimators = train_OvO(X_train, y_train, fit_ann_classification, param=(5,))
test_preds = test_OvO(X_test, predict_ann_classification, score_ann_classification, estimators)

from sklearn.metrics import classification_report
print(classification_report(y_test, test_preds))
