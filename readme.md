### numpy实现MLP

用纯numpy实现一个包含一个隐藏层和一个输出层的全连接神经网络，具体包括前向传播算法、反向传播算法和参数训练过程，其中参数训练可选择的方法有三种：
全批量梯度下降法 (Batch Gradient Descent)、随机梯度下降法（Stochastic Gradient Descent）和小批量梯度下降法（Mini-batch Gradient Descent）。

损失函数为：多分类交叉熵损失函数

输出层激活函数为：线性激活函数

参考资料：

[softmax函数求导](https://zhuanlan.zhihu.com/p/105722023)

[前向传播(Forward Propagation)与反向传播(Back Propagation)详解](https://wangcong.net/article/FPandBP.html)