### How to Retrain Inception's Final Layer for New Categories
现在目标识别模型有很多参数，并且需要花费很多周去训练。迁移学习通过对像ImageNet的一整套目录进行完整的训练，然后对新的类用已经存在的权重进行重新
训练，通过这种技术减少了很多训练时间。在这个例子里面我们将从头开始训练最后一层，不管其它层。对于这个方法的更多信息，可以看Decaf的论文。
