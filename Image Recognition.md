### Image Recognition
我们的头脑使视觉看起来容易。对于人类来说区分狮子和美洲虎，阅读一个信号，以及识别人脸都不是很费劲。但是对于机器来说是比较费劲的事情:它们只是看起来容易，因为我们的大脑非常善于理解图片。

在过去几年里面，机器学习领域已经在解决这些困难问题中取得了巨大的进步。特别的是，我们已经发现了一种被称为深度卷积神经网络的模型可以在困难的视觉识别任务中取得理想的成绩-在某些领域匹配或者超过人类效果。

研究者已经通过在ImageNet上面验证他们的工作，在计算机视觉领域显示了稳定的进步。
ImageNet是计算机视觉的学术基础。连续性的模型持续在展示提升，每一次都实现一个新的当前最好的结果:QuocNet,AlexNet,Inception(GoogleNet),BN-Inception-v2。Google内部的或者外部的人都已经发表文章描述这些模型，但是这些结果都是难以重现的。我们现在采取下一步，通过发布我们最新模型, Inception-v3的代码用于图像识别。

Inception-v3是用来训练ImageNet大型视觉识别挑战的，使用的是从2012以来的数据。这是计算机视觉中标准的任务，模型是用来将全部的图像分成1000个类，例如"斑马","斑点狗"和"洗碗机"。举个例子，下面是AlexNet用来分类一些图片的结果

![AlexClassification](https://github.com/kunmei/tensorflow-tutorials/blob/master/AlexClassification.png)

<div align=center><img width="150" height="150" src="https://github.com/kunmei/tensorflow-tutorials/blob/master/AlexClassification.png"/>
