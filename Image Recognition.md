### Image Recognition
我们的头脑使视觉看起来容易。对于人类来说区分狮子和美洲虎，阅读一个信号，以及识别人脸都不是很费劲。但是对于机器来说是比较费劲的事情:它们只是看起来容易，因为我们的大脑非常善于理解图片。

在过去几年里面，机器学习领域已经在解决这些困难问题中取得了巨大的进步。特别的是，我们已经发现了一种被称为深度卷积神经网络的模型可以在困难的视觉识别任务中取得理想的成绩-在某些领域匹配或者超过人类效果。

研究者已经通过在ImageNet上面验证他们的工作，在计算机视觉领域显示了稳定的进步。
ImageNet是计算机视觉的学术基础。连续性的模型持续在展示提升，每一次都实现一个新的当前最好的结果:QuocNet,AlexNet,Inception(GoogleNet),BN-Inception-v2。Google内部的或者外部的人都已经发表文章描述这些模型，但是这些结果都是难以重现的。我们现在采取下一步，通过发布我们最新模型, Inception-v3的代码用于图像识别。

Inception-v3是用来训练ImageNet大型视觉识别挑战的，使用的是从2012以来的数据。这是计算机视觉中标准的任务，模型是用来将全部的图像分成1000个类，例如"斑马","斑点狗"和"洗碗机"。举个例子，下面是AlexNet用来分类一些图片的结果

<div align=center><img width="450" height="250" src="https://github.com/kunmei/tensorflow-tutorials/blob/master/AlexClassification.png" alt='AlexClassification'/>

为了比较模型，我们检查模型预测排名前五的错误率-命名为"top-5 error rate"。<a href="http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf">AlexNet</a>在2012的验证集上面top5的错误率为15.3%；<a href="http://arxiv.org/abs/1409.4842">Inception (GoogLeNet)</a>错误率为6.67%； <a  href="http://arxiv.org/abs/1502.03167">BN-Inception-v2</a>错误率为4.9%；<a  href="https://arxiv.org/abs/1512.00567">Inception-v3</a>错误率为3.46%。

人类在ImageNet的挑战是多少？Andrej Karpathy他自己进行了尝试，写了博客<a href="http://karpathy.github.io/2014/09/02/what-i-learned-from-competing-against-a-convnet-on-imagenet/">blog post</a>，达到了5.1%的错误率

这个手册会告诉你使用Inceptor-v3。你将会学到如何用Python或者C++将图像分成1000类。我们也会讨论如何从这个模型中提取出更高级的特征，可以会用作其他视觉任务。

我们对于社区将要如何利用这个模型将会很兴奋。

#### Usage with Python API
classify_image.py从**tensorflow.org**下载了训练模型，当这个程序第一次运行的时候，在硬盘上面你大概需要200M可用的空间。

从GitHub上面复制模型相关代码，执行下面的命令:

```
cd models/tutorials/image/imagenet
python classify_image.py
```
上面的命令将对一个熊猫图像进行分类。

如果这个模型执行是正确的，脚本将会产生下面的输出:
```
giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca (score = 0.88493)
indri, indris, Indri indri, Indri brevicaudatus (score = 0.00878)
lesser panda, red panda, panda, bear cat, cat bear, Ailurus fulgens (score = 0.00317)
custard apple (score = 0.00149)
earthstar (score = 0.00127)
```
如果你将提供其他JPEG图像，你可以编辑**--image_file**参数。

如果你将模型数据下载到不同的目录，你可以指定**--model_dir**到已经使用的目录。

![](http://chart.googleapis.com/chart?cht=tx&chl=\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a})
