### Using GPUs

#### Supported devices
在一个典型的系统中，有很多计算装置。在Tensorflow中，支持的计算装置类型是CPU和GPU。他们是用**strings**进行表示的。举个例子:
- "/cpu:0": 你机器的CPU
- "/gpu:0": 你机器的GPU,如果你有一个
- "/gpu:1": 你机器的第二个GPU,等等

当一个Tensorflow操作同时有CPU和GPU操作的话，会优先考虑使用GPU。举个例子，**matmul**都有CPU和GPU核。在一个带有cpu:0和gpu:0的装置系统，gpu:0将被选择进行**matmul**。
```
# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
```
你应该看看下面的输出:
```
Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: Tesla K40c, pci bus
id: 0000:05:00.0
b: /job:localhost/replica:0/task:0/gpu:0
a: /job:localhost/replica:0/task:0/gpu:0
MatMul: /job:localhost/replica:0/task:0/gpu:0
[[ 22.  28.]
 [ 49.  64.]]
```
#### Manual device placement
如果你想一个特别的装置，你可以使用**tf.device**去创造一个装置以至于所有的操作可以有同一个指定装置。
```
# Creates a graph.
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
```
你可以看到**a**和**b**将被指定给**cpu:0**。因为一个装置没有明确的赋值给**MatMul**操作，TensorFlow运行时会基于可操作和可得到的装置选择一个装置，并在装置之间自动拷贝张量。
```
Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: Tesla K40c, pci bus
id: 0000:05:00.0
b: /job:localhost/replica:0/task:0/cpu:0
a: /job:localhost/replica:0/task:0/cpu:0
MatMul: /job:localhost/replica:0/task:0/gpu:0
[[ 22.  28.]
 [ 49.  64.]]
```
#### Allowing GPU memory growth
默认的，Tensorflow使用GPUs的所有的GPU内存，通过减少内存碎片，使用相对宝贵的GPU内存资源会更加高效。

在一些例子上面，可以配置可利用的内存部分，或者增加进程需要的内存。Tensorflow提供了两种配置操作去控制这个。

第一个是**allow_growth**的选项，尝试去配置尽可能多的GPU内存:这个刚开始会配置很少内存，但是随着进程运行，同时GPU内存被需要，我们扩展Tensorflow需要的内存空间。注意我们不释放内存，因为这将导致更差的内存碎片。打开这个设置，在ConfigProto中设置:
```
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config, ...)
```
第二种方法是**per_process_gpu_memory_fraction**选项，决定每一个可用的GPU可配置的内存数比例。举个例子，你可以告诉Tensorflow配置每一个GPU40%的内存:
```
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config, ...)
```
这是有用的，如果你真的想在Tensorflow进程中使用一定数量的GPU内存。

#### Using a single GPU on a multi-GPU system
如果系统中有多余一个GPU，默认将会选择最小ID的GPU。如果你想在不同的GPU上面运行，你将显示的指定首选项:
```
# Creates a graph.
with tf.device('/gpu:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
```
如果你指定的装置不存在，你将得到**InvalidArgumentError**:
```
InvalidArgumentError: Invalid argument: Cannot assign a device to node 'b':
Could not satisfy explicit device specification '/gpu:2'
   [[Node: b = Const[dtype=DT_FLOAT, value=Tensor<type: float shape: [3,2]
   values: 1 2 3...>, _device="/gpu:2"]()]]
```
如果你想Tensorflow自动选择一个已经存在的和支持的装置区去运行，以防指定的装置不存在，你可以在配置选项中设置**allow_soft_placement**为**True**。
```
# Creates a graph.
with tf.device('/gpu:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with allow_soft_placement and log_device_placement set
# to True.
sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
# Runs the op.
print(sess.run(c))
```
#### Using multiple GPUs
如果你想在不同的GPUs上运行Tensorflow，你可以使用多塔楼的方式构建模型，其中每一层楼被指定到不同的GPU上。举个例子:
```
# Creates a graph.
c = []
for d in ['/gpu:2', '/gpu:3']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(sum))
You will see the following output.

Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: Tesla K20m, pci bus
id: 0000:02:00.0
/job:localhost/replica:0/task:0/gpu:1 -> device: 1, name: Tesla K20m, pci bus
id: 0000:03:00.0
/job:localhost/replica:0/task:0/gpu:2 -> device: 2, name: Tesla K20m, pci bus
id: 0000:83:00.0
/job:localhost/replica:0/task:0/gpu:3 -> device: 3, name: Tesla K20m, pci bus
id: 0000:84:00.0
Const_3: /job:localhost/replica:0/task:0/gpu:3
Const_2: /job:localhost/replica:0/task:0/gpu:3
MatMul_1: /job:localhost/replica:0/task:0/gpu:3
Const_1: /job:localhost/replica:0/task:0/gpu:2
Const: /job:localhost/replica:0/task:0/gpu:2
MatMul: /job:localhost/replica:0/task:0/gpu:2
AddN: /job:localhost/replica:0/task:0/cpu:0
[[  44.   56.]
 [  98.  128.]]
```
**cifar10 tutorial**是一个很好的例子展示如何通过多个GPUs训练。
