# CS231n 25spring

## Lec2 Image Classification with Linear Classifiers

### Nearest Neighbor Classifier

表征两张图像相似程度的距离函数 L1 (Manhattan) distance：
$$
d_1(I_1,I_2)=\sum_p|I_1^p-I_2^p|
$$
<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250913235826598.png" alt="image-20250913235826598" style="zoom: 67%;" />

一个简单的近邻分类器代码实现：

```python
import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred
```

不难发现训练过程的时间复杂度为 O(1)，而预测过程是 O(n)；这不是我们期望的，我们希望预测时花费的时间尽可能少，训练的时间长一点无所谓

### K-Nearest Neighbors

在预测一个新样本的分类时，从判断其距离哪**一个**训练集样本最近，拓展到**K个**训练集样本投票

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250914003919028.png" alt="image-20250914003919028" style="zoom:80%;" />

上图中有颜色的点为训练样本，其他部分为需要预测分类的新样本，(x,y) 为其输入的特征，颜色为其预测结果

空白区域表示该处样本点无法使用当前模型预测出结果，这说明需要搜集更多靠近**模糊区域**的训练样本

L2 (Euclidean) distance：
$$
d_2(I_1,I_2)=\sqrt{\sum_p\left(I_1^p-I_2^p\right)^2}
$$
==**当特征特别具体且有含义，并且我们希望保留其信息时，最好使用 L1**==

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250914005553154.png" alt="image-20250914005553154" style="zoom: 67%;" />

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250914005611556.png" alt="image-20250914005611556" style="zoom:67%;" />

当使用 L1 时，可以发现边界大多水平/垂直，也就是**对于特征的变化更敏感**，而使用 L2 时边界整体更圆滑

### Linear Classifier

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250914092732464.png" alt="image-20250914092732464" style="zoom: 67%;" />

几何视角，线性分类器就是在二维空间画直线/三维空间画平面来进行分类

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250914093323229.png" alt="image-20250914093323229" style="zoom:80%;" />

很显然线性分类器很难处理一些分类情况，比如一三象限为A类、二四象限为B类；环内为A类、环外为B类等

**Loss function** 评估当前分类器模型的表现有多**坏** 我们的目标就是通过不断调整参数 **W/b** 来最小化目标函数
$$
dataset\ of\ examples:\{(x_i,y_i)\}_{i=1}^N \quad loss\ function:L=\frac{1}{N}\sum_iL_i(f(x_i,W),y_i)
$$

### Softmax Classifier

希望分类器输出的标量直接就是概率
$$
Softmax\ function:\boxed{s=f(x_i;W)}\quad\boxed{P(Y=k|X=x_i)=\frac{e^{s_k}}{\sum_je^{s_j}}}
$$
损失函数可以定义为==**正确类别概率的负对数**==：
$$
L_i=-\log P(Y=y_i|X=x_i)=-\log(\frac{e^{s_{y_i}}}{\sum_je^{s_j}})
$$
最小化这个损失函数相当于最大化“模型对样本的判断等于样本label”的概率，这就是==**最大似然估计 Maximum Likelihood Estimation**==

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250914100100379.png" alt="image-20250914100100379" style="zoom:80%;" />

损失函数也可以定义为==**当前模型输入样本后产生的概率分布与正确概率分布之间的 KL divergence**==：
$$
\begin{aligned}
H(P,Q)&= H(p)+D_{KL}(P\|Q)\\&=-\sum_xP(x)\log P(x)+\sum_xP(x)\log\frac{P(x)}{Q(x)}\\&=-\sum_xP(x)\log Q(x)
\end{aligned}
$$
注意公式中的 $P$ 是真实概率分布，也就是 label 的 one-hot 编码，$Q$ 是模型预测的概率分布

最小化这个损失函数相当于“让模型预测的概率分布和真实的概率分布尽可能一致”，这就是==**交叉熵 Cross Entropy**==

但其实最终的表达式和通过最大似然估计推导的一模一样，只是出发点和推导思路不同

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250914113031355.png" alt="image-20250914113031355" style="zoom:80%;" />



## Lec3 Regularization and Optimization

### Regularization

**正则化**防止模型在训练集上过拟合，导致无法识别测试集
$$
L(W)=\frac{1}{N}\sum_{i=1}^NL_i(f(x_i,W),y_i)+\color{red}\lambda R(W)
$$
$\lambda$ 是个超参数，**取正值**，表示正则化的强度

常见的正则化函数：
$$
\begin{aligned}&\text{L2 regularization: }R(W)=\sum_k\sum_lW_{k,l}^2\\&\text{L1 regularization: }R(W)=\sum_k\sum_l|W_{k,l}|\\&\text{Elastic net (L1+L2): }R(W)=\sum_{k}\sum_{l}\beta W_{k,l}^{2}+|W_{k,l}|\end{aligned}
$$
训练网络的过程就是最小化损失函数，所以引入正则项会==**在对模型效果影响不大的前提下尽可能缩小权重的值**==

使用 $L_1$ 往往使大量参数取 0 (contain a lot of sparsity)，而使用 $L_2$ 会使参数取到接近零的非零值 (spread out )，权重取零或接近零就相当于最终拟合的多项式的阶数变小了，可以得到更简单的模型

但==正则化的核心思想是**“在训练集上表现稍差，但能在测试集上提升表现”**==，所以正则化项的使用的结果不一定是简化模型，取决于用什么正则化表达式以及怎么用

### Optimization

训练所有模型的核心思想：**梯度下降 Gradient descent**

梯度的定义：

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250914155226187.png" alt="image-20250914155226187" style="zoom:80%;" />

#### Stochastic Gradient Descent (SGD)

$$
L(W)=\frac{1}{N}\sum_{i=1}^NL_i(f(x_i,W),y_i)+\lambda R(W)\\
\nabla_WL(W)=\frac{1}{N}\sum_{i=1}^N\nabla_WL_i(x_i,y_i,W)+\lambda\nabla_WR(W)
$$

如果要遍历所有样本，当 $N$ 很大的时候消耗太大，所以选择使用样本集中**随机采样的 minibatch** 来估计遍历的完整求和

```python
While True:
    data_batch = sample_training_data(data, 256)
    weights_grad = evaluate_gradient(loss_fun, data_batch, weights)
    weights += - step_size * weights_grad
```

SGD可能面临的问题: 1. 如果 loss function 有局部最小值，即梯度为0的地方，梯度下降会卡住  

​				     2. 由于随机采样，必定存在噪声，每次更新不会直接朝着梯度最小方向

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250918090433135.png" alt="image-20250918090433135" style="zoom:80%;" />

#### SGD + Momentum

$$
v_{t+1}=\rho v_t-\alpha\nabla f(x_t) \quad x_{t+1}=x_t-\alpha v_{t+1}
$$

```python
vx = 0
while True:
    dx = compute_gradient(x)
    vx = rho * vx - learning_rate * dx # rho表征动量,一般取0.9或0.99
    x += vx
```

使用**“速度”**而非纯粹梯度来更新参数，这样可以让梯度下降过程保持一个方向的趋势

#### RMSProp

Adds element-wise scaling of the gradient based on the historical sum of squares in each dimension

```python
grad_squard = 0
while True:
    dx = compute_gradient(x)
    grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dx * dx
    x -= learning_rate * dx / (np.sqrt(grad_squared) + 1e-7)
```

相当于学习率是动态的，在梯度很大的“陡峭区域”更新的步长更小，在梯度较小的“平坦区域”更新的步长更大

#### Adam

Sort of like RMSProp with momentum

```python
first_moment = 0
second_moment = 0
for t in range(1, num_iterations):
    dx = compute_gradient(x)
    first_moment = beta1 * first_moment + (1 - bata1) * dx # Momentum
    second_moment = beta2 * second_moment + (1 - bata2) * dx * dx
    first_unbias = first_moment / (1 - beta1 ** t) # Bias correction
    second_unbias = second_moment / (1 - beta2 ** t) # 主要是修正初始步如果dx较小可能带来的超大步长问题
    x -= learning_rate * first_unbias / (np.sqrt(second_unbias) + 1e-7) # RMSProp
```

beta1 = 0.9, beta2 = 0.999, learning_rate = 1e-3 or 5e-4 is a great starting point for any models

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\1758166947648.png" alt="1758166947648" style="zoom:80%;" />

在 Adam 优化器中添加正则化项  位置比较灵活

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250918115436877.png" alt="image-20250918115436877" style="zoom:80%;" />

增加 batch 的大小n倍，初始的 learning rate 也要增加n倍

## Lec4 Neural Networks and Backpropagation

### Multi-Layer Perceptions (MLP)

Full-connected networks

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250918220234428.png" alt="image-20250918220234428" style="zoom:80%;" />

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250918220704824.png" alt="image-20250918220704824" style="zoom:80%;" />

Sigmoid 和 Tanh 可能导致**梯度消失**，所以一般不在隐藏层中使用，通常在比较深的层，比如希望二进制输出的时候使用

```python
import numpy as np
from numpy.random import randn

N, d_in, H, D_out = 64, 1000, 100, 10
x, y = randn(N, D_in), randn(N, D_out)
w1, w2 = randn(D_in, H), randn(H, D_out)

for t in range(2000):
    h = 1 / (1 + np.exp(-x.dot(w1))) # 第一层+sigmoid作为激活函数
    y_pred = h.dot(w2) # 第二层输出
    loss = np.square(y_pred - y).sum()
    print(t, loss)
    
    # 手搓反向传播
    grad_y_pred = 2.0 * (y_pred - y) # loss对预测值y_pred的梯度
    grad_w2 = h.T.dot(grad_y_pred) # loss对输出层权重w2的梯度 ∂loss/∂w2 = (∂loss/∂y_pred)*(∂y_pred/∂w2) 链式法则
    grad_h = grad_y_pred.dot(W2.T) # loss对隐藏层输出h的梯度 ∂y_pred/∂h = w2
    grad_w1 = x.T.dot(grad_h * h * (1 - h)) # loss对输入层权重w1的梯度 ∂loss/∂w1 = (∂loss/∂h)*(∂h/∂w1)
    
    w1 -= 1e-4 * grad_w1
    w2 -= 1e-4 * grad_w2
```

### Backprpagation

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250919091841499.png" alt="image-20250919091841499" style="zoom:80%;" />

这种计算方式很容易模块化，每个神经元要做的事情都是固定的

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250919092030380.png" alt="image-20250919092030380" style="zoom:80%;" />

有一些梯度传递的经典模式：

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250919093049352.png" alt="image-20250919093049352" style="zoom:80%;" />

以上讲解的都是标量梯度的反向传播，下面讲向量梯度，首先复习一下 **Vector to Scalar** 和 **Vector to Vector** 两种情况的梯度怎么求

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250919093909535.png" alt="image-20250919093909535" style="zoom:80%;" />

**Backprop with vectors/Matrices**

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250919094339366.png" alt="image-20250919094339366" style="zoom:80%;" />

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250919094308163.png" alt="image-20250919094308163" style="zoom:80%;" />

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250919094448903.png" alt="image-20250919094448903" style="zoom:80%;" />

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250919094526377.png" alt="image-20250919094526377" style="zoom:80%;" />

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250919094919758.png" alt="image-20250919094919758" style="zoom:80%;" />

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250919094943793.png" alt="image-20250919094943793" style="zoom:80%;" />

## Lec5 Image Classification with CNNs

全连接层做的只是把二维像素矩阵展开成一个大向量，然后做矩阵出发运算，这==**完全破坏了图像的二维结构**==

我们希望能够在**提取图像特征**的同时**尊重图像的二维结构**

### Convolutional Neural Networks

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250920145242179.png" alt="image-20250920145242179" style="zoom:80%;" />

### Convolutional Layer

首先换一种视角来看全连接层

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250920153823076.png" alt="image-20250920153823076" style="zoom:80%;" />

输出的每一个元素都是输入向量与权重矩阵的一行做**点积**得到的，而点积运算都可以视作一种**模板匹配**（两向量方向越近点积结果越大）

而**权重的每一行就是一个模板**，分类过程就是看输入向量与哪一个模板更相近，==**训练过程就是让这些模板越来越合理、准确**==

把这个思路延伸到卷积层，其实**每一个卷积核就是一个小模板**

只不过其大小不再是一整个图像的大小（比如32×32），而是一个小子图像的大小（比如3×3/5×5/7×7）

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250920160245312.png" alt="image-20250920160245312" style="zoom:80%;" />

下面详细介绍卷积层的具体行为：

保持图像三维空间结构 32×32×3，使用和输入图像的通道数相同、二维大小更小的卷积核 5×5×3 ==**遍历图像中所有相同形状的子图做点积**==

![retouch_2025092016432271](F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\retouch_2025092016432271.jpg)

这就是单个卷积核对于单个图像的处理结果：生成一个**每个元素代表原图像子图与卷积核匹配程度**的二维Tensor

后面的处理就是数据维度上的增加，多个卷积核、多个图像批次处理......

![retouch_2025092016570132](F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\retouch_2025092016570132.jpg)

注意，==**每个卷积核有一个独属于自己的bias标量，以及不同卷积核随机初始化的时候一定要不同**==

重复多次这样的卷积层操作，再加上激活函数引入非线性，就得到一个卷积网络

![image-20250920170040183](F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250920170040183.png)

注意到每层运算之后表征图像的Tensor的二维大小是在缩水的，解决方式就是 ==**Padding**==，再周围添加额外的虚拟数据0

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250920170600505.png" alt="image-20250920170600505" style="zoom:80%;" />

Common setting: K is odd, and P = (K-1)/2, so output has same size as input

随着卷积层数加深，卷积核能提取到的特征/训练形成的模板是原图像中更大的结构

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250920225011869.png" alt="image-20250920225011869" style="zoom:80%;" />

为什么会这样？因为随着层数加深，**有效感受野 (Receptive Fields)**，也就是每次卷积运算所涉及到的**原始图像**中的像素范围会增大

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250920230054122.png" alt="image-20250920230054122" style="zoom:80%;" />

潜在问题：最后的输出层我们希望可以汇总整个图像的全局信息，这就代表一张很大的图像会需要很多卷积层叠加

解决方案：通过增加卷积核在遍历时的**步长 stride**，是的感受野随着深度指数增长

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250920231528659.png" alt="image-20250920231528659" style="zoom:80%;" />

### Pooling Layer

Another way to downsample (downsample is benefit to expand perceptive field quickly, and make the calculation faster)

核心思想就是把每一个通道拉出来做处理再重新堆叠起来，最后形成通道数一样，但更小的张量

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250920232424054.png" alt="image-20250920232424054" style="zoom:80%;" />

池化 (polling) 的具体方法有点类似卷积运算，是用一个**池化核**对整个二维向量遍历

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250920233919831.png" alt="image-20250920233919831" style="zoom:80%;" />

最大池化会引入非线性，可能就不需要在该池化层附近再使用RELU激活了；但平均池化仍为线性，还是需要激活函数

## Lec6 Training CNNs and CNN Architectures

### How to build CNN?

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250925211658230.png" alt="image-20250925211658230" style="zoom:80%;" />

#### Layers in CNNs

##### Normalization Layers

核心思想：① 输入数据做归一化处理  ② 通过训练**学习**用于 scale/shift (缩放/平移) 输入数据的参数

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250925214011576.png" alt="image-20250925214011576" style="zoom:80%;" />

不同的 Normalization Layer 区别在于求均值和方差的方法不同

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250925214119376.png" alt="image-20250925214119376" style="zoom:80%;" />

Batch Norm 就是对数据中的某一个维度在一批数据中求均值和方差；Layer Norm 就是在一个数据内部对所有维度求均值和方差

##### Dropout

是CNN中的==**正则化层**==，通过**在训练时添加随机性并在测试时移除**，避免模型过拟合，提升泛化能力

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250925214626675.png" alt="image-20250925214626675" style="zoom:80%;" />

直观理解为什么这种操作是有效的：**强迫网络提取冗余的特征，避免模型过度依赖某一个特征**

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250925220343514.png" alt="image-20250925220343514" style="zoom: 80%;" />

训练时这一步通常是通过给每一层的输出加一层掩码实现的，掩码中哪些位置置为零是根据p随机的

这些通过掩码被”舍弃“的神经元在反向传播时也自然被忽略，因为梯度为0

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250925221540376.png" alt="image-20250925221540376" style="zoom:80%;" />

而在使用训练好的网络进行测试的时候，每一层的输出要乘p，保证和训练时该层输出的期望相同

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250925221830488.png" alt="image-20250925221830488" style="zoom:80%;" />

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250925221848941.png" alt="image-20250925221848941" style="zoom:80%;" />

#### Activation Functions

Sigmoid 的问题：只有中间一小段有梯度，较大的正值和负值对应的梯度都为0

ReLU 的优势：趋向正无穷的时候不会饱和；计算很简单；实验证明比 Sigmoid 收敛更快

ReLU 的问题：x < 0 时梯度为0

GELU：$f(x)=x*\Phi(x)$，其中 $\Phi(x)$ 是高斯分布的概率密度函数

![retouch_2025092600032099](F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\retouch_2025092600032099.jpg)

#### CNN Architectures

##### VGGNet (2014)

使用三层 3×3 的卷积层的感受野与使用一层 7×7 相同，但是更深（有更多激活层，引入更多非线性），并且总参数更少

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250926001245072.png" alt="image-20250926001245072" style="zoom:80%;" />

##### ResNet (2015)

实验发现仅仅在普通 CNN 网络上 不断堆叠更深的层，效果反而变差，并且不是因为过拟合

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250926001643553.png" alt="image-20250926001643553" style="zoom:80%;" />

事实是：更深的网络拥有更多的参数，所以表示特征的能力一定更强；效果反而更差的原因可能是：更深的网络在训练中更难优化

或者可以用另一种角度理解，纯粹增加深度就像下图中左侧一样，能表示的函数空间 $\mathcal{F}$ 增大，但不一定更靠近最优函数

![image-20250926084523943](F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250926084523943.png)

因此，只有当较复杂的函数类包含较小的函数类时，我们才能确保提高它们的性能

对于深度神经网络，如果我们能将新添加的层训练成==**恒等映射 (identity function) $f(x)=x$**==，新模型和原模型将同样有效。 同时，由于新模型**可能**得出更优的解来拟合训练数据集，因此添加层似乎更容易降低训练误差

基于这种思路，提出了==**残差块 (residual block)**== 这一结构

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250926085042084.png" alt="image-20250926085042084" style="zoom:80%;" />

完整的 ResNet 结构如下，注意过程中通过**通道数翻倍和长宽减半 (stride=2)** 实现在保持计算量不增大的同时**将空间信息转换成语义信息**

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250926091504750.png" alt="image-20250926091504750" style="zoom:80%;" />

#### Weight Initialization

权重初始化时过大或者过小都会出现问题，导致每一层输出的activation的均值和方差逐渐增大/减小

![retouch_2025092609453346](F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\retouch_2025092609453346.jpg)

但实际上我们**希望每层输出的activaiton的均值和方差能一直保持在同一水平**，这样更容易优化

Kaiming 提出了针对不同维度输入通用的权重初始化方法：**MSRA Initialization**

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250926092721889.png" alt="image-20250926092721889" style="zoom:80%;" />

### How to train CNNs？

#### Data Preprocessing

图像数据归一化，需要**针对数据集**预先计算每个通道的均值和方差

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250926093555198.png" alt="image-20250926093555198" style="zoom:80%;" />

#### Data Augmentation

水平/垂直翻转、随机缩放/裁切、Color Jitter、Cutout  根据任务选择合适的图像增强方法，核心目的是**增强后人眼仍可识别，但模型更难记忆**

![retouch_2025092609461844](F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\retouch_2025092609461844.jpg)

#### Transfer Learning

实践中我们可能没有像 ImageNet 那么多数据

数据集很少的时候，可以先在 ImageNet 上训一个/直接用别人训好的模型，保留前面所有层的参数，只重新训练最后一个线性层

数据集稍微多一点的时候，可以使用在 ImageNet 上预训练的参数来初始化自己的网络，然后每一层都参与训练

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250926132527813.png" alt="image-20250926132527813" style="zoom:80%;" />

当然，使用迁移学习的效果没有那么好，比如如果你尝试识别的种类中有 ImageNet 中不包含的种类，那分类效果就会较差

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250926132828754.png" alt="image-20250926132828754" style="zoom:80%;" />

#### Hyperparamater Selection

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250926132018578.png" alt="image-20250926132018578" style="zoom:80%;" />

Random search 比网格搜索更好

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20250926131922487.png" alt="image-20250926131922487" style="zoom:80%;" />

## Lec7 Recurrent Neural Networks

### Sequence Modeling

截至目前，所有网络的输入输出维度都是固定不变的，但实际运用中有很多需要可变输入输出维度的场景

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\1759501611236.png" alt="1759501611236" style="zoom:80%;" />

RNN的特殊之处在于其**循环特性**，即其具有内部隐藏状态，随着输入序列中的每一个输入不断更新

![retouch_2025100322322260](F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\retouch_2025100322322260.jpg)

![](F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\8f95ab28-2b5a-4311-a05f-9ca232402a91.png)

![fe368820-eaf5-4bea-ae2c-3d6b54149835](F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\fe368820-eaf5-4bea-ae2c-3d6b54149835.png)

### Vanilla RNN

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20251003224850377.png" alt="image-20251003224850377" style="zoom:80%;" />

手搓一个基础RNN：检测输入中连续的1，若检测到了则输出1，否则输出0

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20251003225158737.png" alt="image-20251003225158737" style="zoom:80%;" />

要实现这样的功能，RNN的隐藏状态需要储存什么信息？**上一时间步长的输入和当前输入的X值**
$$
h_t = ReLU(W_{hh}h_{t-1}+W_{xh}x_t)\\y_t=ReLU(W_{hy}h_t)\\
h_t=\begin{bmatrix}
	Current  \\
	Previous	\\
	1
\end{bmatrix}
$$

```python
w_xh = np.array([[1], [0], [0]]) # 语义为计算当前值,如果当前值为1则为隐藏状态第一项贡献一个1,否则贡献为0

w_hh = np.array([0, 0, 0], # 前一时间步的隐藏状态对新隐藏状态的current项无贡献
                [1, 0, 0], # 前一时间步隐藏状态的current称为新隐藏状态的previous
                [0, 0, 1]) 

w_hy = np.array([1, 1, -1]) # 输出 Max(Current+Previous-1, 0)

x_seq = [0, 1, 0, 1, 1, 1, 0, 1, 1]

h_t_prev = np.array([[0], [0], [1]]) # 隐藏状态初始化为[0,0,1]

for t,x in enumerate(x_seq):
    h_t = relu(w_hh @ h_t_prev + (w_xh @ x))
    y_t = relu(w_yh @ h_t)
    h_t_prev = h_t
```

如何计算梯度？

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20251102231943854.png" alt="image-20251102231943854" width=32%/><img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\ede088f0442bfa4d42afb1341e7b58fb.png" alt="ede088f0442bfa4d42afb1341e7b58fb" width="32%" hidth="60"/><img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20251102232400752.png" alt="image-20251102232400752" width="32%" />

如果一次性对一个完整序列做前向+反向传播更新，每次处理的数据量可能太大，可以把序列分为多个chunk

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20251104233118907.png" alt="image-20251104233118907" style="zoom:80%;" />

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20251104233202308.png" alt="image-20251104233202308" style="zoom:80%;" />

### Character-level Language Model

一个简单的例子，假设字母表为 [h, e, l, o]，期望训练结果为=="输入序列hell时，能预测下一个字母为o"==

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20251104234542455.png" alt="image-20251104234542455" style="zoom:80%;" />

一般不会把 one-hot 编码直接输入模型，而是先经过一个专门的 embedding layer

embedding layer 是一个D×D的大矩阵，D是可能的输入数量

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20251105000858352.png" alt="image-20251105000858352" style="zoom:80%;" />

### RNN tradeoffs

![image-20251105004147400](F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20251105004147400.png)

### Implementations

#### Image Captioning

由CNN等图像encoder提取图像特征，然后利用RNN来理解图像语义

<img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20251105004714401.png" alt="image-20251105004714401" width="32%"><img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20251105005055148.png" alt="image-20251105005055148" width="32%"/> <img src="F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20251105005201053.png" alt="image-20251105005201053" width="32%"/>

#### Visual Question Answering (VQA)

#### Visual Dialog: Conversations about images

#### Visual Language Navigation

### Multilayers RNN

多层隐藏层，每一层的输入x替换为上一层的输出y，注意每一层使用的权重不同

![image-20251105005922904](F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20251105005922904.png)

### RNN Variants: Long Short Term Memory (LSTM)

Vanilla RNN在梯度传播的时候会遇到**梯度消失/梯度爆炸**的问题

![image-20251105011026326](F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20251105011026326.png)

![image-20251105011051782](F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20251105011051782.png)

![image-20251105011119670](F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20251105011119670.png)

梯度爆炸还可以尝试用 Gradient clipping 来归一化一下，梯度消失是人们尝试改进RNN结构的主要原因

![image-20251105012339207](F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20251105012339207.png)

![image-20251105012447020](F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20251105012447020.png)

![image-20251105012517673](F:\王梓恒\学习资料\Machine_Learning\Deep_Learning\CV\CS231n\images\image-20251105012517673.png)

## Lec8 Attention and Transformers

