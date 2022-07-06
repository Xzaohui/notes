# attention

seq2seq + attention 的 attention 的 key 与 value  是相同的，都是解码器的输出，但是在其他框架中就不一定了
## Bahdanau Attention Mechanism
传统seq2seq模型中encoder将输入序列编码成一个context向量，decoder将context向量作为初始隐状态，生成目标序列。随着输入序列长度的增加，编码器难以将所有输入信息编码为单一context向量，编码信息缺失，难以完成高质量的解码。

![Alt](https://img-blog.csdnimg.cn/20200613120415222.png)

Bahdanau本质是一种 加性attention机制，将decoder的隐状态和encoder所有位置输出通过线性组合对齐，得到context向量，用于改善序列到序列的翻译模型。

本质：两层全连接网络，隐藏层激活函数tanh，输出层维度为1。

时刻 $t$, 解码器的隐状态表示为
$$
\boldsymbol{s}_{t}=f\left(\boldsymbol{s}_{t-1}, \boldsymbol{c}_{t}, y_{t-1}\right)
$$
时刻 $t$ 的隐状态 $\boldsymbol{s}_{t-1}$ 对 编码器 $Q$ 各时刻输出 $X$ 的注意力分数为:
$$
\boldsymbol{\alpha}_{t}\left(\boldsymbol{s}_{t-1}, X\right)=\operatorname{softmax}\left(\tanh \left(\boldsymbol{s}_{t-1} W_{\text {decoder }}+X W_{\text {encoder }}\right) W_{\text {alignment }}\right), \quad \boldsymbol{c}_{t}=\sum_{i} \alpha_{t i} \boldsymbol{x}_{i}
$$
X包含了前面所有的编码器输出，$\boldsymbol{x}_{i}$是编码器的输出，$\boldsymbol{c}_{t}$是解码器的输出。每个词的encoder隐状态作为Key和value，两者是一样的。
## Luong Attention Mechanism
![Alt](https://img-blog.csdnimg.cn/20200613140535354.png)
Luong本质是一种 乘性attention机制，将解码器隐状态和编码器输出进行矩阵乘法，得到上下文向量。


## 区别 
1. 注意力的计算方式不同
在 Luong Attention 机制中, 第 $\mathrm{t}$ 步的注意力 $\mathbf{c}_{t}$ 是由 decoder 第 $\mathrm{t}$ 步的 hidden state $\mathbf{h}_{t}$ 与 encoder 中的每一个 hidden state $\overline{\mathbf{h}}_{s}$ 加权计算得出的。而在 Bahdanau Attention 机制中, 第 $\mathrm{t}$ 步的注意力 $\mathbf{c}_{t}$ 是由 decoder 第 $\mathrm{t}-1$ 步的 hidden state $\mathbf{h}_{t-1}$ 与 encoder 中的每一个 hidden state $\overline{\mathbf{h}}_{s}$ 加权计算得出的。
2. decoder 的输入输出不同
在 Bahdanau Attention 机制中, decoder 在第 $\mathrm{t}$ 步时, 输入是由注意力 $\mathbf{c}_{t}$ 与前一步的 hidden state $\mathbf{h}_{t-1}$ 拼接 (concatenate) 得出的, 得到第 $\mathrm{t}$ 步的 hidden state $\mathbf{h}_{t}$ 并直接输出 $\hat{\mathbf{y}}_{t+1}$ 。 而 Luong Attention 机制在 decoder 部分建立了一层额外的网络结构, 以注意力 $\mathbf{c}_{t}$ 与原 decoder 第 $\mathrm{t}$ 步的 hidden state $\mathbf{h}_{t}$ 拼接作为输入, 得到第 $\mathrm{t}$ 步的 hidden state $\tilde{\mathbf{h}}_{t}$ 并输出 $\hat{\mathbf{y}}_{t}$

![Alt](https://pic4.zhimg.com/80/v2-29d59672c917a1b964aea8297b2f9b6b_1440w.jpg)

## soft attention / hard attention
Soft Attention中是对于每个Encoder的Hidden State会match一个概 率值，而在Hard Attention会直接找一个特定的单词概率为1，而其它对应概率为0。

## local attention
用编码器与解码器输出最相关联的时刻的附近的D的区间内值来做attention，减少了参数量。

# pytorch
## ModuleList/Sequential

pytorch可以自动识别nn.ModuleList中的参数而普通的list则不可以。

Sequential定义的网络中各层会按照定义的顺序进行级联，因此需要保证各层的输入和输出之间要衔接。并且nn.Sequential实现了farward()方法，因此可以直接通过类似于x=self.combine(x)的方式实现forward。

而nn.ModuleList则没有顺序性要求，并且也没有实现forward()方法，需要重新自己定义。

```py
class testNet(nn.Module):
    def __init__(self):
        super(testNet, self).__init__()
        self.combine = nn.Sequential(
            nn.Linear(100,50),
            nn.Linear(50,25),
        ) 
    def forward(self, x):
        x = self.combine(x)
        return x

class testNet(nn.Module):
    def __init__(self):
        super(testNet, self).__init__()
        self.combine = nn.ModuleList()
        self.combine.append(nn.Linear(100,50))
        self.combine.append(nn.Linear(50,25))
    #重新定义forward()方法
    def forward(self, x):
        x = self.combine[0](x)
        x = self.combine[1](x)
        return x
```

## repeat
repeat_interleave是一个元素一个元素地重复，而repeat是一组元素一组元素地重复.
参数(a,b,c)从c到a进行复制，从低维到高维。

## LayerNorm/BatchNorm
LayerNorm是一个层规范化，而BatchNorm是一个批量规范化。

参数为num_features，即通道数/（句子*）向量维度。

LayerNorm考虑低维（句子*）向量的归一化，而BatchNorm考虑高维每个batch内同一通道的的归一化。

$$
out =\frac{x-\operatorname{mean}[\text { data, features }]}{\sqrt{\text { Var }[\text { data, features }]}+\epsilon} * gamma + beta
$$

LayerNorm (num_features=data.shape(axis=-1), epsilon= 1e-05, center-True, scale= True, beta _initializer=zeros, gamma_initializer='ones, in_channels=O, prefix=None, params=None)

BatchNorm (num_features=data.shape(axis=1)=channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, in_channels=O, prefix=None, params=None)

# finetune
在源数据集（例如ImageNet数据集）上预训练神经网络模型，即源模型。

创建一个新的神经网络模型，即目标模型。这将复制源模型上的所有模型设计及其参数（输出层除外）。我们假定这些模型参数包含从源数据集中学到的知识，这些知识也将适用于目标数据集。我们还假设源模型的输出层与源数据集的标签密切相关；因此不在目标模型中使用该层。

向目标模型添加输出层，其输出数是目标数据集中的类别数。然后随机初始化该层的模型参数。

在目标数据集（如椅子数据集）上训练目标模型。输出层将从头开始进行训练，而所有其他层的参数将根据源模型的参数进行微调。

ResNet，Bert都用到了微调。


# 损失函数总结

## 0-1损失函数(zero-one loss)
$L(Y, f(X))=\left\{\begin{array}{l}
1, Y \neq f(X) \\
0, Y=f(X)
\end{array}\right.$

0-1损失函数直接对应分类判断错误的个数，但是它是一个非凸函数，不太适用.

## 绝对值损失函数
绝对值损失函数是计算预测值与目标值的差的绝对值：

$L(Y, f(x))=|Y-f(x)|$

## log对数损失函数

$L(Y, P(Y \mid X))=-\log P(Y \mid X)$

(1) log对数损失函数能非常好的表征概率分布，在很多场景尤其是多分类，如果需要知道结果属于每个类别的置信度，那它非常适合。

(2)健壮性不强，相比于hinge loss对噪声更敏感。

(3)逻辑回归的损失函数就是log对数损失函数。

## 平方损失函数

$L(Y \mid f(X))=\sum_{N}(Y-f(X))^{2}$

经常应用与回归问题

## 指数损失函数

$L(Y \mid f(X))=\exp [-y f(x)]$

对离群点、噪声非常敏感。经常用在AdaBoost算法中。

## Hinge 损失函数

$L(y, f(x))=\max (0,1-y f(x))$

(1)hinge损失函数表示如果被分类正确, 损失为 0 , 否则损失就为  $1-y f(x)$。SVM就是使用这 个损失函数。

(2)一般的  $f(x)$  是预测值, 在-1到1之间，  y  是目标值(-1或1)。其含义是,  f(x)  的值在-1和+1 之间就可以了, 并不鼓励  $|f(x)|>1$ , 即并不鼓励分类器过度自信, 让某个正确分类的样本距离 分割线超过1并不会有任何奖励, 从而使分类器可以更专注于整体的误差。

(3) 健壮性相对较高, 对异常点、噪声不敏感, 但它没太好的概率解释。

$\begin{array}{ll}
\operatorname{argmin} & \frac{1}{2}||w||^{2}+C \sum_{i} \xi_{i} \\
\text {st. } \quad & y_{i}\left(w^{T} x_{i}+b\right) \geq 1-\xi_{i} \\
& \xi_{i} \geq 0 \\
& \xi_{i} \geq 1-y_{i}\left(w^{T} x_{i}+b\right)
\end{array}$

$\begin{aligned}
J(w)&=\frac{1}{2}|| w||^{2}+C \sum_{i} \max \left(0,1-y_{i}\left(w^{T} x_{i}+b\right)\right) \\
&=\frac{1}{2}|| w||^{2}+C \sum_{i} \max (0,1-Y f(x)) \\
&=\frac{1}{2}|| w||^{2}+C \sum_{i} L_{H i n g e}
\end{aligned}$

SVM的损失函数可以看做是L2正则化与Hinge loss之和。

## 交叉熵损失函数

交叉熵的标准形式如下:

$H(p, q)=-\sum_{x}(p(x) \log q(x)+(1-p(x)) \log (1-q(x)))$

而我们在pytorch中使用的交叉熵损失函数形式如下:

$H(p, q)=-\sum_{x}(p(x) \log q(x)$

Pytorch中CrossEntropyLoss()函数的主要是将softmax-log-NLLLoss合并到一块得到的结果。

RNN输出的概率矩阵->soft-max归一化为概率q(x)->标签化为one-hot编码->负对数似然损失函数->交叉熵损失函数

即$CrossEntropyLoss=-\frac{1}{N}\sum_{i=1}^N yi(log(softmax(RNN.out)))$

yi是实际标签的one-hot编码后的每一项，softmax(RNN.out)是RNN输出的概率矩阵。

![Alt](https://pic4.zhimg.com/80/v2-ac627eab5f07ead5144cfaaff7f2163b_1440w.jpg)

## 其他概念
-损失函数：用于衡量'单个样本点'预测值与实际值的偏离程度。

-风险函数：用于衡量'样本点平均意义'下的好坏，就是说要除以batch_size。

-风险函数分为经验风险和结构风险。

-经验风险：指预测结果和实际结果的差别。

-结构风险：指经验风险 + 正则项。

-风险函数是训练过程中的模型，对已知训练数据的计算。可以理解为是train过程的loss。

-泛化函数：指模型对未知数据的预测能力。

-泛化函数是训练好的模型，对未知数据的计算。可以理解为是test过程的loss。

# 深度学习优化算法对比

SGD -> SGDM -> NAG ->AdaGrad -> AdaDelta -> Adam -> Nadam -> AdamW

一阶动量为过去各个时刻历史梯度的线性组合，而二阶动量自然是过去各个时刻历史梯度的平方的线性组合。

一阶动量来给梯度下降法加入惯性（即，越陡的坡可以允许跑得更快些）。后来，在引入二阶动量之后，才真正意味着“自适应学习率”优化算法时代的到来。

## 总体定义
计算目标函数关于当前参数的梯度:  $g_{t}=\nabla f\left(w_{t}\right)$

根据历史梯度计算一阶动量和二阶动量:  $m_{t}=\phi\left(g_{1}, g_{2}, \cdots, g_{t}\right) ; v_{t}=\psi\left(g_{1}, g_{2}, \cdots, g_{t}\right)$

计算当前时刻的下降梯度:  $\eta_{t}=\alpha \cdot m_{t} / \sqrt{V_{t}}$

根据下降梯度进行更新:  $w_{t+1}=w_{t}-\eta_{t}$ 

## 批量梯度下降法 (Batch Gradient Descent)
批量梯度下降法，是梯度下降法最常用的形式，具体做法也就是在更新参数时使用所有的样本来进行更新

## 随机梯度下降法（Stochastic Gradient Descent）
$\mathrm{SGD}$  没有动量的概念, 即:  $m_{t}=g_{t}$ 
此时,  $\eta_{t}=\alpha \cdot g_{t}$ 

随机梯度下降随机体现在选择一定量的样本（比如一个batch）来更新参数。

SGD 最大的缺点是下降速度慢，而且可能会在沟壑的两边持续震荡，停留在一个局部最优点。

## SGD with Momentum

为了抑制 SGD 的震荡，SGDM 引入了惯性，即一阶动量。如果发现是陡坡，就用惯性跑得快一些，一阶动量变为各个时刻梯度方向的指数移动平均值。此时：
$m_{t}=\beta_{1} \cdot m_{t-1}+\left(1-\beta_{1}\right) \cdot g_{t}$

$\beta_{1}$的经验值为 0.9 ，也就是每次下降时更偏向于此前累积的下降方向，并稍微偏向于当前下降方向。

## SGD with Nesterov Acceleration
SGD 还有一个问题是会被困在一个局部最优点里。就像被一个小盆地周围的矮山挡住了视野，看不到更远的更深的沟壑。因此，我们不能停留在当前位置去观察未来的方向，而要向前一步、多看一步、看远一些。

Nesterov 提出了一个方法是既然我们有了动量，那么我们可以在步骤 1 中先不考虑当前的梯度。每次决定下降方向的时候先按照一阶动量的方向走一步试试，然后在考虑这个新地方的梯度方向。此时的梯度就变成了：
$g_{t}=\nabla f\left(w_{t}-\alpha \cdot m_{t-1}\right)$
我们用这个梯度带入 SGDM 中计算 mt 的式子里去。



## AdaGrad

想法是这样：神经网络中有大量的参数，对于经常更新的参数，我们已经积累了大量关于它们的知识，不希望它们被单个样本影响太大，希望学习速率慢一些；而对于不经常更新的参数，我们对于它们了解的信息太少，希望能从每个偶然出现的样本身上多学一些，学习速率大一些。

那么怎么样去度量历史更新频率呢? 采用二阶动量一一该维度上, 所有梯度值的平方和:  $v_{t}=\sum_{\tau=1}^{t} g_{\tau}^{2}$  。
回顾步骤 3 中的下降梯度:  $\eta_{t}=\alpha \cdot m_{t} / \sqrt{V_{t}}$  。我们发现引入二阶动量的意义上给了学习率一个缩放比例, 从而达到 了自适应学习率的效果（Ada = Adaptive）。（一般为了防止分母为 0 , 会对二阶动量做一个平滑，分母加入一个微小数值$\delta$。）
这个方法有一个问题是, 因为二阶动量是单调递增的, 所以学习率会很快减至 0 , 这样可能会使训练过程提前结束。

## AdaDelta / RMSProp

由于 AdaGrad 的学习率单调递减太快，我们考虑改变二阶动量的计算策略：不累计全部梯度，只关注过去某一窗口内的梯度。这个就是名字里 Delta 的来历。

修改的思路很直接，前面我们说过，指数移动平均值大约是过去一段时间的平均值，是一种给予近期数据更高权重的平均方法。因此我们用这个方法来计算二阶累积动量,$\beta_{2}$的经验值为 0.999：
$v_{t}=\beta_{2}v_{t-1}+\left(1-\beta_{2}\right)g_{t}^{2}$

## Adam

Adam = Adaptive + Momentum。

$m_{t}=\beta_{1} \cdot m_{t-1}+\left(1-\beta_{1}\right) \cdot g_{t}$

$v_{t}=\beta_{2} \cdot v_{t-1}+\left(1-\beta_{2}\right)g_{t}^{2}$

$\hat{m}_{t} \leftarrow \frac{m_{t}}{1-\beta_{1}^{t}}$ 偏差校正的一阶矩估计    
$\hat{v}_{t} \leftarrow \frac{v_{t}}{1-\beta_{2}^{t}}$ 偏差校正的二阶矩估计  

$w_{t+1} \leftarrow w_{t}-\alpha \cdot \frac{\hat{m}_{t}}{\epsilon+\sqrt{\hat{v}_{t}}}$

初始的学习率$\alpha$乘以梯度均值与梯度方差的平方根之比。其中默认学习率$\alpha=0.001 , \epsilon=10^{-8}$。对更新的步长计算，能够从梯度均值及梯度平方两个角度进行自适应地调节，而不是直接由当前梯度决定。

### 优点

实现简单，计算高效，对内存需求少

参数的更新不受梯度的伸缩变换影响

超参数具有很好的解释性，且通常无需调整或仅需很少的微调

更新的步长能够被限制在大致的范围内（初始学习率）

能自然地实现步长退火过程（自动调整学习率）

很适合应用于大规模的数据及参数的场景

适用于不稳定目标函数

适用于梯度稀疏或梯度存在很大噪声的问题

### 问题

1. 可能不收敛

    SGD没有用到二阶动量，因此学习率是恒定的（实际使用过程中会采用学习率衰减策略，因此学习率递减）。AdaGrad的二阶动量不断累积，单调递增，因此学习率是单调递减的。因此，这两类算法会使得学习率不断递减，最终收敛到0，模型也得以收敛。

    AdaDelta和Adam二阶动量是固定时间窗口内的累积，随着时间窗口的变化，遇到的数据可能发生巨变，使得$v_{t}$可能会时大时小，不是单调变化。这就可能在训练后期引起学习率的震荡，导致模型无法收敛。

2. 可能错过全局最优解
   自适应学习率算法可能会对前期出现的特征过拟合，后期才出现的特征很难纠正前期的拟合效果。后期Adam的学习率太低，影响了有效的收敛。前期用Adam，享受Adam快速收敛的优势；后期切换到SGD，慢慢寻找最优解。

## Nadam
Nadam=Nesterov + Adam 

$g_{t}=\nabla f\left(w_{t}-\alpha \cdot m_{t-1} / (\sqrt{v_{t}}+\delta) \right)$

# 混淆矩阵
## 基本概念
准确率、精确率、查准率、查全率、真阳性率、假阳性率、ROC、AUC、PRC、KS、F1

True Positive-TP： 预测值为1，预测结果正确，即TP

False Positive-FP： 预测值为1，预测结果错误，即FP

False Negative-TN： 预测值为0，预测结果错误，即FN

True Negative-FN：预测值为0，预测结果正确，即TN

$A c c u r a c y=\frac{T P+T N}{A L L}$

$Precision =\frac{T P}{T P+F P}$

$Recall =\frac{T P}{T P+F N}$

## ROC曲线 AUC面积

![Alt](https://pic1.zhimg.com/80/v2-592f08ddbf22c4efbdbe714a1c99b9d1_1440w.jpg?source=1940ef5c)

![Alt](https://pic1.zhimg.com/80/v2-43ed3d276eebb475af0c7645c2452aa7_1440w.jpg?source=1940ef5c)

![Alt](https://pic1.zhimg.com/80/v2-83f1a3fa2db8c707df046bd819ce7e8c_1440w.jpg?source=1940ef5c)

TPR越高，同时FPR越低（即ROC曲线越陡），那么模型的性能就越好

AUC (Area Under Curve) 被定义为ROC曲线下的面积，完全随机的二分类器的AUC为0.5，因此与之相对的相对面积更大，更靠近左上角的曲线代表着一个更加稳健的二分类器。

### 计算

根据AUC的统计意义，我们可以通过计算逆序对数来计算AUC。假设真实label和预测label是以(true_label, predicted_label)的形式保存，那么我们可以排序true label来统计predicted label的逆序对数（下面法1），也可以排序predicted label来统计true label的逆序对数（下面法2）：

#### 法1
对真实label排序，统计预测label的逆序对数

在有M个正样本,N个负样本的数据集里。一共有M*N对样本（一对样本即，一个正样本与一个负样本，注意这里的定义！他不是任意抽两个样本！）。统计这M*N对样本里，正样本的预测概率大于负样本的预测概率的个数。

$$
\frac{\sum I\left(P_{\text {positive }}, P_{\text {negtive }}\right)}{M \times N}
$$

其中
$$
I\left(P_{\text {positive }}, P_{\text {negtive }}\right)=\left\{\begin{array}{l}1, P_{\text {positive }}>P_{\text {positive }} \\ 0.5, P_{\text {positice }}=P_{\text {negtive }} \\ 0, P_{\text {positive }}<P_{\text {negtive }}\end{array}\right.
$$

1. 统计所有正样本个数P，负样本个数N；
2. 遍历所有正负样本对，统计正样本预测值大于负样本预测值的样本总个数number
3. AUC = number / (P * N)
4. 一些计算细节是当正负样本预测值刚好相等时，该样本记为0.5个。

#### 法2
首先对score从小到大排序，然后令最小score对应的sample的rank为1，第二小score对应sample的rank为2，以此类推。然后把所有的正类样本的rank相加，再减去两个正样本组合的情况（因为AUC要求是抽取一个正样本和一个负样本，正样本pred排在负样本前的概率，所以两个都是正样本这种情况要排除）。得到的就是所有的样本中有多少对正类样本的score大于负类样本的score。然后再除以M×N。

$A U C=\frac{\sum_{\mathrm{i} \in \text { positiveClass }} \operatorname{ran}_{i}-\frac{M(1+M)}{2}}{M \times N}$

首先解释一下rank的含义，我们是pred从小到大排序，当前数的rank就代表当前数与其排在前面的数能够形成多少对。那么显然如果这里是正样本，它形成的就是正序对或者“正样本-正样本”对（因为我们最小的秩是1，但形成的对数是0，因此秩也包括其“自身-自身”对）（“正样本-正样本”和“自身-自身”后面会被减掉）。如果是负样本，形成的就是逆序对或者“负样本-负样本”对，这显然对计算auc没用，舍去，所以求和符号只会对正样本的rank求和。

那么如何减去“正样本-正样本”和“自身-自身”对的个数呢。我们记正样本的个数为M，那么第1小的正样本会形成这样的对数1对（“自身-自身”），第2小的正样本形成2对......，总共形成1+2+3+...+M对，即M(M+1)/2，这是等差数列求和公式。这也是为什么后面要减去这一项。




gini=2AUC-1

## F1/PR曲线
PR曲线中的P代表的是precision（精准率），R代表的是recall（召回率），其代表的是精准率与召回率的关系，一般情况下，将recall设置为横坐标，precision设置为纵坐标。

![Alt](https://img-blog.csdnimg.cn/20200813084123107.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2d1emhhbzk5MDE=,size_16,color_FFFFFF,t_70#pic_center)

F1分数同时考虑了查准率和查全率，让二者同时达到最高，取一个平衡。

$F_{1}=\frac{2}{\frac{1}{\text { precision }}+\frac{1}{\text { recall }}}=2 \frac{\text { precision } \times \text { recall }}{\text { precision }+\text { recall }}$

## 总结
![Alt](https://pic2.zhimg.com/80/v2-30d2be27555b2062e85e755bd25c46a6_1440w.jpg?source=1940ef5c)

