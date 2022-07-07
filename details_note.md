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
Soft Attention中是对于每个Encoder的Hidden State会match一个概率值，而在Hard Attention会直接找一个特定的单词概率为1，而其它对应概率为0。

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
repeat_interleave是一个元素一个元素地重复，而repeat是一组元素一组元素地重复。

# LayerNorm/BatchNorm
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

# 自回归
自回归（auto-regressive）属性，预测仅依赖于已发生的事情或已生成的信息。

GPT，Transformer，BERT都用到了自回归。

# nn.apply
apply(fn)的官网介绍，该方法会将fn递归的应用于模块的每一个子模块（.children()的结果）及其自身。典型的用法是，对一个model的参数进行初始化。

# 转置函数

## transpose() 

input (Tensor) – 输入张量，必填

dim0 (int) – 转置的第一维，默认0，可选

dim1 (int) – 转置的第二维，默认1，可选

两个维度交换

## permute()
dims (int…*)-换位顺序，必填

0，1，2.....表示真实的维度位置。

torch.transpose(x)合法， x.transpose()合法。

tensor.permute(x)不合法，x.permute()合法。

## contiguous()
经常有人用view()函数改变通过转置后的数据结构，导致报错
RuntimeError: invalid argument 2: view size is not compatible with input tensor's....

这是因为tensor经过转置后数据的内存地址不连续导致的,也就是tensor .is_contiguous()==False
这时候reshape()可以改变该tensor结构，但是view()不可以

调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一毛一样，x.reshape(3,4) 这个操作等于x = x.contiguous().view()。