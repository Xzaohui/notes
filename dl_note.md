
# 注意力机制

## 注意力评分函数
注意力评分函数（attention scoring function）， 简称评分函数（scoring function）， 然后把这个函数的输出结果输入到softmax函数中进行运算。 通过上述步骤，我们将得到与键对应的值的概率分布（即注意力权重）。 最后，注意力汇聚的输出就是基于这些注意力权重的值的加权和。

![Alt](https://zh.d2l.ai/_images/attention-output.svg)

用数学语言描述，假设有一个查询 $\mathbf{q} \in \mathbb{R}^{q}$ 和 $m$ 个"键一值"对 $\left(\mathbf{k}_{1}, \mathbf{v}_{1}\right), \ldots,\left(\mathbf{k}_{m}, \mathbf{v}_{m}\right)$, 其中 $\mathbf{k}_{i} \in \mathbb{R}^{k}, \mathbf{v}_{i} \in \mathbb{R}^{v}$ 。 注意 力汇聚函数 $f$ 就被表示成值的加权和：
$$
f\left(\mathbf{q},\left(\mathbf{k}_{1}, \mathbf{v}_{1}\right), \ldots,\left(\mathbf{k}_{m}, \mathbf{v}_{m}\right)\right)=\sum_{i=1}^{m} \alpha\left(\mathbf{q}, \mathbf{k}_{i}\right) \mathbf{v}_{i} \in \mathbb{R}^{v},
$$
其中查询 $\mathbf{q}$ 和键 $\mathbf{k}_{i}$ 的注意力权重（标量）是通过注意力评分函数 $a$ 将两个向量映射成标量, 再经过softmax运算得到的:
$$
\alpha\left(\mathbf{q}, \mathbf{k}_{i}\right)=\operatorname{softmax}\left(a\left(\mathbf{q}, \mathbf{k}_{i}\right)\right)=\frac{\exp \left(a\left(\mathbf{q}, \mathbf{k}_{i}\right)\right)}{\sum_{j=1}^{m} \exp \left(a\left(\mathbf{q}, \mathbf{k}_{j}\right)\right)} \in \mathbb{R} .
$$
正如我们所看到的, 选择不同的注意力评分函数 $a$ 会导致不同的注意力汇聚操作。在本节中, 我们将介绍两个流行的评分 函数, 稍后将用他们来实现更复杂的注意力机制。
### 掩蔽softmax操作
正如上面提到的，softmax操作用于输出一个概率分布作为注意力权重。 在某些情况下，并非所有的值都应该被纳入到注意力汇聚中。 例如，为了在 9.5节中高效处理小批量数据集， 某些文本序列被填充了没有意义的特殊词元。 为了仅将有意义的词元作为值来获取注意力汇聚， 我们可以指定一个有效序列长度（即词元的个数）， 以便在计算softmax时过滤掉超出指定范围的位置。 通过这种方式，我们可以在下面的masked_softmax函数中 实现这样的掩蔽softmax操作（masked softmax operation）， 其中任何超出有效长度的位置都被掩蔽并置为0。

### 加性注意力
一般来说, 当查询和键是不同长度的矢量时, 我们可以使用加性注意力作为评分函数。 给定查询 $\mathbf{q} \in \mathbb{R}^{q}$ 和 键 $\mathbf{k} \in \mathbb{R}^{k}$, 加性注意力 (additive attention) 的评分函数为
$$
a(\mathbf{q}, \mathbf{k})=\mathbf{w}_{v}^{\top} \tanh \left(\mathbf{W}_{q} \mathbf{q}+\mathbf{W}_{k} \mathbf{k}\right) \in \mathbb{R},
$$
其中可学习的参数是 $\mathbf{W}_{q} \in \mathbb{R}^{h \times q} 、 \mathbf{W}_{k} \in \mathbb{R}^{h \times k}$ 和 $\mathbf{w}_{v} \in \mathbb{R}^{h}$ 。将查询和键连结起来后输入到一个多层 感知机 (MLP) 中, 感知机包含一个隐藏层, 其隐藏单元数是一个超参数 $h_{\circ}$ 通过使用 $\tanh$ 作为激活函数, 并且禁用偏置项。

```py
class AdditiveAttention(nn.Module):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        # 在维度扩展后，
        # queries的形状：(batch_size，查询的个数，1，num_hidden)
        # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
        # 使用广播方式进行求和
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
        # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
        scores = self.w_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        return torch.bmm(self.dropout(self.attention_weights), values)
```

### 缩放点积注意力
使用点积可以得到计算效率更高的评分函数，但是点积操作要求查询和键具有相同的长度 $d$ 。假设查询和键的所有元素都 是独立的随机变量, 并且都满足零均值和单位方差, 那么两个向量的点积的均值为 0 , 方差为 $d \circ$ 为确保无论向量长度如 何, 点积的方差在不考虑向量长度的情况下仍然是 1 , 我们将点积除以 $\sqrt{d}$, 则缩放点积注意力 (scaled dot-product attention）评分函数为：
$$
a(\mathbf{q}, \mathbf{k})=\mathbf{q}^{\top} \mathbf{k} / \sqrt{d} .
$$
在实践中, 我们通常从小批量的角度来考虑提高效率, 例如基于 $n$ 个查询和 $m$ 个键一值对计算注意力, 其中查询和键的 长度为 $d$, 值的长度为 $v_{\text {。 }}$ 查询 $\mathbf{Q} \in \mathbb{R}^{n \times d}$ 、键 $\mathbf{K} \in \mathbb{R}^{m \times d}$ 和 值 $\mathbf{V} \in \mathbb{R}^{m \times v}$ 的缩放点积注意力是:
$$\operatorname{softmax}\left(\frac{\mathbf{Q K}}{\sqrt{d}}\right) \mathbf{V} \in \mathbb{R}^{n \times v}$$

## 多头注意力
在实践中，当给定相同的查询、键和值的集合时， 我们希望模型可以基于相同的注意力机制学习到不同的行为， 然后将不同的行为作为知识组合起来， 捕获序列内各种范围的依赖关系 （例如，短距离依赖和长距离依赖关系）。 因此，允许注意力机制组合使用查询、键和值的不同子空间表示（representation subspaces）可能是有益的。

为此，与其只使用单独一个注意力汇聚， 我们可以用独立学习得到的组不同的 线性投影（linear projections）来变换查询、键和值。 然后，这组变换后的查询、键和值将并行地送到注意力汇聚中。 最后，将这个注意力汇聚的输出拼接在一起， 并且通过另一个可以学习的线性投影进行变换， 以产生最终输出。 这种设计被称为多头注意力（multihead attention）。 对于个注意力汇聚输出，每一个注意力汇聚都被称作一个头（head）。展示了使用全连接层来实现可学习的线性变换的多头注意力。

![Alt](https://zh.d2l.ai/_images/multi-head-attention.svg)

在实现多头注意力之前, 让我们用数学语言将这个模型形式化地描述出来。 给定查询 $\mathbf{q} \in \mathbb{R}^{d_{q}}$ 、键 $\mathbf{k} \in \mathbb{R}^{d_{k}}$ 和 值 $\mathbf{v} \in \mathbb{R}^{d_{v}}$ ，每个注意力头 $\mathbf{h}_{i}(i=1, \ldots, h)$ 的计算方法为：
$$
\mathbf{h}_{i}=f\left(\mathbf{W}_{i}^{(q)} \mathbf{q}, \mathbf{W}_{i}^{(k)} \mathbf{k}, \mathbf{W}_{i}^{(v)} \mathbf{v}\right) \in \mathbb{R}^{p_{v}},
$$
其中, 可学习的参数包括 $\mathbf{W}_{i}^{(q)} \in \mathbb{R}^{p_{q} \times d_{q}} 、 \mathbf{W}_{i}^{(k)} \in \mathbb{R}^{p_{k} \times d_{k} \text { 和 }} \mathbf{W}_{i}^{(v)} \in \mathbb{R}^{p_{v} \times d_{v}}$, 以及代表注意力汇聚的函数 $f_{\circ} f$ 可 以是 $10.3$ 节中的 加性注意力和缩放点积注意力。多头注意力的输出需要经过另一个线性转换, 它对应着 $h$ 个头连结后的 结果, 因此其可学习参数是 $\mathbf{W}_{o} \in \mathbb{R}^{p_{o} \times h p_{v}}$ :
$$
\mathbf{W}_{o}\left[\begin{array}{c}
\mathbf{h}_{1} \\
\vdots \\
\mathbf{h}_{h}
\end{array}\right] \in \mathbb{R}^{p_{o}} .
$$
基于这种设计, 每个头都可能会关注输入的不同部分, 可以表示比简单加权平均值更复杂的函数。

![Alt](picture/5.png)

## 自注意力机制 self attention
给定一个由词元组成的输入序列 $\mathbf{x}_{1}, \ldots, \mathbf{x}_{n}$, 其中任意 $\mathbf{x}_{i} \in \mathbb{R}^{d}(1 \leq i \leq n)$。该序列的自注意力输出为一个长度相同 的序列 $\mathbf{y}_{1}, \ldots, \mathbf{y}_{n}$, 其中:
$$
\mathbf{y}_{i}=f\left(\mathbf{x}_{i},\left(\mathbf{x}_{1}, \mathbf{x}_{1}\right), \ldots,\left(\mathbf{x}_{n}, \mathbf{x}_{n}\right)\right) \in \mathbb{R}^{d}
$$


![Alt](https://img-blog.csdnimg.cn/20190802192736772.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzc1ODU1MQ==,size_16,color_FFFFFF,t_70)

![Alt](https://img-blog.csdnimg.cn/20190802202927777.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzc1ODU1MQ==,size_16,color_FFFFFF,t_70)

### 位置编码
在处理词元序列时, 循环神经网络是逐个的重复地处理词元的, 而自注意力则因为并行计算而放弃了顺序操作。为了使 用序列的顺序信息，我们通过在输入表示中添加 位置编码（positional encoding）来注入绝对的或相对的位置信息。位 置编码可以通过学习得到也可以直接固定得到。 接下来，我们描述的是基于正弦函数和余弦函数的固定位置编码 [Vaswani et al.,2017]。
假设输入表示 $\mathbf{X} \in \mathbb{R}^{n \times d}$ 包含一个序列中 $n$ 个词元的 $d$ 维嵌入表示。 位置编码使用相同形状的位置嵌入矩阵 $\mathbf{P} \in \mathbb{R}^{n \times d}$ 输 出 $\mathbf{X}+\mathbf{P}$, 矩阵第 $i$ 行、第 $2 j$ 列和 $2 j+1$ 列上的元素为：
$$
\begin{aligned}
p_{i, 2 j} &=\sin \left(\frac{i}{10000^{2 j / d}}\right) \\
p_{i, 2 j+1} &=\cos \left(\frac{i}{10000^{2 j / d}}\right)
\end{aligned}
$$
由：
$$cos(x)*cos(y)+sin(x)sin(y)=cos(x-y) $$
可知，两个位置编码的点积将会得到相对位置的信息。
$$p_m.p_n^T=\sum_{j=0}^{d/2} cos((m-n)\frac{1}{10000^{2j/d}}) $$
除了捕获绝对位置信息之外, 上述的位置编码还允许模型学习得到输入序列中相对位置信息。 这是因为对于任何确定的 位置偏移 $\delta$, 位置 $i+\delta$ 处 的位置编码可以线性投影位置 $i$ 处的位置编码来表示。
这种投影的数学解释是, 令 $\omega_{j}=1 / 10000^{2 j / d}$, 对于任何确定的位置偏移 $\delta$, (10.6.2).中的任何一对 $\left(p_{i, 2 j}, p_{i, 2 j+1}\right)$ 都可以 线性投影到 $\left(p_{i+\delta, 2 j}, p_{i+\delta, 2 j+1}\right)$ :
$$
\begin{aligned}
& {\left[\begin{array}{cc}
\cos \left(\delta \omega_{j}\right) & \sin \left(\delta \omega_{j}\right) \\
-\sin \left(\delta \omega_{j}\right) & \cos \left(\delta \omega_{j}\right)
\end{array}\right]\left[\begin{array}{c}
p_{i, 2 j} \\
p_{i, 2 j+1}
\end{array}\right] } \\
=& {\left[\begin{array}{c}
\cos \left(\delta \omega_{j}\right) \sin \left(i \omega_{j}\right)+\sin \left(\delta \omega_{j}\right) \cos \left(i \omega_{j}\right) \\
-\sin \left(\delta \omega_{j}\right) \sin \left(i \omega_{j}\right)+\cos \left(\delta \omega_{j}\right) \cos \left(i \omega_{j}\right)
\end{array}\right] } \\
=& {\left[\begin{array}{c}
\sin \left((i+\delta) \omega_{j}\right) \\
\cos \left((i+\delta) \omega_{j}\right)
\end{array}\right] } \\
=& {\left[\begin{array}{c}
p_{i+\delta, 2 j} \\
p_{i+\delta, 2 j+1}
\end{array}\right] }
\end{aligned}
$$
$2 \times 2$ 投影矩阵不依赖于任何位置的索引 $i_{\circ}$

### 小结
在自注意力中，查询、键和值都来自同一组输入。

卷积神经网络和自注意力都拥有并行计算的优势，而且自注意力的最大路径长度最短。但是因为其计算复杂度是关于序列长度的二次方，所以在很长的序列中计算会非常慢。

为了使用序列的顺序信息，我们可以通过在输入表示中添加位置编码，来注入绝对的或相对的位置信息。


## transformer

### 基本结构
![Alt](https://zh.d2l.ai/_images/transformer.svg)

从宏观角度来看, transformer的编码器是由多个相同的层叠加而成的, 每个层 都有两个子层（子层表示为sublayer）。第一个子层是多头自注意力（multi-head self-attention）汇聚；第二个子层是基 于位置的前馈网络（positionwise feed-forward network）。具体来说, 在计算编码器的自注意力时, 查询、键和值都来 自前一个编码器层的输出。受 $7.6$ 节中残差网络的启发, 每个子层都采用了残差连接（residual connection）。在 transformer中, 对于序列中任何位置的任何输入 $\mathbf{x} \in \mathbb{R}^{d}$, 都要求满足sublayer $(\mathbf{x}) \in \mathbb{R}^{d}$, 以便残差连接满足 $\mathbf{x}+\operatorname{sublayer}(\mathbf{x}) \in \mathbb{R}^{d}$ 。在残差连接的加法计算之后，紧接着应用层规范化（layer normalization）。因此，输入序列对应的每个位置, transformer编码器都将输出一个 $d$ 维表示向量。

Transformer解码器也是由多个相同的层叠加而成的, 并且层中使用了残差连接和层规范化。除了编码器中描述的两个子 层之外, 解码器还在这两个子层之间揷入了第三个子层, 称为编码器一解码器注意力 (encoder-decoder attention) 层。 在编码器一解码器注意力中, 查询来自前一个解码器层的输出, 而键和值来自整个编码器的输出。在解码器自注意力中, 查询、键和值都来自上一个解码器层的输出。但是，解码器中的每个位置只能考虑该位置之前的所有位置。这种掩蔽 (masked) 注意力保留了自回归（auto-regressive）属性，确保预测仅依赖于已生成的输出词元。




### 基于位置的前馈网络
基于位置的前馈网络对序列中的所有位置的表示进行变换时使用的是同一个多层感知机（MLP），这就是称前馈网络是基于位置的（positionwise）的原因。

它接受一个形状为（batch_size，seq_length, feature_size）的三维张量。Position-wise FFN由两个全连接层组成，他们作用在最后一维上。因为序列的每个位置的状态都会被单独地更新，所以我们称他为position-wise，这等效于一个1x1的卷积。




# word2vec
word2vec将每个词映射到一个固定长度的向量，这些向量能更好地表达不同词之间的相似性和类比关系。word2vec工具包含两个模型，即跳元模型（skip-gram） [Mikolov et al., 2013b]和连续词袋（CBOW） [Mikolov et al., 2013a]。对于在语义上有意义的表示，它们的训练依赖于条件概率，条件概率可以被看作是使用语料库中一些词来预测另一些单词。由于是不带标签的数据，因此跳元模型和连续词袋都是自监督模型。

word2vec在预训练之后，输出可以被认为是一个矩阵，其中每一行都是一个表示预定义词表中词的向量。事实上，这些词嵌入模型都是与上下文无关的，无法区分一词多义，即依据上下文理解词义。

## 跳元模型（Skip-Gram）
跳元模型假设一个词可以用来在文本序列中生成其周围的单词。

在跳元模型中, 每个词都有两个 $d$ 维向量表示, 用于计算条件概率。更具体地说, 对于词典中索引为 $i$ 的任何词, 分别用 $\mathbf{v}_{i} \in \mathbb{R}^{d}$ 和 $\mathbf{u}_{i} \in \mathbb{R}^{d}$ 表示其用作中心词和上下文词时的两个向量。给定中心词 $w_{c}$ （词典中的索引 $c$ ），生成任何上下文词 $w_{o}$ （词典中的索引o）的条件概率可以通过对向量点积的softmax操作来建模:
$$
P\left(w_{o} \mid w_{c}\right)=\frac{\exp \left(\mathbf{u}_{o}^{\top} \mathbf{v}_{c}\right)}{\sum_{i \in \mathcal{V}} \exp \left(\mathbf{u}_{i}^{\top} \mathbf{v}_{c}\right)}
$$
其中词表索引集 $\mathcal{V}=\{0,1, \ldots,|\mathcal{V}|-1\}_{\circ}$ 给定长度为 $T$ 的文本序列，其中时间步 $t$ 处的词表示为 $w^{(t)}$ 。假设上下文词是 在给定任何中心词的情况下独立生成的。对于上下文窗口 $m$ ，跳元模型的似然函数是在给定任何中心词的情况下生成所有 上下文词的概率：
$$
\prod_{t=1}^{T} \prod_{-m \leq j \leq m, j \neq 0} P\left(w^{(t+j)} \mid w^{(t)}\right)
$$
其中可以省略小于1或大于 $T$ 的任何时间步。

## 连续词袋（CBOW）
连续词袋（CBOW）模型类似于跳元模型。与跳元模型的主要区别在于，连续词袋模型假设中心词是基于其在文本序列中的周围上下文词生成的。

由于连续词袋模型中存在多个上下文词, 因此在计算条件概率时对这些上下文词向量进行平均。具体地说, 对于字典中索 引 $i$ 的任意词，分别用 $\mathbf{v}_{i} \in \mathbb{R}^{d}$ 和 $\mathbf{u}_{i} \in \mathbb{R}^{d}$ 表示用作上下文词和中心词的两个向量（符号与跳元模型中相反）。给定上下 文词 $w_{o_{1}}, \ldots, w_{o_{2 m}}$ （在词表中索引是 $o_{1}, \ldots, o_{2 m}$ ）生成任意中心词 $w_{c}$ （在词表中索引是 $c$ ）的条件概率可以由以下公式 建模:
$$
P\left(w_{c} \mid w_{o_{1}}, \ldots, w_{o_{2 m}}\right)=\frac{\exp \left(\frac{1}{2 m} \mathbf{u}_{c}^{\top}\left(\mathbf{v}_{o_{1}}+\ldots,+\mathbf{v}_{o_{2 m}}\right)\right)}{\sum_{i \in \mathcal{V}} \exp \left(\frac{1}{2 m} \mathbf{u}_{i}^{\top}\left(\mathbf{v}_{o_{1}}+\ldots,+\mathbf{v}_{o_{2 m}}\right)\right)}
$$
为了简洁起见，我们设为 $\mathcal{W}_{o}=\left\{w_{o_{1}}, \ldots, w_{o_{2 m}}\right\}$ 和 $\mathbf{v}_{o}=\left(\mathbf{v}_{o_{1}}+\ldots,+\mathbf{v}_{o_{2 m}}\right) /(2 m)$ 。那么可以简化为:
$$
P\left(w_{c} \mid \mathcal{W}_{o}\right)=\frac{\exp \left(\mathbf{u}_{c}^{\top} \overline{\mathbf{v}}_{o}\right)}{\sum_{i \in \mathcal{V}} \exp \left(\mathbf{u}_{i}^{\top} \overline{\mathbf{v}}_{o}\right)} .
$$
给定长度为 $T$ 的文本序列, 其中时间步 $t$ 处的词表示为 $w^{(t)}$ 。对于上下文窗口 $m$, 连续词袋模型的似然函数是在给定其上 下文词的情况下生成所有中心词的概率:
$$
\prod_{t=1}^{T} P\left(w^{(t)} \mid w^{(t-m)}, \ldots, w^{(t-1)}, w^{(t+1)}, \ldots, w^{(t+m)}\right) .
$$

## 下采样
文本数据通常有“the”、“a”和“in”等高频词：它们在非常大的语料库中甚至可能出现数十亿次。然而，这些词经常在上下文窗口中与许多不同的词共同出现，提供的有用信息很少。此外，大量（高频）单词的训练速度很慢。因此，当训练词嵌入模型时，可以对高频单词进行下采样。具体地说，数据集中的每个词将有概率地被丢弃。

## 负采样(Negative Sampling)
只考虑那些正样本的事件。仅当所有词向量都等于无穷大时，联合概率才最大化为1。当然，这样的结果毫无意义。为了使目标函数更有意义，负采样添加从预定义分布中采样的负样本。

以word2vec中的负采样优化策略为例，即把语料中的一个词串的中心词替换为别的词，构造语料集中不存在的词串作为负样本。在这种策略下，优化目标变为了：较大化正样本的概率，同时最小化负样本的概率;

词汇表的大小决定了word2vec进行词向量训练时神经网络将会有一个非常大的权重参数，并且所有的权重参数会随着数十亿训练样本不断调整。negative sampling 每次让一个训练样本仅仅更新一小部分的权重参数，从而降低梯度下降过程中的计算量。

现在每个训练步的梯度计算成本与词表大小无关，而是线性依赖于K噪声词数。当将超参数设置为较小的值时，在负采样的每个训练步处的梯度的计算成本较小。

## 层序Softmax
作为另一种近似训练方法，层序Softmax（hierarchical softmax）使用二叉树，其中树的每个叶节点表示词表中的一个词。

![Alt](https://zh.d2l.ai/_images/hi-softmax.svg)
图14.2.1 用于近似训练的分层softmax，其中树的每个叶节点表示词表中的一个词

用 $L(w)$ 表示二叉树中表示字 $w$ 的从根节点到叶节点的路径上的节点数（包括两端）。设 $n(w, j)$ 为该路径上的 $j^{\text {th }}$ 节点， 其上下文字向量为 $\mathbf{u}_{n(w, j)}$ 。例如, 图14.2.1中的 $L\left(w_{3}\right)=4$ 。分层softmax将 (14.1.4).中的条件概率近似为
$$
P\left(w_{o} \mid w_{c}\right)=\prod_{j=1}^{L\left(w_{o}\right)-1} \sigma\left(\llbracket n\left(w_{o}, j+1\right)=\operatorname{leftChild}\left(n\left(w_{o}, j\right)\right) \rrbracket \cdot \mathbf{u}_{n\left(w_{o}, j\right)}^{\top} \mathbf{v}_{c}\right),
$$
其中函数 $\sigma$ 在 (14.2.2)中定义, leftChild $(n)$ 是节点 $n$ 的左子节点：如果 $x$ 为真, $\llbracket x \rrbracket=1$;否则 $\llbracket x \rrbracket=-1$ 。
为了说明, 让我们计算 图14.2.1中给定词 $w_{c}$ 生成词 $w_{3}$ 的条件概率。这需要 $w_{c}$ 的词向量 $\mathbf{v}_{c}$ 和从根到 $w_{3}$ 的路径（图14.2.1 中加粗的路径）上的非叶节点向量之间的点积, 该路径依次向左、向右和向左遍历：
$$
P\left(w_{3} \mid w_{c}\right)=\sigma\left(\mathbf{u}_{n\left(w_{3}, 1\right)}^{\top} \mathbf{v}_{c}\right) \cdot \sigma\left(-\mathbf{u}_{n\left(w_{3}, 2\right)}^{\top} \mathbf{v}_{c}\right) \cdot \sigma\left(\mathbf{u}_{n\left(w_{3}, 3\right)}^{\top} \mathbf{v}_{c}\right) .
$$
由 $\sigma(x)+\sigma(-x)=1$, 它认为基于任意词 $w_{c}$ 生成词表 $\mathcal{V}$ 中所有词的条件概率总和为 1 :
$$
\sum_{w \in \mathcal{V}} P\left(w \mid w_{c}\right)=1 .
$$
幸运的是, 由于二叉树结构, $L\left(w_{o}\right)-1$ 大约与 $\mathcal{O}\left(\log _{2}|\mathcal{V}|\right)$ 是一个数量级。当词表大小 $\mathcal{V}$ 很大时, 与没有近似训练的相 比, 使用分层softmax的每个训练步的计算代价显著降低。

# ELMo
通过将整个序列作为输入，ELMo是为输入序列中的每个单词分配一个表示的函数。具体来说，ELMo将来自预训练的双向长短期记忆网络的所有中间层表示组合为输出表示。然后，ELMo的表示将作为附加特征添加到下游任务的现有监督模型中，例如通过将ELMo的表示和现有模型中词元的原始表示（例如GloVe）连结起来。一方面，在加入ELMo表示后，冻结了预训练的双向LSTM模型中的所有权重。另一方面，现有的监督模型是专门为给定的任务定制的。利用当时不同任务的不同最佳模型，添加ELMo改进了六种自然语言处理任务的技术水平：情感分析、自然语言推断、语义角色标注、共指消解、命名实体识别和问答。

# GPT
GPT（Generative Pre Training，生成式预训练）模型为上下文的敏感表示设计了通用的任务无关模型。GPT建立在Transformer解码器的基础上，预训练了一个用于表示文本序列的语言模型。当将GPT应用于下游任务时，语言模型的输出将被送到一个附加的线性输出层，以预测任务的标签。与ELMo冻结预训练模型的参数不同，GPT在下游任务的监督学习过程中对预训练Transformer解码器中的所有参数进行微调。由于语言模型的自回归特性，GPT只能向前看（从左到右）。

# BERT
ELMo对上下文进行双向编码，但使用特定于任务的架构；而GPT是任务无关的，但是从左到右编码上下文。BERT（来自Transformers的双向编码器表示）结合了这两个方面的优点。它对上下文进行双向编码，并且对于大多数的自然语言处理任务只需要最少的架构改变。通过使用预训练的Transformer编码器，BERT能够基于其双向上下文表示任何词元。在下游任务的监督学习过程中，BERT在两个方面与GPT相似。首先，BERT表示将被输入到一个添加的输出层中，根据任务的性质对模型架构进行最小的更改，例如预测每个词元与预测整个序列。其次，对预训练Transformer编码器的所有参数进行微调，而额外的输出层将从头开始训练。

BERT进一步改进了11种自然语言处理任务的技术水平，这些任务分为以下几个大类：（1）单一文本分类（如情感分析）、（2）文本对分类（如自然语言推断）、（3）问答、（4）文本标记（如命名实体识别）。

## 输入与输出
在自然语言处理中，有些任务（如情感分析）以单个文本作为输入，而有些任务（如自然语言推断）以一对文本序列作为输入。BERT输入序列明确地表示单个文本和文本对。当输入为单个文本时，BERT输入序列是特殊类别词元$<cls>$、文本序列的标记、以及特殊分隔词元$<sep>$的连结。当输入为文本对时，BERT输入序列是$<cls>$、第一个文本序列的标记、$<sep>$、第二个文本序列标记、以及$<sep>$的连结。我们将始终如一地将术语“BERT输入序列”与其他类型的“序列”区分开来。例如，一个BERT输入序列可以包括一个文本序列或两个文本序列。


# Word2vec,fasttext,ELMo,GPT,BERT的区别
![Alt](https://zh.d2l.ai/_images/elmo-gpt-bert.svg)


# 损失函数总结

## 0-1损失函数(zero-one loss)
$$L(Y, f(X))=\left \{ \begin{array}{l}
1, Y \neq f(X) \\
0, Y=f(X)
\end{array}\right.$$

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


# 激活函数

## sigmoid
sigmoid 函数是一个 logistic 函数，意思就是说：不管输入是什么，得到的输出都在 0 到 1 之间。也就是说，你输入的每个神经元、节点或激活都会被缩放为一个介于 0 到 1 之间的值。

sigmoid 函数输入一个很大的 x 值（正或负）时，我们得到几乎为 0 的 y 值——也就是说，当我们输入 w×a+b 时，我们可能得到一个接近于 0 的值。梯度消失问题使得 sigmoid 函数在神经网络中并不实用。

## ReLU 整流线性单元
$ReLU=\max (0,x)$

死亡 ReLU 问题：如果在计算梯度时有太多值都低于 0 会得到相当多不会更新的权重和偏置，因为其更新的量为 0。

1. 优点
   
   相比于 sigmoid，由于稀疏性，时间和空间复杂度更低；不涉及成本更高的指数运算；
   
   能避免梯度消失问题。

2. 缺点
   
   引入了死亡 ReLU 问题，即网络的大部分分量都永远不会更新。但这有时候也是一个优势；
    
    ReLU 不能避免梯度爆炸问题。

## ELU
指数线性单元激活函数解决了 ReLU 的一些问题，同时也保留了一些好的方面。这种激活函数要选取一个 $\alpha$值；常见的取值是在 0.1 到 0.3 之间。

$$
\begin{aligned}
ELU(x)&=\left\{\begin{array}{l} x \quad\quad\quad\quad\quad &if \quad x>0\\\alpha(e^x-1)\quad ~~ &if\quad x \leq 0 \end{array}\right.
\\
ELU'(x)&=\left\{\begin{array}{l} 1 \quad\quad\quad\quad\quad &if \quad x>0\\ ELU(x)+\alpha ~ &if\quad x \leq 0 \end{array}\right.
\end{aligned}
$$

1. 优点

    能避免死亡 ReLU 问题；

    能得到负值输出，这能帮助网络向正确的方向推动权重和偏置变化；

    在计算梯度时能得到激活，而不是让它们等于 0。
2. 缺点：
    由于包含指数运算，所以计算时间更长；

    无法避免梯度爆炸问题；

    神经网络不学习 α 值。

## Leaky ReLU 渗漏型整流线性单元激活函数

渗漏型整流线性单元激活函数也有一个 α 值，通常取值在 0.1 到 0.3 之间。Leaky ReLU 激活函数很常用，但相比于 ELU 它也有一些缺陷，但也比 ReLU 具有一些优势。Leaky ReLU 的数学形式如下：
$$
LReLU(x)=\left\{\begin{array}{l} x \quad\quad\quad\quad\quad &if \quad x>0\\ \alpha x ~ &if\quad x \leq 0 \end{array}\right.
$$

能够解决死亡 ReLU 问题，因为梯度的值不再被限定为 0，另外，这个函数也能避免梯度消失问题。
1. 优点：

    类似 ELU，Leaky ReLU 也能避免死亡 ReLU 问题，因为其在计算导数时允许较小的梯度；

    由于不包含指数运算，所以计算速度比 ELU 快。
2. 缺点：

    无法避免梯度爆炸问题；

    神经网络不学习 α 值；

    在微分时，两部分都是线性的；而 ELU 的一部分是线性的，一部分是非线性的。

## SELU 扩展型指数线性单元激活函数

当实际应用这个激活函数时，必须使用 lecun_normal 进行权重初始化。如果希望应用 dropout，则应当使用 AlphaDropout。论文作者已经计算出了公式的两个值：α 和 λ。
$$
SELU(x)=\lambda \left \{\begin{array}{l} x \quad\quad\quad\quad\quad &if \quad x>0\\\alpha(e^x-1)\quad ~~ &if\quad x \leq 0 \end{array}\right.
$$
SELU 激活能够对神经网络进行自归一化（self-normalizing）。

1. 优点：

    内部归一化的速度比外部归一化快，这意味着网络能更快收敛；
    不可能出现梯度消失或爆炸问题。
2. 缺点：

    这个激活函数相对较新——需要更多论文比较性地探索其在 CNN 和 RNN 等架构中应用

## GELU
为了避免深度神经网络只作为一种深度线性分类器，必须要加入激活函数以希望其拥有非线性拟合的能力，其中，ReLU就是一个较好的激活函数。而同时为了避免其过拟合，又需要通过加入正则化来提高其泛化能力，其中，Dropout就是一种主流的正则化方式，而zoneout是Dropout的一个变种。

我们将神经元的输入 $x$ 乘上一个服从伯努利分布的 $m$ 。而该伯努利分布又是依赖于 $x$ 的:
$m \sim \operatorname{Bernoulli}(\Phi(x))$, where $\Phi(x)=P(X<=x)$
其中, $X$ 服从标准正态分布, 即 $X \sim N(0,1)$, 那么 $\Phi(x)$ 就是标准正态分布的累积分布函 数, 如下图红线所示。这么做的原因是因为神经元的输入 $x$ 往往遵循正态分布, 尤其是深度网络 中普遍存在Batch Normalization的情况下。

![Alt](https://pic4.zhimg.com/80/v2-53dabc61fef4c916739a83893837af37_1440w.jpg)
当$x$减小时， $\Phi(x)$ 的值也会减小，此时$x$被“丢弃”的可能性更高。所以说这是随机依赖于输入的方式。

现在, 给出GELU函数的形式:
$$
G E L U(x)=\Phi(x) * I(x)+(1-\Phi(x)) * 0 x=x \Phi(x)
$$
其中 $\Phi(x)$ 是上文提到的标准正态分布的累积分布函数, 即:
$$
\Phi(x)=\int_{-\infty}^{x} \frac{e^{-t^{2} / 2}}{\sqrt{2 \pi}} d t=\frac{1}{2}\left[1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$
其中, 中间的 $\int_{-\infty}^{x} \frac{e^{-t^{2} / 2}}{\sqrt{2 \pi}} d t$ 是原本标准正态分布的累积分布函数, 而同时, 标准正太分布的累 积分布函数又可以用误差函数 (erf) 来表示: $\quad \operatorname{erf}(x)=\frac{2}{\sqrt{\pi}} \int_{0}^{x} e^{-t^{2}} d t$ 。最终得到了等式最右 边的式子。
最终GELU的定义如下：
$$
G E L U(x)=x P(X \leq x)=x \Phi(x)=x \cdot \frac{1}{2}[1+\operatorname{erf}(x / \sqrt{2})]
$$
因为erf无解析表达式（实际上已经有精确的计算函数torch.erf），因此原论文给出了两种近似表达。
Sigmoid近似：

$$x\Phi(x)\approx x \sigma(1.702x)$$

tanh逼近方式：
$$
GELU(x)=0.5x*(1+tanh(\sqrt{2/\pi}(x+0.044715x^3)))
$$

1. 优点：

    似乎是 NLP 领域的当前最佳；尤其在 Transformer 模型中表现最好；
    能避免梯度消失问题。