
# pytorch
## ModuleList/Sequential

pytorch可以自动识别nn.ModuleList中的参数而普通的list则不可以。

Sequential定义的网络中各层会按照定义的顺序进行级联，因此需要保证各层的输入和输出之间要衔接。并且nn.Sequential实现了farward()方法，因此可以直接通过类似于x=self.combine(x)的方式实现forward。

而nn.ModuleList则没有顺序性要求，并且也没有实现forward()方法，需要重新自己定义。

```python
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


## torch.rand()、torch.randn()、torch.randint()、torch.randperm()
torch.rand(*size, out=None, dtype=None) size为一个元组，均匀分布。

torch.randn(*size, out=None, dtype=None) size为一个元组，标准正态分布，均值0方差1。

torch.randint(low=0, high, size, out=None, dtype=None) 整数范围[low, high)，size为一个元组。

y = torch.randperm(n) y是把1到n这些数随机打乱得到的一个数字序列

## repeat

repeat_interleave是一个元素一个元素地重复，而repeat是一组元素一组元素地重复。
## torch.matmul
矩阵乘法

## 转置函数
reshape 与 view 可以重新设置维度；permute 和 transpose 只能 在已有的维度之间转换，并且包含转置的概念

### reshape/view
reshape会创建一个新张量，并且把原张量的数据复制到新张量中，而view不会复制。

### transpose() 

input (Tensor) – 输入张量，必填

dim0 (int) – 转置的第一维，默认0，可选

dim1 (int) – 转置的第二维，默认1，可选

两个维度交换

### permute()
dims (int…*)-换位顺序，必填

0，1，2.....表示真实的维度位置。

torch.transpose(x)合法， x.transpose()合法。

tensor.permute(x)不合法，x.permute()合法。

### contiguous()
经常有人用view()函数改变通过转置后的数据结构，导致报错
RuntimeError: invalid argument 2: view size is not compatible with input tensor's....

这是因为tensor经过转置后数据的内存地址不连续导致的,也就是tensor .is_contiguous()==False
这时候reshape()可以改变该tensor结构，但是view()不可以

调用contiguous()时，会强制拷贝一份tensor，让它的布局和从头创建的一毛一样，x.reshape(3,4) 这个操作等于x = x.contiguous().view()。

### 计算两个矩阵间向量的距离
```py
import torch
H_1 = torch.ones(6,100)
H_2 = torch.ones(10,100)
H_dot = torch.matmul(H_1,H_2.transpose(-2,-1))
H_1_square = torch.sum(H_1 * H_1, dim=-1).unsqueeze(1).repeat(1,10)
H_2_square = torch.sum(H_2 * H_2, dim=-1).unsqueeze(1).repeat(1,6).transpose(-2,-1)
H = H_1_square + H_2_square - 2 * H_dot
```

## range/arange

range(begin,end,step)

[begin,end)，step为正数时，从begin到end-1，step为负数时，从end到begin-1。当设置起点、终点或步长为小数时，将会报错。

arange(begin,end,step)step为正数时，从begin到end-1，step为负数时，从end到begin-1。三个参数支持小数，终点为小数，默认起点为0.0，步长为1.0。


## nn.apply
apply(fn)的官网介绍，该方法会将fn递归的应用于模块的每一个子模块（.children()的结果）及其自身。典型的用法是，对一个model的参数进行初始化。

# 自回归
自回归（auto-regressive）属性，预测仅依赖于已发生的事情或已生成的信息。

GPT，Transformer，lstm都用到了自回归。
# LayerNorm/BatchNorm 稳定梯度，加速收敛
LayerNorm (num_features=data.shape(axis=-1), epsilon= 1e-05, center-True, scale= True, beta _initializer=zeros, gamma_initializer='ones, in_channels=O, prefix=None, params=None)

BatchNorm (num_features=data.shape(axis=1)=channel, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, in_channels=O, prefix=None, params=None)

LayerNorm是一个层规范化，而BatchNorm是一个批量规范化。为了保证数据特征分布的稳定性，会加入Normalization。可以稳定梯度，从而可以使用更大的学习率，从而加速模型的收敛速度。同时，Normalization也有一定的抗过拟合作用，使训练过程更加平稳。具体地，Normalization的主要作用就是把每层特征输入到激活函数之前，对它们进行normalization，使其转换为均值为0，方差为1的数据，从而可以避免数据落在激活函数的饱和区，以减少梯度消失的问题。

参数为num_features，即通道数/（句子*）向量维度。

LayerNorm考虑低维（句子*）向量的归一化，而BatchNorm考虑高维每个batch内同一通道的的归一化。

layerNorm更适合序列问题，序列内关系紧密，batch间关系稀疏。BatchNorm更适合图像问题，不同通道内关系更加紧密。

$$
out =\frac{x-\operatorname{mean}[\text { data, features }]}{\sqrt{\text { Var }[\text { data, features }]}+\epsilon} * gamma + beta
$$

除以标准差这一项，更像是一个自适应的学习率校正项，它一定程度上消除了不同层级的输入对参数优化的差异性，使得整个网络的优化更为“同步”，或者说使得神经网络的每一层更为“平权”，从而更充分地利用好了整个神经网络，减少了在某一层过拟合的可能性。当然，如果输入的量级过大时，除以标准差这一项也有助于降低梯度的L常数。

gamma为再缩放参数，beta是再偏移参数，epsilon是防止除零的情况。为了保证模型的表达能力不因为规范化而下降。我们将规范化后的数据进行再平移和再缩放，使得每个神经元对应的输入范围是针对该神经元量身定制的一个确定范围。新参数很容易通过梯度下降来学习，简化了神经网络的训练。

**去掉center这一步后性能还略有提升**,称之为RMS Norm。论文总的结果显示：RMS Norm比Layer Normalization更快，效果也基本一致。center操作，类似于全连接层的bias项，储存到的是关于预训练任务的一种先验分布信息，而把这种先验分布信息直接储存在模型中，反而可能会导致模型的迁移能力下降。所以T5不仅去掉了Layer Normalization的center操作，它把每一层的bias项也都去掉了。

## 白化 whitening
独立同分布的数据可以简化模型的训练以及提升模型的预测能力——这是通过训练数据获得的模型能够在测试集获得好的效果的一个基本保障。也就是说我们在使用机器学习和深度学习的时候，会把数据尽可能的做一个独立同分布的处理，用来加快模型的训练速度和提升模型的性能。

1. 去除数据之间的关联性，使之满足独立这个条件；
2. 使得特征具有相同的均值和方差，就是同分布。

也是normalization的原因之一。

## Pre Norm/Post Norm
post-norm和pre-norm其实各有优势，post-norm在残差之后做归一化，对参数正则化的效果更强，进而模型的鲁棒性也会更好；pre-norm相对于post-norm，因为有一部分参数直接加在了后面，不需要对这部分参数进行正则化，正好可以防止模型的梯度爆炸或者梯度消失。

同一设置之下，Pre Norm结构往往更容易训练，因为它的恒等路径更突出，但最终效果通常不如Post Norm。

这个意思是说，当t比较大时，xt,xt+1相差较小，所以Ft+1(Norm(xt+1))与Ft+1(Norm(xt))很接近，因此原本一个t层的模型与t+1层和，近似等效于一个更宽的t层模型，所以在Pre Norm中多层叠加的结果更多是增加宽度而不是深度，层数越多，这个层就越“虚”。

Pre Norm结构无形地增加了模型的宽度而降低了模型的深度，而我们知道深度通常比宽度更重要，所以是无形之中的降低深度导致最终效果变差了。同时这样最后的xl方差将会很大，所以在接预测层之前xl也还要加个Normalization。

而Post Norm刚刚相反，它每Norm一次就削弱一次恒等分支的权重，所以Post Norm反而是更突出残差分支。这种做法虽然稳定了前向传播的方差，但事实上已经严重削弱了残差本身，所以反而失去了残差“易于训练”的优点，通常要warmup并设置足够小的学习率才能使它收敛。

# TextCNN/CharCNN
文本分类模型

Text-CNN 和传统的 CNN 结构类似，具有词嵌入层、卷积层、池化层和全
连接层的四层结构，Text-CNN 的卷积层是主要部分，卷积核的宽度等于词向量的维度，经卷积 后可以提取文本的特征向量。与在图像领域应用类似，Text-CNN 可以设置多个 卷积核以提取文本的多层特征，长度为 N 的卷积核可以提取文本中的 N-gram 特征。

# 自回归语言模型（Autoregressive LM）/ 自编码语言模型（Autoencoder LM）

在ELMO／BERT出来之前，大家通常讲的语言模型其实是根据上文内容预测下一个可能跟随的单词，就是常说的自左向右的语言模型任务，或者反过来也行，就是根据下文预测前面的单词，这种类型的LM被称为自回归语言模型。GPT 就是典型的自回归语言模型。ELMO尽管看上去利用了上文，也利用了下文，但是本质上仍然是自回归LM，这个跟模型具体怎么实现有关系。ELMO是做了两个方向（从左到右以及从右到左两个方向的语言模型），但是是分别有两个方向的自回归LM，然后把LSTM的两个方向的隐节点状态拼接到一起，来体现双向语言模型这个事情的。所以其实是两个自回归语言模型的拼接，本质上仍然是自回归语言模型。

自回归语言模型有优点有缺点，缺点是只能利用上文或者下文的信息，不能同时利用上文和下文的信息，当然，貌似ELMO这种双向都做，然后拼接看上去能够解决这个问题，因为融合模式过于简单，所以效果其实并不是太好。它的优点，其实跟下游NLP任务有关，比如生成类NLP任务，比如文本摘要，机器翻译等，在实际生成内容的时候，就是从左向右的，自回归语言模型天然匹配这个过程。而Bert这种DAE模式，在生成类NLP任务中，就面临训练过程和应用过程不一致的问题，导致生成类的NLP任务到目前为止都做不太好。

自回归语言模型只能根据上文预测下一个单词，或者反过来，只能根据下文预测前面一个单词。相比而言，Bert通过在输入X中随机Mask掉一部分单词，然后预训练过程的主要任务之一是根据上下文单词来预测这些被Mask掉的单词，如果你对Denoising Autoencoder(去噪自编码器)比较熟悉的话，会看出，这确实是典型的DAE的思路。那些被Mask掉的单词就是在输入侧加入的所谓噪音。类似Bert这种预训练模式，被称为DAE LM。

这种DAE LM的优缺点正好和自回归LM反过来，它能比较自然地融入双向语言模型，同时看到被预测单词的上文和下文，这是好处。缺点是啥呢？主要在输入侧引入[Mask]标记，导致预训练阶段和Fine-tuning阶段不一致的问题，因为Fine-tuning阶段是看不到[Mask]标记的。DAE吗，就要引入噪音，[Mask] 标记就是引入噪音的手段，这个正常。

XLNet的出发点就是：能否融合自回归LM和DAE LM两者的优点。就是说如果站在自回归LM的角度，如何引入和双向语言模型等价的效果；如果站在DAE LM的角度看，它本身是融入双向语言模型的，如何抛掉表面的那个[Mask]标记，让预训练和Fine-tuning保持一致。

# 其他预训练模型

GPT 等自回归普通语言模型的训练目标是 Causal Language Model，BERT 的训练目标是 MLM或DAE LM，XLNet的目标是 PLM

## RoBERTa
### 1.0
1. Dynamic Masking

   原本的BERT采用的是static mask的方式，现在改为每一次将训练example喂给模型的时候，才进行随机mask。
2. 使用BPE（Byte-Pair Encoding）
3. 更大的参数量
4. 取消了NSP
### 2.0
1. 更换tokenizer，将词典扩大了。从1.0版的50k扩成了128k。这个扩大无疑大大增加了模型的capacity。

2. 在第一个transformer block后加入卷积。这个技巧在token classification、span prediction任务里经常用到。

3. 共享位置和内容的变换矩阵

4. 把相对位置编码换成了log bucket，各个尺寸模型的bucket数都是256

### 3.0
将预训练任务由掩码语言模型（MLM）换成了ELECTRA一样类似GAN的Replaced token detect任务。

## XLNet
XLNet引入了自回归语言模型以及自编码语言模型的提法

既想要自编码语言模型的双向性，又想要自回归语言模型的预测性。

XLNet在预训练阶段引入Permutation Language Model的训练目标。仍然是个自回归的从左到右的语言模型，但是其实通过对句子中单词排列组合，把一部分Ti下文的单词排到Ti的上文位置中，于是，就看到了上文和下文，但是形式上看上去仍然是从左到右在预测后一个单词。

一是来源于 Transformer-XL 对更长的上文信息的提取，二是来源于 PLM 训练目标与下游任务的一致性（没有 train-test skewness）。
## ALBERT
当我们让一个模型的参数变多的时候，一开始模型效果是提高的趋势，但一旦复杂到了一定的程度，接着再去增加参数反而会让效果降低，这个现象叫作“model degratation"。

- Factorized Embedding Parameterization. 

    他们做的第一个改进是针对于Vocabulary Embedding。在BERT、XLNet中，词表的embedding size(E)和transformer层的hidden size(H)是等同的，所以E=H。但实际上词库的大小一般都很大，这就导致模型参数个数就会变得很大。为了解决这些问题他们提出了一个基于factorization的方法。

    他们没有直接把one-hot映射到hidden layer, 而是**先把one-hot映射到低维空间之后，再映射到hidden layer**。这其实类似于做了矩阵的分解。

- Cross-layer parameter sharing. 
  
  Zhenzhong博士提出**每一层的layer可以共享参数**，这样一来参数的个数不会以层数的增加而增加。所以最后得出来的模型相比BERT-large小18倍以上。

- Inter-sentence coherence loss. 
  
  在BERT的训练中提出了next sentence prediction loss, 也就是给定两个sentence segments, 然后让BERT去预测它俩之间的先后顺序，但在ALBERT文章里提出这种是有问题的，其实也说明这种训练方式用处不是很大。 所以他们做出了改进，他们使用的是setence-order prediction loss (SOP)，**其实是基于主题的关联去预测是否两个句子调换了顺序**。

## DeBERTa
### 改进
1. 更加解耦的self attention

    解耦是将位置信息和内容信息分别/交叉做attention。DeBERTa是相对位置编码，不同于BERT的绝对位置编码。

2. 考虑绝对位置的MLM任务，Enhanced Mask Decoder；DeBERTa比较有意思的地方，是提供了使用相对位置和绝对位置编码的一个新视角，它指出NLP的大多数任务可能都只需要相对位置信息，但确实有些场景下绝对位置信息更有帮助，于是它将整个模型分为两部分来理解。以Base版的MLM预训练模型为例，它一共有13层，前11层只是用相对位置编码，这部分称为Encoder，后面2层加入绝对位置信息，这部分它称之为Decoder，还弄了个简称EMD（Enhanced Mask Decoder）；至于下游任务的微调截断，则是使用前11层的Encoder加上1层的Decoder来进行。

3. 预训练时引入对抗训练
    DeBERTa预训练里面引入的对抗训练叫SiFT，他攻击的对象不是word embedding，而是embedding之后的layer norm。

## ERNIE
由三种level的mask组成，分别是basic-level masking（word piece）+ phrase level masking（WWM style） + entity level masking。

模型在预测未知词的时候，没有考虑到外部知识。但是如果我们在mask的时候，加入了外部的知识，模型可以获得更可靠的语言表示。

1. Basic level masking 在预训练中，第一阶段是先采用基本层级的masking，即随机mask掉中文中的一个字。
2. Phrase level masking第二阶段是采用词组级别的masking。我们mask掉句子中一部分词组，然后让模型预测这些词组，在这个阶段，词组的信息就被encoding到word embedding中了。
3. Entity level masking在第三阶段命名实体，如：人名，机构名，商品名等，在这个阶段被mask掉，模型在训练完成后，也就学习到了这些实体的信息。

## T5
T5不仅去掉了Layer Normalization的center操作，它把每一层的bias项也都去掉了。

初始化q,k的全连接层的时候，其初始化方差要多除以一个d，这同样能使得使q⋅k的初始方差变为1，T5采用了这样的做法。

T5模型出自文章《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》，里边用到了一种更简单的相对位置编码。思路依然源自展开式(7)，如果非要分析每一项的含义，那么可以分别理解为“输入-输入”、“输入-位置”、“位置-输入”、“位置-位置”四项注意力的组合。如果我们认为输入信息与位置信息应该是独立（解耦）的，那么它们就不应该有过多的交互，所以“输入-位置”、“位置-输入”两项Attention可以删掉，剩下的位置编码换成相对位置编码我们可以直接将它作为参数训练出来。它仅仅是在Attention矩阵的基础上加一个可训练的偏置项而已。


## Bert改进
随机mask，增大参数量，取消NSP，增大Batch_size （RoBERTa）

修改为PLM，预训练任务改为PLM位置相关（解决未来文本的信息泄漏问题，没有 train-test skewness，拥有文本生成的能力）（XLNet）

降低隐藏层参数量，共享隐藏层参数，修改NSP为预测句子顺序是否调换（ALBERT）

位置编码改成相对位置，并且与输入分离，放在多头attention处（Transformer-XL，XlNet，DeBERTa，T5）

增加三种level的mask，分别是basic-level masking（word piece）+ phrase level masking（WWM style） + entity level masking。（ERNIE）

分段后的数据做段间递归，增大长距离文本的依赖（Transformer-XL） 

改进Layer Norm，取消center和bias（T5）
# Bert问题

## 为什么除以更号dk
对于Mul attention 来说，如果分布都分布在0-1，在相乘时引入一次对所有位置的求和，整体的分布就会扩大到 [0，dk]。输入输出方差一致(公式推导)/防止梯度消失。

$D(\sum XY)=\sum D(XY)=d_k$

反过来看 Add attention，右侧是被 tanh() 钳位后的值，分布在[-1，1]。整体分布和 dk没有关系。

相应地，解决方法就有两个:

1. 像NTK参数化那样，在内积之后除以$\sqrt{d}$，使q⋅k的方差变为1
2. 初始化q,k的全连接层的时候，其初始化方差要多除以一个d，这同样能使得使q⋅k的初始方差变为1，T5采用了这样的做法。
3. 用余弦相似度代替点积

## 怎么处理padding
[PAD]权重置为负无穷，softmax置为0

## 多头作用
我们可以用独立学习得到的组不同的 线性投影（linear projections）来变换查询、键和值。

实际上为了并行操作并没有多个attention操作，而是将hidden层的数据分层后交由一个大的参数矩阵运算，每一个头并行的关注到不同的输入。

## 为什么选择使用[cls]的输出代表整句话的语义表示？
或者说为什么不选择token1的输出

BERT在第一句前会加一个[CLS]标志，最后一层该位对应向量可以作为整句话的语义表示，从而用于下游的分类任务等。

为什么选它呢，

因为与文本中已有的其它词相比，这个无明显语义信息的符号会更“公平”地融合文本中各个词的语义信息，从而更好的表示整句话的语义。

从self-attention结构中可以看出把cls位置当做q，类似于站在全局的角度去观察整个句子文本。

## 为什么是双线性点积模型（经过线性变换Q != K）？

双线性点积模型，引入非对称性，更具健壮性（Attention mask对角元素值不一定是最大的，也就是说当前位置对自身的注意力得分不一定最高）
展开解释：

self-attention中，sequence中的每个词都会和sequence中的每个词做点积去计算相似度，也包括这个词本身。

对于 self-attention，一般会说它的 q=k=v，这里的相等实际上是指它们来自同一个基础向量，而在实际计算时，它们是不一样的，因为这三者都是乘了QKV参数矩阵的。那如果不乘，每个词对应的q,k,v就是完全一样的。

在相同量级的情况下，𝑞𝑖 与 𝑘𝑖 点积的值会是最大的（可以从“两数和相同的情况下，两数相等对应的积最大”类比过来）。

那在softmax后的加权平均中，该词本身所占的比重将会是最大的，使得其他词的比重很少，无法有效利用上下文信息来增强当前词的语义表示。

而乘以QKV参数矩阵，会使得每个词的q,k,v都不一样，能很大程度上减轻上述的影响。

## BERT 怎么处理长文本?
长于512的会截断，当作两个文本的输入

transformer-XL将前一个文本的输出和后一个的输入拼接，这也是XLNet的由来。

## transformer中multi-head attention中每个head为什么要进行降维？
简洁的说：在不增加时间复杂度的情况下，同时，借鉴CNN多核的思想，在更低的维度，在多个独立的特征空间，更容易学习到更丰富的特征信息。
## self-attention的时间复杂度

时间复杂复杂度为：𝑂(𝑛^2 𝑑)

相似度计算可以看成矩阵(n×d)×(d×n)，这一步的时间复杂度为O(n×d×n)

softmax的时间复杂度O(n^2)

加权平均可以看成矩阵(n×n)×(n×d), 时间复杂度为O(n×n×d)
## feed forword作用
先将数据映射到高维空间再映射到低维空间的过程，可以学习到更加抽象的特征，即该Feed Forward层使得单词的representation的表达能力更强，更加能够表示单词与contex中其他单词之间的作用关系。

## MLM的两个问题：

1. 首先，预训练和finetuning之间不匹配，因为在finetuning期间从未看到[MASK]token。

2. 每个batch只预测了15％的token，这表明模型可能需要更多的预训练步骤才能收敛。团队证明MLM的收敛速度略慢于 left-to-right的模型（预测每个token），但MLM模型在实验上获得的提升远远超过增加的训练成本。

## 自学习位置编码和固定位置编码的区别
Transformer的位置编码是一个固定值，因此只能标记位置，但是不能标记这个位置有什么用。

BERT的位置编码是可学习的Embedding，因此不仅可以标记位置，还可以学习到这个位置有什么用。

BERT选择这么做的原因可能是，相比于Transformer，BERT训练所用的数据量充足，完全可以让模型自己学习。

## BERT“向量坍缩”
由于BERT对所有的句子都倾向于编码到一个较小的空间区域内，这使得大多数的句子对都具有较高的相似度分数，这种现象是由于句子中的高频词的影响，即当通过平均词向量的方式计算句向量时，那些高频词的词向量将会主导句向量，使之难以体现其原本的语义。

Observation 1：BERT句向量的空间分布是不均匀的，受到词频的影响。因为词向量在训练的过程中起到连接上下文的作用，词向量分布受到了词频的影响，导致了上下文句向量中包含的语义信息也受到了破坏。

Observation 2：BERT句向量空间是各向异性的，高频词分布较密集且整体上更靠近原点，低频词分布较稀疏且整体分布离原点相对较远，如上图A所示。因为低频词的词向量分布较稀疏，因此它们周围存在较多的“空洞”，所谓的“空洞”即几乎不能表征语义或者说在进行语义表征时，该部分空间没有被使用到是语义不明确的（poorly definded）。
### BERT-flow
为了解决BERT句向量分布不平滑问题，可以利用标准化流(Normalizing Flows)将BERT句向量分布变换为一个光滑的、各向同性的标准高斯分布。
### BERT-whitening
对输出矩阵做SVD奇异值分解，降维。

通过引入两个超参数的方式来赋予BERT-whitening一定的调参空间，使其具备“不逊色于变换前的效果”的可能性，并且保留了降维的能力。换言之，即便是之前已经训练好的句向量模型，我们也可以用新的BERT-whitening将它降维，并且保持效果基本不变，有时候甚至还更优。
### 对比学习

SimCSE: Simple Contrastive Learning of Sentence Embeddings

SIMCSE采用的方法很简单，尤其是无监督，将一个batch的数据两次经过BERT（实际上是复制batch的数据再经过BERT，可以加快效率），得到不同的两个输出，比如输入的样本x再两次经过BERT后，由于DropMask的不同，得到的输出就是不一样的，将正例和其他的负例对比损失计算loss。

Dropout是在神经网络中随机关闭掉一些神经元的连接，那么关闭的神经元不一样，模型最终的输出也就不一样。因此，作者在这里通过将一句话分两次过同一个模型，但使用两种不同的dropout，这样得到的两个sentence embedding就作为模型的正例，而同一个batch中的其他embedding就变为了负例。
## Bert参数量
### Bert base
BertModel(vocab_size=30522，hidden_size=768，max_position_embeddings=512，token_type_embeddings=2)

bert的参数主要可以分为四部分：embedding层的权重矩阵、multi-head attention、layer normalization、feed forward。接下来我们就分别来计算。

我们可以看到embedding层有三部分组成：token embedding、segment embedding和position embedding。

token embedding：词表大小词向量维度就是对应的参数了，也就是30522*768

segment embedding：主要用01来区分上下句子，那么参数就是2*768

position embedding：文本输入最长为512，那么参数为512*768

在embedding层最后有Layer Norm 层，参数量为768+768

![](https://zh.d2l.ai/_images/bert-input.svg)

每个head的参数为768* 768/12，对应到QKV三个权重矩阵自然是768* 768/12* 3，12个head的参数就是768* 768/12* 3* 12，我们可以从下列公式看到拼接后经过一个线性变换，这个线性变换对应的权重为768* 768+768。

因此1层multi-head attention部分的参数为768* 768/12* 3* 12+768* 768+768，12层自然是 * 12

layer normalization有两个参数，分别是gamma和beta。有三个地方用到了layer normalization，分别是embedding层后、multi-head attention后、feed forward后，2*768

feed forward的参数主要由两个全连接层组成，intermediate_size为3072，那么参数为12*(768* 3072+3072+3072* 768+3072)

在最后一层feed forward层后还有一个pooler层，维度为768* 768，参数量为(768*768+768)，为获取训练数据中第一个特殊字符[CLS]的词向量，进一步计算bert中的NSP任务中的loss

总的参数=embedding+multi-head attention+layer normalization+feed forward

### Bert large
模型参数计算：
BertEmbedding:
包含三个表和一个LayerNorm，分别是wordembedding,tokentype_embedding, position_embedding. 其中:

word_embedding为Embedding(30522, 1024),
segment_embedding为Embedding(2, 1024),
position_embedding为Embedding(512, 1024) 。
则参数量为(30522+2+512)* 1024 + 1024*2 = 31782912
BertEncoder:包含了24个BertLayer
BertLayer：由Q，K，V，拼接Linear，2层的FFN和两个LayerNorm组成。

Q，K，V和拼接Linear都是Linear(1024,1024)，

FFN为Linear(1024,4096)以及Linear(4096,1024)。

则参数量为(1024*(1024+1))* 4 + (1024+1)* 4096 + 1024*(4096+1) + 1024*4=12596224

BertPooler：一个Linear(1024, 1024)则参数量为1024*(1024+1)=1049600

总参数量： 31782912+24*12596224+1049600 = 335141888

# 为什么门控网络有两种激活函数
影响后续数据的用sigmod，保证数据在0-1内，符合门控的物理含义。

靠近0代表消除之前数据的影响，靠近1代表更加影响后续数据。

候选记忆选择tanh，保证数据在-1-1内，由于原本数据特征分布均值为0，所以记忆要满足数据在-1-1内，同时保证有更大的梯度。


# dropout和bagging
bagging是一种集成方法（ensemble methods）,可以通过集成来减小泛化误差（generalization error）。

bagging的最基本的思想是通过分别训练几个不同分类器，最后对测试的样本，每个分类器对其进行投票。在机器学习上这种策略叫model averaging。

model averaging 之所以有效，是因为并非所有的分类器都会产生相同的误差，只要有不同的分类器产生的误差不同就会对减小泛化误差非常有效，同时可以减少过拟合。

对于bagging方法，允许采用相同的分类器，相同的训练算法，相同的目标函数。但是在数据集方面，新数据集与原始数据集的大小是相等的。每个数据集都是通过在原始数据集中随机选择一个样本进行替换而得到的。意味着，每个新数据集中会存在重复的样本。

我们可以把dropout类比成将许多大的神经网络进行集成的一种bagging方法。
但是每一个神经网络的训练是非常耗时和占用很多内存的，训练很多的神经网络进行集合分类就显得太不实际了。

dropout可以训练所有子网络的集合，这些子网络通过去除整个网络中的一些神经元来获得。我们通过仿射和非线性变换，使神经元的输出乘以0。

## 对比
在bagging中，所有的分类器都是独立的，而在dropout中，所有的模型都是共享参数的。

在bagging中，所有的分类器都是在特定的数据集下训练至收敛，而在dropout中没有明确的模型训练过程。网络都是在一步中训练一次（输入一个样本，随机训练一个子网络）

（相同点）对于训练集来说，每一个子网络的训练数据是通过原始数据的替代采样得到的子集。

dropout没有很显著的限制模型的大小和训练的过程。

## 预测时不加dropout
深度学习模型训练时候使用dropout实际上只是让部分神经元在当前训练批次以一定的概率不参与更新，这样使得每一轮迭代获得的模型都是不一样的。这个过程一定程度上保持了不同模型之间最优参数设置，使得训练出的每一个模型不至于太差。在预测时候，不使用dropout，但会在权重上都乘上保留概率。最终的输出可以被认为是Bagging的一种近似。


# 剪枝
机器学习的决策树算法，为防止过拟合现象和过度开销，而采用剪枝的方法，主要有预剪枝和后剪枝两种常见方法。

## 预剪枝
自顶至下，从根节点开始，如果当前节点展开后的预测效果大于未展开的预测效果，则会展开，否则不展开。

优点：算法简单，有效避免过拟合现象。

缺点：欠拟合现象严重，比如原本60%的正确率，展开后变成55%，而禁止节点划分，显然不合理。

## 后剪枝
自下至上，从叶子节点开始，如果某个节点剪枝后的正确率更高，则进行剪枝，否则不剪枝。

优点：有效避免欠拟合现象，正确率较高。

缺点：需要先生成完整的决策树，开销大。

# 正则化
能够提高模型在test上的准确率，能够提高模型的泛化能力所做的任何改动，我们都可以称之为正则化。

正则化是结构风险最小化策略的实现，是在经验风险上加一个正则化项或惩罚项。正则化项一般是模型复杂度的单调递增函数，模型越复杂，正则化项就越大。
## L0
向量中非零元素的个数。
## L1-norm Lasso
$$
L=loss(x,w)+\frac{\lambda}{d}||w||_1
$$

优化的目标是使得模型的预测的结果尽可能准的条件下，希望模型的参数权重尽可能为稀疏。

L1假设模型参数服从拉普拉斯分布。在损失函数中加入这样一个惩罚项，让模型的权重参数服从拉普拉斯分布，若权重参数离分布的中心值较远，则对应的概率越小，产生的损失loss就会大，起到惩罚的作用。L1正则化可以让模型更加稀疏，简化模型的效果，进而防止模型过拟合。

优点：

L1正则可以使得某些特征的系数为0，具有特征选择的能力，这便称为稀疏性(Sparsity)。通俗一点的理解就是，L1正则会使损失函数在0附近反复经过，当其大于0时，loss值会减小；当其小于0时，loss值又会增大。

我觉得还是图像生动一点，极小值产生时，也就是说损失函数时是一个凹图

，即此处的损失函数的导数是为0的，不就是梯度为0吗？梯度不就是w嘛。。。所以会产生稀疏解（就是导致许多参数的解变为0，这样模型就变得稀疏了）

无论对于什么样的输入值，都有着稳定的梯度，不会导致梯度爆炸问题，具有较为稳健性的解

缺点：

L1 范数没有解析解，因此计算效率上比 L2 差。

在中心点是折点，不可导

1. 回归任务
2. 简单模型
3. 神经网络通常比较复杂，直接使用L1 loss作为损失函数的非常少
## L2-norm Ridge：
$$
L=loss(x,w)+\frac{\lambda}{d}||w||_2^2
$$

$||w||$表示模型的权重向量的L2范数，$\lambda$是正则化参数，$d$是参数维度，一般要除。

第一：整个loss不仅仅看模型预测结果与label的度量结果产生的loss，还与本身权重参数w的有关系，也就是说优化的最终目标是使得模型的预测的结果尽可能准的条件下，希望模型的参数权重也尽可能小，防止模型结构复杂，减少模型过拟合。

第二：从贝叶斯角度理解，L2正则化其实是加入了数据的先验知识，认为权重参数服从高斯先验分布(均值为0，标准差近似为$\lambda^{-1}$)，所以在损失函数中加入这样的先验知识，让模型表现更好。

优点：

性能比 LASSO 回归稍好，各点都连续光滑，也方便求导计算得到解析解

岭回归的平方偏差因子向模型中引入了少量偏差，但大大减少了方差，使得模型更稳定

岭回归的假设和最小平方回归相同，但是在最小平方回归的时候我们假设数据服从高斯分布使用的是极大似然估计 (MLE)，在岭回归的时候由于添加了偏差因子，即θ的先验信息，使用的是极大后验估计 (MAP) 来得到最终的参数 

缺点：

岭回归没有特征选择功能

不是特别的稳健，因为当函数的输入值距离真实值较远的时候，对应loss值很大在两侧，则使用梯度下降法求解的时候梯度很大，可能导致梯度爆炸

L2正则和高斯分布的关系：https://blog.csdn.net/saltriver/article/details/57544704

1. 回归任务
2. 数值特征不大（防止loss太大，继而引起梯度大，梯度爆炸）
3. 问题维度不高（loss本身比较简单，高纬度的还是得要更复杂的loss支撑）

## Elastic Net 弹性网络

$$
L=loss(x,w)+(\lambda||w||_1+(1-\lambda)||w||_2^2)
$$

优点：

弹性网络回归是 Lesso 回归和岭回归技术的混合体，它使用了 L1 和 L2 正则化，也具备了两种技术各自的优点。

在进行特征选择过程中出现一些高度相关的变量时，弹性网络更容易考虑到特征的群体效应，而不像 Lasso 那样将其中一些置为 0。所以当某个特征和另一个特征高度相关的时候弹性网络非常有用。Lasso 倾向于随机选择其中一个，而弹性网络倾向于选择两个。 所以弹性网络对所选变量的数量没有限制（多重共线性变量可以是多个）。

## 区别
1）解决不可逆的情况，使得可直接求得参数解析式

2）防止参数值太高导致模型不稳定，降低模型复杂度

3）通过限制条件，防止过拟合

相同点：

防止模型过于复杂，通过对损失函数加入正则化项，防止模型过拟合

假设模型参数服从一个分布，通过加入先验知识，限制模型参数服从一定的分布

不同点:

L2正则化相比L1对大数值参数w更敏感，惩罚粒度也会更大

L1相比L2会更容易获得稀疏解

一般情况下，L2 范数往往比 L1 范数表现的更好。

|项目|	L1（Lasso）|	L2（Ridge）|
|---|---|---|
防过拟合（泛化能力）|	有💡|	有💡
稀疏编码|	有💡|	没有
特征选择|	有💡|	没有
解析解	|没有|	有💡
解决不适定（病态）问题|	没有	|有💡
解个数|	多解	|唯一解💡
特征系数|	大	|小💡
拟合性能|	弱	|强💡
离群值抗干扰能力|	强💡	|弱
计算效率|	低（非稀疏情况）|	高💡
零附近下降速度|	快💡	|慢
先验假设|	参数服从拉普拉斯分布	|参数服从高斯分布


|英文名|中文翻译|范数|约束方式|优点|
|---|---|---|---|---|
LASSO|	最小绝对衰减和选择子|	L1|	绝对值和|	特征选择
Ridge|	岭回归	|L2	|平方和	|性能稍好，方便计算
Elastic Net	|弹性网络|	L1、L2	|绝对值和、平方和|	结合 LASSO 和 Ridge 的优点

## L1、L2正则化预期分布

在机器学习建模中, 我们知道了和以后 $y$, 需要对参数w进行建模。那么后验概率表达式如下。 
$$
M A P=\log P(y \mid X, w) P(w)=\log P(y \mid X, w) + \log P(w)
$$
我可以看出来后验概率函数为在似然函数的基础上增加了 $\log P(w), P(w)$ 的 意义是对权重系数 $w$ 的概率分布的先验假设, 在收集到训练样本 $X, y$ 后, 则可根据 $w$ 在 $X, y$ 下的后验概率对 $w$ 进行修正, 从而做出对 $w$ 的更好 地估计。若假设 $w$ 的先验分布为 0均值的高斯分布, 即
$$
w \sim N\left(0, \sigma^{2}\right)
$$
则有
$$
\begin{aligned}
&\log P(w)=\log \prod_{j} P\left(w_{j}\right)= \\
&\log \prod_{j}\left[\frac{1}{\sqrt{2 \pi} \sigma} e^{\left.-\frac{\left(w_{j}\right)^{2}}{2 \sigma^{2}}\right]}\right. \\
&=-\frac{1}{2 \sigma^{2}} \sum_{j} w_{j}^{2}+C \\
&P\left(w_{j}\right)=\frac{1}{\sqrt{2 a}} e^{\frac{\mid w j}{a}}
\end{aligned}
$$
则有
$$
\begin{aligned}
&\log P(w)=\log \prod_{j} P\left(w_{j}\right) \\
&\log \prod_{j}\left[\frac{1}{\sqrt{2 a} \sigma} e^{-\frac{|w_{j}|}{a}}\right] \\
&=-\frac{1}{2 a} \sum_{j}\left|w_{j}\right|+C
\end{aligned}
$$
## Initialization methods
通过对模型参数的初始化，让模型一开始有个好的状态。

## Update methods
在梯度更新过程中，有两方面影响最终权重的计算，一方面是梯度更新规则，常见的有SGD，momentum，RMSProp，Adam等，另一方面权重和梯度的值的更改，比如在梯度上注入噪声值等。

## Early Stopping
在恰当时刻结束模型训练也有可能提高模型的泛化能力，而常见的是通过early stopping准则，通过模型在validation的测试表现，保存结果最优的模型，防止模型一直训练，可能在训练集上表现越来越好，但模型过拟合了，有可能某个时刻，在测试集上已经开始变差。

# 范数
## $L_n$范数
$$
||x||_n=n\sqrt{\sum_{i=1}^m x_i^n}
$$
## $L_∞$ 范数
$$||𝑥||_∞=𝑚𝑎𝑥(|𝑥_𝑖|)$$
$$||𝑥||_{−∞}=𝑚𝑖𝑛(|𝑥_𝑖|)$$


# 归一化

## min-max normalization
将值缩放至[0,1）
## Mean normalization
将值缩放至[-1,1]
## 标准差标准化
经过处理的数据符合标准正态分布，即均值为0，标准差为1。batch/layer normalization是标准化的一种特殊情况。该种归一化方式要求原始数据的分布可以近似为高斯分布，否则归一化的效果会变得很糟糕。
## 缩放特征向量
使得整个向量的长度为1，将该向量的每个元素除以向量的欧几里德长度。

## 归一化需求
归一化不是由方法决定的，而是由数据决定的

树模型一般不需要做归一化处理，
做归一化处理的目的主要为了使同一特征的取值在同一量纲，降低方差太大带来的影响。

树模型并不关心特征的具体取值，只关心特征取值的分布。

概率模型不需要归一化，因为它们不关心变量的量纲，而是关心变量的分布和变量之间的条件概率，如决策树，GBDT

GBDT的树是在上一颗树的基础上通过梯度下降求解最优解，归一化能收敛的更快。所以可以不归一化，但收敛变慢

需要归一化的模型：

1.基于距离计算的模型：KNN。

2.通过梯度下降法求解的模型：线性回归、逻辑回归、支持向量机、神经网络。


# 双塔模型
双塔模型最大的特点就是「user和item是独立的两个子网络」，左侧是用户塔，右侧是item塔，这两个塔的参数不共享。

「User特征」主要包括和用户相关的特征：用户id、手机系统、地域、年龄、历史行为序列等

「Item特征」主要包括和Item相关的特征：ItemId、Item类别、Item来源等

之后我们将User特征和Item特征分别输入到特征提取网络（比如说DNN等）得到User Embedding和Item Embedding。之后我们可以计算这两个Embedding之间的余弦距离。「用户点击过的Item其距离更近，用户没有点击过或者讨厌的Item其距离更远」。之后利用算得的loss来更新模型的参数。通过Item侧塔，离线将所有Item转化成Embedding，并存储进ANN检索系统，比如FAISS，以供查询。

当来了一个用户后，我们首先利用User塔计算得到其用户向量；

之后拿用户向量去FAISS中和Item向量进行相似度计算，并返回距离最近的Top K个Item作为个性化的召回结果。

faiss就是一个相似向量查找的数据库。

速度快，但牺牲掉模型的部分精准性，而且这个代价是结构内生的，也就是说它这种结构必然会面临这样的问题。

## 目前主流的双塔模型结构

第一类
在离线阶段直接将BERT编码的document映射为固定长度的向量，在线阶段将query映射为固定长度的向量，然后通过打分函数计算最后的得分，例如：Sentence-BERT，DPR。

第二类
模型在离线阶段将BERT编码document得到的多个向量（每个向量对应一个token）全部保留，在线阶段利用BERT将query编码成多个向量，和离线阶段保留的document的多个向量进行交互打分（复杂度O(mn))，得到最后的得分，代表工作，Col-BERT。

第三类
模型是前两种的折中，将离线阶段BERT编码得到的document向量进行压缩，保留k个（k < m)个向量，并且使用一个向量来表示query（一般query包含的信息较少），在线阶段计算一个query向量和k个document向量的交互打分（复杂度O(k))，代表工作：Poly-BERT，PQ-BERT。

这类工作的主要思想是增强双塔模型的向量表示能力，由于document较长，可能对应多种语义，而原始双塔模型对query和document只使用一个向量表示，可能造成语义缺失。那么可以使用多个向量来表示document，在线阶段再进行一些优化来加速模型的推断。
## 问题
双塔两侧特征交互过晚

双塔结构，人工筛选的信息，来自两侧的特征组合就不能用，因为它既不能放在User侧，也不能放在Item侧，这是特征工程方面带来的效果损失。

## 解决方案
模型结构，训练样本构造，模型目标函数

### 训练样本构造
负采样，让训练数据中包含更多的负样本，这样可以提高模型的泛化能力，提高模型的鲁棒性。这类工作从训练数据着手，弥补原有的训练模式对于缺少负样本优化的不足。

### 训练目标改进
利用交互模型改进双塔模型，交互模型的表现更好，但复杂度更高，因此很多工作的idea是通过模型蒸馏将交互模型的文本表示能力迁移到双塔模型中。

### 模型目标函数

一般的预训练模型使用的目标函数主要是MLM或者seq2seq，这种预训练目标和双塔的匹配任务还是有一些不一致。并且已有的预训练模型即使有匹配任务（比如NSP），也是使用交互而非双塔的工作方式。可以通过对比学习，提升模型对句子的表示能力。

## 双塔模型优势，缺点，如何改进
双塔模型的优势是速度快，但模型精度还有待提升。

速度快是因为将所有Item转化成Embedding，并存储进ANN检索系统，比如FAISS，以供查询。类似FAISS这种ANN检索系统对海量数据的查询效率高。

而双塔模型为了速度快，必须对用户和Item进行特征分离，而特征分离，又必然导致上述两个原因产生的效果损失。

改进：SENet双塔模型，把SENet放在Embedding层之上，目的是通过SENet网络，动态地学习这些特征的重要性：对于每个特征学会一个特征权重，然后再把学习到的权重乘到对应特征的Embedding里，这样就可以动态学习特征权重，通过小权重抑制噪音或者无效低频特征，通过大权重放大重要特征影响的目的。

# focal loss
## 样本不均衡问题
使用交叉熵损失函数，当样本分布失衡时，在损失函数L的分布也会发生倾斜，如m<< n时，负样本就会在损失函数占据主导地位。由于损失函数的倾斜，模型训练过程中会倾向于样本多的类别，造成模型对少样本类别的性能较差。

基于样本非平衡造成的损失函数倾斜, 一个直观的做法就是在损失函数中添加权重因子, 提高少数 类别在损失函数中的权重, 平衡损失函数的分布。如在上述二分类问题中, 添加权重参数 $\alpha \in[0,1]$ 和 $1-\alpha$
$$
L=\frac{1}{N}\left(\sum_{y_{i}=1}^{m}-\alpha \log (\hat{p})+\sum_{y_{i}=0}^{n}-(1-\alpha) \log (1-\hat{p})\right)
$$
其中 $\frac{\alpha}{1-\alpha}=\frac{n}{m}$, 即权重的大小根据正负样本的分布进行设置。
## 解决
focal loss针对分类问题中类别不平衡、分类难度差异的一个 loss。

focal loss从样本难易分类角度出发，解决样本非平衡带来的模型训练问题。loss聚焦于难分样本，解决了样本少的类别分类准确率不高的问题，当然难分样本不限于样本少的类别，也就是focal loss不仅仅解决了样本非平衡的问题，同样有助于模型的整体性能提高。
$$
\begin{aligned}
L_{f l}&= \begin{cases}-\alpha (1-\hat{p})^{\gamma} \log (\hat{p}) & \text { if }~~ y=1 \\
-(1-\alpha)\hat{p}^{\gamma} \log (1-\hat{p}) & \text { if }~~y=0\end{cases} \\
&\text { 令 } p_{t}= \begin{cases}\hat{p} & \text { if } \mathrm{y}=1 \\
1-\hat{p} & \text { if }y=0\end{cases}
\end{aligned}
$$
$p_{t}$ 反映了与ground truth即类别 $\mathrm{y}$ 的接近程度, $p_{t}$ 越大说明越接近类别 $\mathrm{y}$, 即分类越准确。
$\gamma,\alpha>0$ 为可调节因子。
对比表达式(3)和(4), focal loss相比交叉熵多了一个modulating factor即 $\left(1-p_{t}\right)^{\gamma}$ 。对于分类 准确的样本 $p_{t} \rightarrow 1$, modulating factor趋近于0。对于分类不准确的样本 $1-p_{t} \rightarrow 1$, modulating factor趋近于1。即相比交叉熵损失, focal loss对于分类不准确的样本, 损失没有改变, 对于分类准确的样本, 损失会变小。整体而言, 相当于增加了分类不准确样本在损失函数中的权重。

$p_{t}$ 也反应了分类的难易程度, $p_{t}$ 越大, 说明分类的置信度越高, 代表样本越易分; $p_{t}$ 越小, 分类的置信度越低, 代表样本越难分。因此focal loss相当于增加了难分样本在损失函数的权重, 使得损失函数倾向于难分的样本, 有助于提高难分样本的准确度。

比如负样本远比正样本多的话，模型肯定会倾向于数目多的负类（可以想象全部样本都判为负类），这时候，负类的 $\hat{p}$  γ 都很小，而正类的 (1−$\hat{p}$  )γ 或 就很大，这时候模型就会开始集中精力关注正样本。

# Hadoop

Hadoop就是一个实现了Google云计算系统的开源系统，包括并行计算模型Map/Reduce，分布式文件系统HDFS，以及分布式数据库Hbase

Mapreduce是一个分布式运算程序的编程框架，是用户开发“基于hadoop的数据分析应用”的核心框架；

Mapreduce核心功能是将用户编写的业务逻辑代码和自带默认组件整合成一个完整的分布式运算程序，并发运行在一个hadoop集群上；

## 为什么要MapReduce
（1）海量数据在单机上处理因为硬件资源限制，无法胜任

（2）而一旦将单机版程序扩展到集群来分布式运行，将极大增加程序的复杂度和开发难度

（3）引入mapreduce框架后，开发人员可以将绝大部分工作集中在业务逻辑的开发上，而将分布式计算中的复杂性交由框架来处理，提高开发效率。

# 位置编码
## 绝对位置编码
0，1，2，3.。。。再embedding

transformer 的三角函数编码
## 可学习位置编码
bert 位置编码是参数，直接对不同的位置随机初始化一个postion embedding，加到word embedding上输入模型，作为参数进行训练。
## 相对位置编码
### 经典相对位置编码
一般认为, 相对位置编码是由绝对位置编码启发而来, 考虑一般的带绝对位置编码的Attention:
$$
\left\{\begin{aligned}
\boldsymbol{q}_{i} &=\left(\boldsymbol{x}_{i}+\boldsymbol{p}_{i}\right) \boldsymbol{W}_{Q} \\
\boldsymbol{k}_{j} &=\left(\boldsymbol{x}_{j}+\boldsymbol{p}_{j}\right) \boldsymbol{W}_{K} \\
\boldsymbol{v}_{j} &=\left(\boldsymbol{x}_{j}+\boldsymbol{p}_{j}\right) \boldsymbol{W}_{V} \\
a_{i, j} &=\operatorname{softmax}\left(\boldsymbol{q}_{i} \boldsymbol{k}_{j}^{\top}\right) \\
\boldsymbol{o}_{i} &=\sum_{j} a_{i, j} \boldsymbol{v}_{j}
\end{aligned}\right.
$$
其中softmax对 $j$ 那一维归一化, 这里的向量都是指行向量。我们初步展开 $\boldsymbol{q}_{i} \boldsymbol{k}_{j}^{\top}$ :
$$
\boldsymbol{q}_{i} \boldsymbol{k}_{j}^{\top}=\left(\boldsymbol{x}_{i}+\boldsymbol{p}_{i}\right) \boldsymbol{W}_{Q} \boldsymbol{W}_{K}^{\top}\left(\boldsymbol{x}_{j}+\boldsymbol{p}_{j}\right)^{\top}=\left(\boldsymbol{x}_{i} \boldsymbol{W}_{Q}+\boldsymbol{p}_{i} \boldsymbol{W}_{Q}\right)\left(\boldsymbol{W}_{K}^{\top} \boldsymbol{x}_{j}^{\top}+\boldsymbol{W}_{K}^{\top} \boldsymbol{p}_{j}^{\top}\right)
$$
为了引入相对位置信息, Google把第一项位置去掉, 第二项 $\boldsymbol{p}_{j} \boldsymbol{W}_{K}$ 改为二元位置向量 $\boldsymbol{R}_{i, j}^{K}$, 变成
$$
a_{i, j}=\operatorname{softmax}\left(\boldsymbol{x}_{i} \boldsymbol{W}_{Q}\left(\boldsymbol{x}_{j} \boldsymbol{W}_{K}+\boldsymbol{R}_{i, j}^{K}\right)^{\top}\right)
$$
$$
\begin{array}{r}
\text { 以及 } \boldsymbol{o}_{i}=\sum_{j} a_{i, j} \boldsymbol{v}_{j}=\sum_{j} a_{i, j}\left(\boldsymbol{x}_{j} \boldsymbol{W}_{V}+\boldsymbol{p}_{j} \boldsymbol{W}_{V}\right) \text { 中的 } \boldsymbol{p}_{j} \boldsymbol{W}_{V} \text {换成} \boldsymbol{R}_{i, j}^{V}
\end{array}
$$
$$
\begin{array}{r}
\boldsymbol{o}_{i}=\sum_{j} a_{i, j}\left(\boldsymbol{x}_{j} \boldsymbol{W}_{V}+\boldsymbol{R}_{i, j}^{V}\right)
\end{array}
$$
所谓相对位置, 是将本来依赖于二元坐标 $(i, j)$ 的向量 $\boldsymbol{R}_{i, j}^{K}, \boldsymbol{R}_{i, j}^{V}$, 改为只依赖于相对距离 $i-j$, 并且通 常来说会进行截断, 以适应不同任意的距离
$$
\begin{aligned}
&\boldsymbol{R}_{i, j}^{K}=\boldsymbol{p}_{K}\left[\operatorname{clip}\left(i-j, p_{\min }, p_{\max }\right)\right] \\
&\boldsymbol{R}_{i, j}^{V}=\boldsymbol{p}_{V}\left[\operatorname{clip}\left(i-j, p_{\min }, p_{\max }\right)\right]
\end{aligned}
$$
这样一来, 只需要有限个位置编码, 就可以表达出任意长度的相对位置（因为进行了截断）, 不管 $\boldsymbol{p}_{K}, \boldsymbol{p}_{V}$ 是选择可训练式的还是三角函数式的, 都可以达到处理任意长度文本的需求。

### XLNET/Transformer-XL
XLNET式位置编码其实源自Transformer-XL的论文 《Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context》, 只不过因为使用了Transformer-XL架构的XLNET模型并在一定 程度上超过了BERT后, Transformer-XL才算广为人知，因此这种位置编码通常也被冠以XLNET之 名。
XLNET式位置编码源于对上述 $\boldsymbol{q}_{i} \boldsymbol{k}_{j}^{\top}$ 的完全展开:
$$
\boldsymbol{q}_{i} \boldsymbol{k}_{j}^{\top}=\boldsymbol{x}_{i} \boldsymbol{W}_{Q} \boldsymbol{W}_{K}^{\top} \boldsymbol{x}_{j}^{\top}+\boldsymbol{x}_{i} \boldsymbol{W}_{Q} \boldsymbol{W}_{K}^{\top} \boldsymbol{p}_{j}^{\top}+\boldsymbol{p}_{i} \boldsymbol{W}_{Q} \boldsymbol{W}_{K}^{\top} \boldsymbol{x}_{j}^{\top}+\boldsymbol{p}_{i} \boldsymbol{W}_{Q} \boldsymbol{W}_{K}^{\top} \boldsymbol{p}_{j}^{\top}
$$
Transformer-XL的做法很简单, 直接将 $\boldsymbol{p}_{j}$ 替换为相对位置向量 $\boldsymbol{R}_{i-j}$, 至于两个 $\boldsymbol{p}_{i}$, 则干脆替换为两个 可训练的向量 $u, v$ :
$$
\boldsymbol{x}_{i} \boldsymbol{W}_{Q} \boldsymbol{W}_{K}^{\top} \boldsymbol{x}_{j}^{\top}+\boldsymbol{x}_{i} \boldsymbol{W}_{Q} \boldsymbol{W}_{K}^{\top} R_{i-j}^{\top}+u \boldsymbol{W}_{Q} \boldsymbol{W}_{K}^{\top} \boldsymbol{x}_{j}^{\top}+v \boldsymbol{W}_{Q} W_{K}^{\top} R_{i-j}^{\top}
$$
该编码方式中的 $\boldsymbol{R}_{i-j}$ 没有像式(6)那样进行截断，而是直接用了Sinusoidal式的生成方案。此外, $\boldsymbol{v}_{j}$ 上 的位置偏置就直接去掉了, 即直接令 $o_{i}=\sum_{j} a_{i, j} x_{j} W_{V}$ 。似乎从这个工作开始, 后面的相对位置编码 都只加到Attention矩阵上去, 而不加到 $\boldsymbol{v}_{j}$ 上去了。

$$
\begin{aligned}
p_{i, j}[2k] &=\sin \left(\frac{j-i}{10000^{2k / d}}\right) \\
p_{i, j}[2k+1] &=\cos \left(\frac{j-i}{10000^{2k / d}}\right)
\end{aligned}
$$

2k是向量的第几个数，i,j是索引位置。

是在attention阶段加入。

### T5式
T5模型出自文章《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》，里边用到了一种更简单的相对位置编码。思路依然源自展开式(7)，如果非要分析每一项的含义，那么可以分别理解为“输入-输入”、“输入-位置”、“位置-输入”、“位置-位置”四项注意力的组合。如果我们认为输入信息与位置信息应该是独立（解耦）的，那么它们就不应该有过多的交互，所以“输入-位置”、“位置-输入”两项Attention可以删掉，剩下的位置编码换成相对位置编码我们可以直接将它作为参数训练出来。它仅仅是在Attention矩阵的基础上加一个可训练的偏置项而已。

### DeBERTa式
其实DeBERTa的主要改进也是在位置编码上，同样还是从展开式(7)出发，T5是干脆去掉了第2、3项，只保留第4项并替换为相对位置编码，而DeBERTa则刚刚相反，它扔掉了第4项，保留第2、3项并且替换为相对位置编码。

DeBERTa比较有意思的地方，是提供了使用相对位置和绝对位置编码的一个新视角，它指出NLP的大多数任务可能都只需要相对位置信息，但确实有些场景下绝对位置信息更有帮助，于是它将整个模型分为两部分来理解。以Base版的MLM预训练模型为例，它一共有13层，前11层只是用相对位置编码，这部分称为Encoder，后面2层加入绝对位置信息，这部分它称之为Decoder，还弄了个简称EMD（Enhanced Mask Decoder）；至于下游任务的微调截断，则是使用前11层的Encoder加上1层的Decoder来进行。

### 旋转位置编码 RoPE
RoPE的基本思路是通过绝对位置编码的方式，使得模型可以注意到相对位置。其思路和三角式位置编码有一些相似，实际上其形式也恰巧与三角函数也有所相似。
### 复数式位置编码

### CNN式

# LR和SVM的比较
## 相同
第一，LR和SVM都是分类算法（SVM也可以用与回归），且一般都用于处理线性二分类问题（在改进的情况下可以处理多分类问题）。

第二，如果不考虑核函数，LR和SVM都是线性分类算法，也就是说他们的分类决策面都是线性的。

第三，LR和SVM都是监督学习算法。

第四，LR和SVM都是判别模型。
## 区别
1. LR是参数模型，SVM是非参数模型。
2. 从目标函数来看，区别在于逻辑回归采用的是logistical loss，SVM采用的是hinge loss.这两个损失函数的目的都是增加对分类影响较大的数据点的权重，减少与分类关系较小的数据点的权重。
3. SVM的处理方法是只考虑support vectors，也就是和分类最相关的少数点，去学习分类器。而逻辑回归通过非线性映射，大大减小了离分类平面较远的点的权重，相对提升了与分类最相关的数据点的权重。
4. 逻辑回归相对来说模型更简单，好理解，特别是大规模线性分类时比较方便。而SVM的理解和优化相对来说复杂一些，SVM转化为对偶问题后,分类只需要计算与少数几个支持向量的距离,这个在进行复杂核函数计算时优势很明显,能够大大简化模型和计算。
5. logic 能做的 svm能做，但可能在准确率上有问题，svm能做的logic有的做不了。

第一，本质上是其loss function不同

    LR是对数损失函数，类似于交叉熵损失函数。
    
    SVM是带L2正则的合叶损失函数。
    
    逻辑回归方法基于概率理论，假设样本为1的概率可以用sigmoid函数来表示，然后通过极大似然估计取对数的方法估计出参数的值
    
    支持向量机基于几何间隔最大化原理，认为存在最大几何间隔的分类面为最优分类面

第二，支持向量机只考虑局部的边界线附近的点，而逻辑回归考虑全局（远离的点对边界线的确定也起作用，虽然作用会相对小一些）

    SVM决策面的样本点只有少数的支持向量，当在支持向量外添加或减少任何样本点对分类决策面没有任何影响
    
    LR中，每个样本点都会影响决策面的结果。
    
    线性SVM不直接依赖于数据分布，分类平面不受非支持向量点影响；
    
    LR则受所有数据点的影响，如果数据不同类别strongly unbalance，一般需要先对数据做balancing
    
    LR对异常值敏感，SVM对异常值不敏感。
    
    在训练集较小时，SVM较适用，而LR需要较多的样本。

第三，在解决非线性问题时，支持向量机采用核函数的机制，而LR通常不采用核函数的方法，LR主要靠特征构造，必须组合交叉特征，特征离散化。

第四，​线性SVM依赖数据表达的距离测度，所以需要对数据先做normalization，LR不受其影响

    一个基于概率，一个基于距离

第五，SVM的损失函数就自带正则，这就是为什么SVM是结构风险最小化算法的原因，而LR必须另外在损失函数上添加正则项

    所谓结构风险最小化，意思就是在训练误差和模型复杂度之间寻求平衡，防止过拟
    合，从而达到真实误差的最小化。未达到结构风险最小化的目的，最常用的方法就是
    添加正则项

如果Feature的数量很大，跟样本数量差不多，这时候选用LR或者是Linear Kernel的SVM

如果Feature的数量比较小，样本数量一般，不算大也不算小，选用SVM+Gaussian Kernel

如果Feature的数量比较小，而样本数量很多，需要手工添加一些feature变成第一种情况。(LR和不带核函数的SVM比较类似。)
# 防止欠拟合、过拟合、梯度爆炸、梯度消失、网络退化的方式
## 欠拟合
定义：模型无法得到较低的训练误差

1）增大训练量，增大数据集

2）减少特征数量，减少模型复杂度

## 过拟合
定义：模型在训练集上的表现很好，但在测试集和新数据上的表现很差。

1）重新清洗数据，导致过拟合的一个原因也有可能是数据不纯导致的，如果出现了过拟合就需要我们重新清洗数据。

2）减小训练量，增大数据集

3）采用正则化方法，加上正则项。

4）droppout

5）mask方法

6）降低模型复杂度

7）数据增强

8）修改优化算法、损失函数。如使用focal loss、adamW

9）早停止 Early Stop

10）调整学习率

11）集成学习，bagging、boosting方法

## 梯度消失、梯度爆炸
1、对于RNN，可以通过梯度截断，避免梯度爆炸

2、可以通过添加正则项，避免梯度爆炸

3、使用LSTM/GRU等自循环和门控制机制，可以记忆一些长期的信息,相应的也就保留了更多的梯度，避免梯度消失。

4、优化激活函数

5、Normalization

## 网络退化
随着网络层数的增加，网络会发生退化现象：随着网络层数的增加训练集loss逐渐下降，然后趋于饱和，如果再增加网络深度的话，训练集loss反而会增大，注意这并不是过拟合，因为在过拟合中训练loss是一直减小的。

残差链接。 

# 残差链接作用
1. 防止梯度爆炸/增量爆炸
   
   模型每一步的更新量是正比于模型深度N的（宽度不在本文讨论范围），如果模型越深，那么更新量就越大，这意味着初始阶段模型越容易进入不大好的局部最优点，然后训练停滞甚至崩溃，这就是“增量爆炸”问题。
2. 治标之法
   
   “增量爆炸”就是在层数变多时，参数的微小变化就会导致损失函数的大变化，这对于模型的训练，特别是初始阶段的训练时尤其不利的。对此，一个直接的应对技巧就是Wamrup，初始阶段先用极小的学习率，然后再慢慢增大，避免在初始阶段学习过快。待模型平稳渡过初始阶段的“危险期”后，就可以正常训练了。

    然而，尽管Wamrup能起到一定的作用，但其实是“治标不治本”的，因为“参数的微小变化就会导致损失函数的大变化”意味着模型本身的抖动很大，用更专业的话说就是模型的landscape极度不平滑了，这不是一个好模型应该具备的性质。因此，我们应该通过修改模型来解决这个问题，而不是通过降低学习率这种“表面”方法。
3. 稳定传播
   
   如果只是纯粹地缩小梯度，那么很简单，只要尽量降低初始化方差就行。但实际上我们在缩小梯度的同时，必须还要保持前向传播稳定性，因为前向传播的稳定性是我们对所做任务的一种先验知识，它意味着是模型更好的起点。
4. 因为残差结构是可以同时稳定前向传播和反向传播、并且可以缩放参数梯度以解决增量爆炸的一种设计，它能帮助我们训练更深层的模型。


# Feature-based和Fine-tune的区别
Feature-based指利用语言模型的中间结果也就是LM embedding, 将其作为额外的特征，引入到原任务的模型中。

通常feature-based方法包括两步：

1. 首先在大的语料A上无监督地训练语言模型，训练完毕得到语言模型
2. 然后构造task-specific model例如序列标注模型，采用有标记的语料B来有监督地训练task-sepcific model，将语言模型的参数固定，语料B的训练数据经过语言模型得到LM embedding，作为task-specific model的额外特征

feature-based只变化了最后一层的参数，ELMo是这方面的典型工作。

Fine-tuning方式是指在已经训练好的语言模型的基础上，加入少量的task-specific parameters, 例如对于分类问题在语言模型基础上加一层softmax网络，然后在新的语料上重新训练来进行fine-tune。

LM + Fine-Tuning的方法工作包括两步：

1. 构造语言模型，采用大的语料A来训练语言模型
2. 在语言模型基础上增加少量神经网络层来完成specific task例如序列标注、分类等，然后采用有标记的语料B来有监督地训练模型，这个过程中语言模型的参数并不固定，依然是trainable variables.

# bert蒸馏
知识蒸馏的本质是让超大线下 teacher model来协助线上student model的training。

bert的知识蒸馏，大致分成两种。
1. 从transformer到非transformer框架的知识蒸馏

    这种由于中间层参数的不可比性，导致从teacher model可学习的知识比较受限。但比较自由，可以把知识蒸馏到一个非常小的model，但效果肯定会差一些。
2. 从transformer到transformer框架的知识蒸馏

    由于中间层参数可利用，所以知识蒸馏的效果会好很多，甚至能够接近原始bert的效果。但transformer即使只有三层，参数量其实也不少，另外蒸馏过程的计算也无法忽视。

所以最后用那种，还是要根据线上需求来取舍。

# 给你一些很稀疏的特征，用LR还是树模型
很稀疏的特征表明是高维稀疏，用树模型（GBDT）容易过拟合。建议使用加正则化的LR。

lr 等线性模型的正则项是对权重的惩罚，也就是 W1一旦过大，惩罚就会很大，进一步压缩 W1的值，使他不至于过大，而树模型则不一样，树模型的惩罚项通常为叶子节点数和深度等。这也就是为什么在高维稀疏特征的时候，线性模型会比非线性模型好的原因了：带正则化的线性模型比较不容易对稀疏特征过拟合。

# 为什么分类用交叉熵不用MSE（均方误差）

先用2分类LR求导做例子

使用平方损失函数，会发现梯度更新的速度和sigmod函数本身的梯度是很相关的。sigmod函数在它在定义域内的梯度都不大于0.25。这样训练会非常的慢。使用交叉熵的话就不会出现这样的情况，它的导数就是一个差值，误差大的话更新的就快，误差小的话就更新的慢点，这正是我们想要的。

在使用 Sigmoid 函数作为正样本的概率时，同时将平方损失作为损失函数，这时所构造出来的损失函数是非凸的，不容易求解，容易得到其局部最优解。如果使用极大似然，其目标函数就是对数似然函数，该损失函数是关于未知参数的高阶连续可导的凸函数，便于求其全局最优解。

原因一，使用交叉熵loss下降的更快；

原因二，使用交叉熵是凸优化，MSE是非凸优化

回归问题常用mse作为损失函数，这里面一个隐含的预设是数据误差符合高斯分布。交叉熵则是以数据分布服从多项式分布为前提。损失函数的选择本身也是一种先验偏好，选择mse意味着你认为数据误差符合高斯分布，选择交叉熵则表示你倾向于认为数据接近多项式分布。

缺点：

sigmoid(softmax)+cross-entropy loss 擅长于学习类间的信息，因为它采用了类间竞争机制，它只关心对于正确标签预测概率的准确性，忽略了其他非正确标签的差异，导致学习到的特征比较散。

回归用MSE，分类用交叉熵

# 参数模型/非参数模型
二者最主要的区别是关于数据分布的假设—参数模型对数据分布 (distribution，density）有假设，而非参数模型对数据分布假设自由(distribution一free)，但是对数据必须可以排序（rank， score)。所以，回顾二者的名字”参数”，即指数据分布的参数。

参数模型：对数据的分布 (Distributions， or Density）有理想的假设，模型更加robust；然而现实的情况是，往往数据不足以提供给我们机会去判断分布、抑或本身没有明显的特征。这时，非参数模型，更加关注数据先后顺序(Ranks，Scores)便显得更加powerful，数据越多非参数模型相应的越复杂。

其次，是方法上的区别：参数模型正如其名，利用数据的数量关系及其分布进行检验和预测 (tests and inference)；然而，非参数模型（如果可能，可称作：排序模型Q(rank models)），利用数据本身的排序进行检验和预测。特别的，排序只（ranks）是分数（scores） 的一种特殊情况。

从方法可以看出，非参数模型的建立亦有其假设，即对数据可以排序。换言之，如果数据中有相等量（ties)，会影响其排序，从而影响信度。

如果一个学习模型，它的参数量是固定的，参数的规模跟训练的数据无关系，不会随着训练数据增加而变化使模型更加庞大，那就是参数模型。

参数模型一般包含两步：
1. 选择一个确定形式的函数（比如线性的，非线性的，参数多少个等）；
2. 从训练数据中学习到该函数的系数。

Logistic Regression

Linear Discriminant Analysis

Perceptron感知机

Naive Bayes

Simple Neural Networks

非参数模型就是不会用固定的参数，确定的函数形式描述的模型，模型是跟训练数据有关的，模型的参数会随着训练数据增加而增加。

k-Nearest Neighbors

Decision Trees like CART and C4.5

Support Vector Machines

## 优缺点
参数模型的优点：

1. 更加的简单：方法更好理解，结果解释性更好；
2. 学习的更快；
3. 需要更少的数据也可以得到比较好的训练结果
缺点：
1. 因为一开始已经限定了形式，通常可能并不符合学习的问题的分布，所以偏差会比较大，不适应比较复杂的问题。
2. 拘束：以指定的函数形式来指定学习方式。
3. 有限的复杂度：通常只能应对简单的问题。
4. 拟合度小：实际中通常无法和潜在的目标函数完全吻合，也就是容易出现欠拟合。

综上，如果数据量不大，问题比较简单，应该优先考虑参数模型。

非参数机器学习算法对目标函数形式不做过多的假设，因此算法可以通过对训练数据进行拟合而学习出某种形式的函数。

当你拥有许多数据而先验知识很少时，非参数学习通常很有用，此时你不需要关注于参数的选取。

非参数模型的优点：

1. 灵活且强大；因为不限定形式，就能够适应更多的分布；
2. 训练得到的模型更好。
缺点：

1. 需要更多的数据
2. 因为一般需要学习更多的参数，所以训练的会更慢
3. 解释性更差，更容易过拟合 （方差偏大）


# transformer、RNN、CNN对比
## transformer
避免递归，不依赖于过去的隐藏状态来捕获对先前单词的依赖性，而是整体上处理一个句子，以便允许并行计算（减少训练时间），并减少由于长期依赖性而导致的性能下降。

但是长文本信息之间的依赖性减弱，因此要限制文本长度。诸如Transformer-XL试图通过引入已存储的已编码语句的隐藏状态以在后续语句的后续编码中利用它们的隐含状态来重新引入递归，从而完全解决此问题。
## RNN
对长文本信息提取更充分，依赖性更强，更适合做生成任务。
## CNN
运算速度快，可以提取段距依赖，但捕获句子中单词的所有可能组合之间的依存关系所需的不同内核的数量将是巨大且不切实际的，因为在增加输入句子的最大长度时，组合的数量呈指数增长。
## transformer的并行化
RNN之所以不支持并行化是因为它天生是个时序结构，t时刻依赖t-1时刻的输出，而t-1时刻又依赖t-2时刻，如此循环往前，我们可以说t时刻依赖了前t时刻所有的信息。RNN的实质目的是学习时序数据中的局部依赖关系，实际上深度学习很多模型的目的都是为了学习样本之间的局部关系，如CNN也是为了学习空间数据中局部依赖关系。为了完成对长距离样本之间依赖关系的学习，CNN采用多次卷积、池化等操作，这样需要加大网络的深度，才能更多的学习到更大范围的样本局部依赖关系。

### Encoder支持并行化
自注意力机制就是利用 xi 之间两两相关性作为权重的一种加权平均将每一个 xi 映射到结果，即是乘性attention自己点积自己作为注意力分数。而矩阵乘法本身支持并行操作，因此Encoder支持并行化。
### Decoder支持并行化
masked self attention

训练阶段每一轮预测时，不使用上一轮预测的输出，而强制使用正确的单词。在代码中体现的是训练阶段输入的X即是预测结果，但是针对预测的位置后增加了mask。

而预测阶段会将上次预测结果和历史结果拼接作为下一阶段预测的输入。
## 总结
1. 语义特征提取能力
    Transformer在这方便的能力非常显著超过RNN和CNN，RNN和CNN两者能力差不多。

2. 长距离特征捕捉能力
  
    原生CNN特征抽取器在这方面显著弱于RNN和Transformer

    Transformer微弱优于RNN模型

    在比较远的距离上RNN微弱优于Transformer
3. 任务综合特征抽取能力（机器翻译）
  
    Transformer > 原生CNN == 原生RNN

4. 并行计算能力及运行效率
  
    Transformer和CNN差不多，都强于RNN

RNN要逐步递归才能获得全局信息，因此一般要双向RNN才比较好；CNN事实上只能获取局部信息，是通过层叠来增大感受野；Attention的思路最为粗暴，它一步到位获取了全局信息：纯Attention！单靠注意力就可以。

Attention层的好处是能够一步到位捕捉到全局的联系，因为它直接把序列两两比较（代价是计算量变为$O(n^2)$，当然由于是纯矩阵运算，这个计算量相当也不是很严重）；相比之下，RNN需要一步步递推才能捕捉到，而CNN则需要通过层叠来扩大感受野，这是Attention层的明显优势。

# 梯度截断 gradient clip norm
1. 首先设置一个梯度阈值：clip_gradient

2. 在后向传播中求出各参数的梯度，这里我们不直接使用梯度进去參数更新，我们求这些梯度的L2范数然后比较梯度的L2范数$||g||_2$ 与clip_gradient的大小

3. 如果前者大，求缩放因子$\frac{clip gradient}{||g||_2}$，由缩放因子可以看出梯度越大，则缩放因子越小，这样便很好地控制了梯度的范围
4. 最后将梯度乘上缩放因子便得到最后所需的梯度


# 停用词
是指在信息检索中，为节省存储空间和提高搜索效率，在处理自然语言数据（或文本）之前或之后会自动过滤掉某些字或词，这些字或词即被称为Stop Words（停用词）。这些停用词都是人工输入、非自动化生成的，生成后的停用词会形成一个停用词表。

# 核函数
1. 径向基函数核（RBF kernel）高斯核函数（Gaussian kernel）
2. 马顿核（Matérn kernel）
3. 指数函数核（exponential kernel）
4. 二次有理函数核（rational quadratic kernel, RQ kernel）
5. 周期核函数（periodickernel）
6. 内积核函数（dot product kernel）
7. 各向异性核函数

# 牛顿法、拟牛顿法
牛顿法是用二阶的海森矩阵的逆矩阵求解。牛顿法在选择方向时，不仅会考虑坡度是否够大，还会考虑你走了一步之后，坡度是否会变得更大。

在牛顿法的迭代中，需要计算海森矩阵的逆矩阵，这一计算比较复杂，考虑用一个n阶矩阵来近似代替森矩阵的逆矩阵。这就是拟牛顿法的基本想法。

# 项目改进
标注任务，增加数据量，交叉标注，增加冗余信息

负采样，针对被标注为实体的span和未被标注为实体但其本身是实体的span计算交叉熵损失。

EDA数据增强，增加噪声，防止过拟合

NLP的Data Augmentation大致有两条思路，一个是加噪，另一个是回译，均为有监督方法。加噪即为在原数据的基础上通过替换词、删除词等方式创造和原数据相类似的新数据。回译则是将原有数据翻译为其他语言再翻译回原语言，由于语言逻辑顺序等的不同，回译的方法也往往能够得到和原数据差别较大的新数据。

dropout

半监督

BERT-MRC模型，去掉无用的CRF

可以通过问题加入一些先验知识，减小由于数据量太小带来的问题，在实际实验中，在数据量比较小的情况下，BERT-MRC模型的效果要较其他模型要更好一点。BERT-MRC模型很适合在缺乏标注数据的场景下使用。

focal loss

active learning，需要一个外在的能够对其请求进行标注的实体(通常就是相关领域人员)，即主动学习是交互进行的。
# 序列标注
1. 进行序列标注时CRF是必须的吗？

    如果你已经将问题本身确定为序列标注了，并且正确的标注结果是唯一的，那么用CRF理论上是有正的收益的，但如果主体是BERT等预训练模型，那么可能要调一下CRF层的学习率，参考CRF层的学习率可能不够大

2. 进行NER时必须转为序列标注吗？
  
    就原始问题而言，不论是NER、词性标注还是阅读理解等，都不一定要转化为序列标注问题，既然不转化为序列标注问题，自然也就用不着CRF了。 

BERT的拟合能力太强了，就连Softmax效果都能达到最优了，转移矩阵自然也不能带来太大的提升。

# 高偏差和高方差
对于高偏差的模型增加训练数据并不会改善模型的效果

在不增加数据和特征的情况下，还可以采用复杂的模型

对于高方差的模型增加训练数据会在一定程度上改善模型的效果

增加更多的训练样本，减少训练特征，增加正则

# LR与决策树的区别
1）逻辑回归通常用于分类问题，决策树可回归、可分类。

2）逻辑回归是线性函数，决策树是非线性函数。

3）逻辑回归的表达式很简单，回归系数就确定了模型。决策树的形式就复杂了，叶子节点的范围+取值。两个模型在使用中都有很强的解释性，银行较喜欢。

4）逻辑回归可用于高维稀疏数据场景，比如ctr预估；决策树变量连续最好，类别变量的话，稀疏性不能太高。

5）逻辑回归的核心是sigmoid函数，具有无限可导的优点，常作为神经网络的激活函数。

6）在集成模型中，随机森林、GBDT以决策树为基模型，Boosting算法也可以用逻辑回归作为基模型。

# 温度系数
在原始的softmax损失基础上额外增加一个温度系数T

温度系数T主要是用来调整logits-softmax曲线的平滑程度。随着T的减小，softmax输出各类别之间的概率差距越大（陡峭），从而导致loss变小；同样，当增大T，softmax输出的各类别概率差距会越来越小（平滑），导致loss变大。

由于训练好的模型本身会出现过度自信的问题（softmax输出的概率分布熵很小），所以除以一个大于1的T，让分布变得平滑，来放大这种类别相似信息。

比如我把温度系数调低(T<1)，拉大softmax的输出分布，降低loss，不过度优化模型。当然，相反的情况下，比如我们想增加模型的判别能力，也把温度系数提高(T>1)。

记 $x_{\max }=\max \left(x_{1}, x_{2}, \cdots, x_{n}\right)$, 那么显然有
$$
e^{x_{\max }}<\sum_{i=1}^{n} e^{x_{i}} \leq \sum_{i=1}^{n} e^{x_{\max }}=n e^{x_{\max }}
$$
各端取对数即得
$$
x_{\max }<\log \operatorname{sumexp}(x) \leq x_{\max }+\log n
$$
这是关于logsumexp上下界的最基本结果, 它表明logsumexp对max的近似误差不超过 $\log n$ 。注意这 个误差跟 $x$ 本身无关, 于是我们有
$$
x_{\max } / \tau<\operatorname{logsumexp}(x / \tau) \leq x_{\max } / \tau+\log n
$$
各端乘以 $\tau$ 得到
$$
x_{\max }<\tau \operatorname{logsumexp}(x / \tau) \leq x_{\max }+\tau \log n
$$
当 $\tau \rightarrow 0$ 时, 误差就趋于o了, 这告诉我们可以通过降低温度参数来提高对max的近似程度。

$$
\operatorname{logsumexp}(x) \geq \hat{x} + logn
$$

# 最大似然估计（MLE）、最大后验概率估计（MAP）和贝叶斯估计
最大似然估计（MLE）只关注模型的参数，对参数的分布没有假设，即直接求求最大化p(x|$\theta$)的概率。将概率密度估计问题转化为参数估计问题.

最大后验概率估计（MAP）关注参数的分布，对模型的参数有假设，需要提前使用先验概率估计确定的p($\theta$)，最后求最大化p(x|$\theta$)*p($\theta$)的概率。

贝叶斯估计关注参数分布，但不会确定p($\theta$)，而是估计成分布变量进行计算。

# warmup
warmup是一种学习率优化方法（最早出现在ResNet论文中）。在模型训练之初选用较小的学习率，训练一段时间之后（如：10epoches或10000steps）使用预设的学习率进行训练；

如果不进行Wamrup，那么模型一开始就快速地学习，由于梯度消失，模型对越靠后的层越敏感，也就是越靠后的层学习得越快后面的层是以前面的层的输出为输入的，前面的层根本就没学好，所以后面的层虽然学得快，但却是建立在糟糕的输入基础上的。很快地，后面的层以糟糕的输入为基础到达了一个糟糕的局部最优点此时它的学习开始放缓（因为已经到达了它认为的最优点附近），同时反向传播给前面层的梯度信号进一步变弱，这就导致了前面的层的梯度变得不准。

# 样本不均衡
会导致模型泛化能力大大降低，对比例大的样本造成过拟合，预测偏向样本数较多的分类。

## 解决方案
1. 欠采样（undersampling）
   
    又叫下采样，减少样本数较多的样本，采用丢弃或选取部分样本的方法。但若随机丢弃负样本，可能丢失一些重要信息（导致模型只学习到总体模式的一部分 ）。其代表性算法为EasyEnsemble。

2. 过采样（oversampling）
   
    又叫上采样，增加样本数少的样本，但不能简单地对初始正样本进行重复采样（直接复制），否则会导致过拟合。可以加入轻微的随机扰动。

    其代表性算法SMOTE是通过对训练集中的正例进行插值来产生额外的正例。

3. 数据合成
   
    合成少数类样本，组合已有样本特征(从各个feature中随机选出一个已有值，拼接成一个新样本)，产生新样本。代表性方法是SMOTE，在相似样本中进行特征的随机选择并拼接出新样本。

4. 增大少数类样本权重
   
    当少数类样本被误分时，其损失值要乘上相应的权重，从而让分类器更加关注这一类数目较少的样本。类似boosting的方法。

5. focal loss

6. 数据增强
# 初始化
## Xavier
Xavier 是2010 年提出的，针对有非线性激活函数时的权值初始化方法，目标是均值为0、方差为1/m的随机分布，可以选择正态分布初始化或匀分布初始化，主要针对饱和激活函数如 sigmoid 和 tanh 等。同时考虑前向传播和反向传播。
## kaiming初始化
针对relu和L-relu。
## NTK参数化
NTK参数化：用“均值为0、方差为1的随机分布”来初始化，但是将输出结果除以$\sqrt{m}$。高斯过程中被称为“NTK参数化”

NTK参数化跟直接用Xavier初始化相比，有什么好处吗？

理论上，利用NTK参数化后，所有参数都可以用方差为1的分布初始化，这意味着每个参数的量级大致都是相同的O(1)级别，于是我们可以设置较大的学习率

总的来说，NTK参数化能让我们更平等地处理每一个参数，并且比较形象地了解到训练的更新幅度，以便我们更好地调整参数。

考虑激活函数得场景

1. tanh(x) 在x比较小的时候有tanh(x)≈x，所以可以认为 Xavier初始化直接适用于tanh激活；

2. relu时可以认为relu(y)会有 大约一半的元素被置零，所以模长大约变为原来的$\frac{1}{\sqrt{2}}$，而要保持模长不变，可以让W乘上$\sqrt{2}$，也就是说初始化方差从1/m变成2/m
# 数据增强
## EDA
Easy Data Augmentation for Text Classification Tasks （EDA）提出并验证了几种加噪的 text augmentation 技巧，分别是同义词替换（SR: Synonyms Replace）、随机插入(RI: Randomly Insert)、随机交换(RS: Randomly Swap)、随机删除(RD: Randomly Delete)

(1) 同义词替换（SR: Synonyms Replace）：不考虑stopwords，在句子中随机抽取n个词，然后从同义词词典中随机抽取同义词，并进行替换。

Eg: “我非常喜欢这部电影” —> “我非常喜欢这个影片”，句子仍具有相同的含义，很有可能具有相同的标签。

(2) 随机插入(RI: Randomly Insert)：不考虑stopwords，随机抽取一个词，然后在该词的同义词集合中随机选择一个，插入原句子中的随机位置。该过程可以重复n次。

Eg : “我非常喜欢这部电影” —> “爱我非常喜欢这部影片”。

(3) 随机交换(RS: Randomly Swap)：句子中，随机选择两个词，位置交换。该过程可以重复n次。

Eg: “如何评价 2017 知乎看山杯机器学习比赛?” —> “2017 机器学习?如何比赛知乎评价看山杯”。

(4) 随机删除(RD: Randomly Delete)：句子中的每个词，以概率p随机删除。

Eg: “如何评价 2017 知乎看山杯机器学习比赛?" —> “如何 2017 看山杯机器学习 ”。

训练数据越少，提升效果效果越明显。

同义词替换SR有一个小问题，同义词具有非常相似的词向量，而训练模型时这两个句子会被当作几乎相同的句子，但在实际上并没有对数据集进行有效的扩充。

随机插入RI很直观的可以看到原本的训练数据丧失了语义结构和语义顺序，而不考虑停用词的做法使得扩充出来的数据并没有包含太多有价值的信息，同义词的加入并没有侧重句子中的关键词，在数据扩充的多样性上实际会受限较多。

随机交换RS实质上并没有改变原句的词素，对新句式、句型、相似词的泛化能力实质上提升很有限。

随机删除RD不仅有随机插入的关键词没有侧重的缺点，也有随机交换句式句型泛化效果差的问题。随机的方法固然能够照顾到每一个词，但是没有关键词的侧重，若随机删除的词刚好是分类时特征最强的词，那么不仅语义信息可能被改变，标签的正确性也会存在问题。
## 回译
用机器翻译把一段中文翻译成另一种语言，然后再翻译回中文。

回译的方法往往能够增加文本数据的多样性，相比替换词来说，有时可以改变句法结构等，并保留语义信息。但是，回译的方法产生的数据依赖于翻译的质量，大多数出现的翻译结果可能并不那么准确。

## 半监督
猜测数据扩增方法产生的无标签样本的低熵标签，并把无标签数据和有标签数据混合起来。

## 无监督数据增强UDA
传统的数据增广方法有一定的效果，但主要针对小数据量，对于渴求大量训练数据的深度学习模型，传统的方法效果始终有限。而Unsupervised Data Augmentation（UDA）无监督数据扩增方法的提出，为大量数据缺失打开了一扇大门。

得益于对特定任务使用特定目标的数据增强算法。

UDA对增广后未标记的数据预测结果使用KL散度，对有标签的数据训练时加入了cross entropy loss 函数

UDA采用了Training Signal Annealing（TSA）方法在训练时逐步释放训练信号。

当收集了少量的标注的数据和大量未标记的数据时，可能会面临标记数据和未标记数据相差很大的情况。因为需要采用大量的未标记数据进行训练，所需的模型会偏大，而大模型又会轻松的在有限的有监督数据上过拟合，这时TSA就要逐步的释放有监督数据的训练信号了。

作者对每个training step 都设了一个阈值ηt，且小于等于1，当一个标签例子的正确类别P的概率高于阈值ηt时，模型从损失函数中删除这个例子，只训练这个minibatch下其他标记的例子。

## 总结
训练机器学习或深度学习模型时，良好的数据往往是影响模型的效果最重要的因素之一。而数据不足时数据增强是一个常用的方法。

文本数据增强从对原数据词的变动到句子的变动到段落的变动都有不同的方法，为了保证能够真实提高数据的质量，有以下几个点尤为重要：

（1）增加的数据要保证和原数据一致的语义信息。

新增后的数据和原数据拥有一样标签的同时，更需要保证有一样的语义信息。单独随机去掉某个词的方式很可能会改变整句的含义（比如去掉一个否定词）。

（2）增加的数据需要多样化。

从替换词、句式、句型等方面都需要有新的数据以增强模型的泛化能力，单独交换词的方式较为局限。

（3）增加的数据要避免在有标签数据上过拟合。

当大量的数据在少量的有标签数据上过拟合时，模型虽然可能会出现很高的f1值，但真实的预测效果会相差很多。保证多样化的数据还要保证数据的质量。

（4）增加的数据和原数据保持一定的平滑性会更有价值，提高训练效率。

生成的数据更接近于真实数据可以保证数据的安全性，大噪音产生的数据和原始数据的标签很可能不同。尤其在某些序列模型中，文本数据的通顺程度严重影响模型的预测。

（5）增加数据的方法需要带着目标去选择。

对数据缺失的需求明确才能更快的找到理想的数据，对某些关键词的同义词需求较多可以偏重替换词的方式，对句式缺失较多可以偏重回译或者句式语法结构树变换的方式。

对于小数据的情况，使用文本回译或EDA中的简单方法可以达到效果的提升；但想要使用大批量的数据训练神经网络模型，EDA或者回译的方式产生的文本可能并不能满足需求。

而UDA这种无监督数据增强技术，无论对于小数据量或大数据量数据，都可以找到带有目标性的方法获得增强后的平滑的数据，甚至有时效果高于有监督方法训练的模型。

综上，数据增强的方法可以作为我们训练nlp模型时一个快速解决数据不平衡或数据缺失的强有力的工具。

