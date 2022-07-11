# 熵
## 信息熵
$$
H(X)=E\left(-\log _{2}(X)\right)=\sum-p(x) \log _{2}(p(x))
$$
信息熵是信源编码中, 压缩率的下限。当我们使用少于信息熵的信息量做编码，那么一定有信息的损失。 
## 自信息
自信息表示概率空间中与单一事件或离散变量的值相关的信息量的量度。
$$
I(x)=-\log (p(x))
$$
平均的自信息就是信息熵。
## 联合熵
联合熵是一集变量之间不确定的衡量手段。
$$
H(X, Y)=\sum \sum-p(x, y) \log (p(x, y))
$$
## 条件熵
条件熵描述变量Y在变量X确定的情况下, 变量Y的熵还剩多少。
$$
\begin{aligned}
H(Y \mid X)&=\sum \sum-p(x, y) \log (p(y \mid x))\\
&=-\sum_{x, y} p(x, y) \log p(y \mid x)\\

&=-\sum_{y} \sum_{x} p(y, x) \log p(y \mid x)\\

&=-\sum_{y} \sum_{x} p(x) p(y \mid x) \log p(y \mid x)\\

&=-\sum_{x} p(x) \sum_{y} p(y \mid x) \log p(y \mid x)\\

&=\sum_{x} p(x)\left[-\sum_{y} p(y \mid x) \log p(y \mid x)\right]\\

&=\sum_{x} p(x) H(Y \mid X=x)
\end{aligned}
$$
联合熵和条件熵的关系是：
$$
H(X, Y)=H(X)+H(Y \mid X)=H(Y)+H(X \mid Y)=H(Y, X)
$$
## 信息增益
又称互信息，信息散度（information divergence)

两个随机变量的互信息，是变量间相互依赖性的量度，不同于相关系数，互信息不限于实值随机变量，其更加一般。

信息熵-条件熵
$$\operatorname{I}(X,Y)=H(X)-H(X \mid Y)$$

![Alt](https://img-blog.csdn.net/20170907220115224)

$$
\begin{aligned} 
X\cap Y&=X+Y-(X\cup Y) \\
I(X ; Y) &=H(X)-H(X \mid Y) \\ &=H(Y)-H(Y \mid X) \\ &=H(X)+H(Y)-H(X, Y) \\ &=H(X, Y)-H(X \mid Y)-H(Y \mid X) 
\end{aligned}
$$

$$
\begin{aligned} I(X ; Y) &=H(X)-H(X \mid Y) \\ &=H(X)+H(Y)-H(X, Y) \\ &=\sum_{x} p(x) \log \frac{1}{p(x)}+\sum_{y} p(y) \log \frac{1}{p(y)}-\sum_{x, y} p(x, y) \log \frac{1}{p(x, y)} \\ &=\sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)} \end{aligned}
$$

## 交叉熵
假设有这样一个样本集，p为它的真实分布，q为它的估计分布。如果按照真实分布p来度量识别一个样本所需要的最短编码长度的期望为信息熵，那么按照估计分布q来度量识别一个样本所需要的编码长度的期望为交叉熵。
$$
H(p, q)=\sum-p(x) \log (q(x))
$$

$$
H(p, q)=E_{p}[-\log q]=H(p)+D_{k l}(p \mid\mid q)
$$
在机器学习中，真实标签的概率p和预测标签的概率q的交叉熵是一个很好的度量方法。真实标签为一个one-hot编码的向量，$q(x_i)$为softmax后的概率，让预测的概率值越来越接近于1。因此交叉上损失函数为：
$$
H(p, q)=\sum_{x_i=label_i}-\log (q(x_i))
$$
## KL散度

KL散度，又称为相对熵（relative entropy）、

由交叉熵可知，用估计的概率分布所需的编码长度，比真实分布的编码长，但是长多少呢？这个就需要另一个度量，相对熵，也称KL散度。
$$
D_{k l}(p \mid\mid q)=H(p, q)-H(p)
$$

$$
D_{K L}(P \| Q)=\sum_{i} P(i) \log \frac{P(i)}{Q(i)}
$$
# 语言模型指标

## 交叉熵 困惑度
交叉熵是按照语言模型$p$来度量识别一个句子T所需要的编码长度的期望
$$
H_p(T)=-\frac{1}{W_T} \log (p(T))
$$
文本T的概率为$p(T)$，$W_T$是以词为单位度量的文本T的长度(可以包括句首标志
〈BOS〉或句尾标志〈EOS〉)，并且默认句子每个词出现概率相等。

模型p的困惑度$PP_T(T)$是模型分配给测试集T中每一个词汇的概率倒数的几何平均值
$$
\begin{aligned}
PP_T(T)&=2^{H_P(T)}
\end{aligned}
$$

$$
=\sqrt[N]{\prod_{i=1}^{N} \frac{1}{P\left(w_{i} \mid w_{1} \ldots w_{i-1}\right)}} ~~\text{链式法则计算句子概率} \\
=\sqrt[N]{\prod_{i=1}^{N} \frac{1}{P\left(w_{i} \mid w_{i-1}\right)}} \quad\quad \text{Bi-gram计算句子概率}
$$
困惑度可以理解为，如果每个时间步都根据语言模型计算的概率分布随机挑词，那么平均情况下，挑多少个词才能挑到正确的那个。

## 数据平滑

### 加法平滑方法
$$p_{\mathrm{add}}\left(w_{i} \mid w_{i-n+1}^{i-1}\right)=\frac{\delta+c\left(w_{i-n+1}^{i}\right)}{\delta|V|+\sum_{w_{i}} c\left(w_{i-n+1}^{i}\right)}$$

### good-turing方法

首先引入一个新的符号 $N_{c}$, 表示频率 $c$ 的频率, 即在 $N$ 个样本中出现了 $c$ 次的类别的数量。然后 对于任何一个出现了 $c$ 次的类别, 都假设它出现了 $c^{*}$ 次
$$
c^{*}=(c+1) \frac{N_{c+1}}{N_{c}}
$$
假设 $m$-gram $w_{i}^{m}$ 出现了 $c\left(w_{i}^{m}\right)$ 次, Good-Turing给出其出现的概率为
$$
P_{G T}\left(w_{i}^{m}\right)=P({c^*\left(w_{i}^{m}\right)})=\frac{c^{*}\left(w_{i}^{m}\right)}{N}
$$
那么对于 $c=0$ （末观察到的）的样本
$$
\begin{aligned}
P_{0}
=& 1-\sum_{c>0} N_{c} * p_{c} \\
=& 1-\frac{1}{N} \sum_{c>0} N_{c} c^{*} \\
=& \frac{N-\sum_{c>0} N_{c} c^{*}}{N} \\
=& \frac{N-\sum_{c>0} N_{c+1}(c+1)}{N} \\
=& \frac{N-\left(N-N_{1}\right)}{N} \\
=& \frac{N_{1}}{N}
\end{aligned}
$$
因此有 $\frac{N_{1}}{N}$ 的概率余量分配给末见的类别，因此$p_0=\frac{P_0}{N_0}=\frac{N_1}{N*N_0}$。

我们可以将用$c^{*}$代替$c$的过程称为折扣，比值$\frac{c^{*}}{c}$称为折扣因子$d_c$。

### Katz平滑方法
Katz平滑方法通过加入高阶模型与低阶模型的结合，扩展了Good-Turing估计方法。

基本思想:当某一事件在样本中出现的频率大于阈 值K (通常取 K 为0 或1)时，运用最大似然估计的减值 法来估计其概率，否则，使用低阶的，即 (n-1)gram 的 概率替代n-gram概率，而这种替代需受归一化因子$\alpha$的作用。

对于每个计数 r > 0 的n元文法的出现次数减值, 把因减值而节省下来的剩余概率根据低阶的 (n-1)gram 分 配给未见事件。

### 绝对减值法

基本思想:从每个计数 r 中减去同样的量，剩余的 概率量由未见事件均分。
设 R 为所有可能事件的数目(当事件为 n-gram 时， 如果统计基元为词，且词汇集的大小为 L, 则 $R=L^n$)。
$$
p_{r}=\left\{\begin{array}{cl}
\frac{r-b}{N} & \text { 当 } r>0 \\
\frac{b\left(R-n_{0}\right)}{N n_{0}} & \text { 当 } r=0
\end{array}\right.
$$
其中, $n_{0}$ 为样本中末出现的事件的数目。 $b$ 为减去 的常量, $b \leq 1$ 。 $b\left(R-n_{0}\right) / N$ 是由于减值而产生的概率量。 $N$ 为样本中出现了 $r$ 次的事件总次数: $n_{r} \times r$ 。

### 线性减值法
基本思想: 从每个计数 $r$ 中减去与该计数成正比 的量 (减值函数为线性的), 剩余概率量 $\alpha$ 被 $n_{0}$ 个末 见事件均分。
$$
p_{r}=\left\{\begin{array}{cc}
\frac{(1-\alpha) r}{N} & \text { 当 } r>0 \\
\frac{\alpha}{n_{0}} & \text { 当 } r=0
\end{array}\right.
$$
自由参数 $\alpha$ 的优化值为: $\frac{n_{1}}{N}$

### 总结
大多数平滑算法可以用下 面的等式表示:
$$
p_{\text {smooth }}\left(w_{i} \mid w_{i-n+1}^{i-1}\right)= \begin{cases}\alpha\left(w_{i} \mid w_{i-n+1}^{i-1}\right), & c\left(w_{i-n+1}^{i}\right)>0 \\ \gamma\left(w_{i-n-1}^{i-1}\right) p_{\text {smooth }}\left(w_{i} \mid w_{i-n+2}^{i-1}\right), & c\left(w_{i-n+1}^{i}\right)=0\end{cases}
$$
也就是说, 如果n阶语言模型具有非零的计数, 就使用分布 $\alpha\left(\mathrm{w}_{\mathrm{i}} \mid\right. \left.w_{i-n}^{i-1}+1\right)$; 否则, 就后退到低阶分布 $\mathrm{p}_{\mathrm{smooth}}\left(\mathrm{w}_{\mathrm{i}} \mathrm{w}_{i-n+2}^{i-1}\right)$, 选择比例因子 $\gamma\left(\boldsymbol{w}_{i-n+1}^{i-1}\right)$ 使条件概率分布之和等于 1 。通常称符合 这种框架的平滑算法为后备模型 (back-off model)。前面介绍的Katz平滑算法是后备平滑算法的一个典型例子。

Good-Turing 法:对非0事件按公式削减出现的次
数，节留出来的概率均分给0概率事件。

Katz 后退法:对非0事件按Good-Turing法计算减
值，节留出来的概率按低阶分布分给0概率事件。

绝对减值法:对非0事件无条件削减某一固定的出现次数值，节留出来的概率均分给0概率事件。 

线性减值法:对非0事件根据出现次数按比例削减
次数值，节留出来的概率均分给0概率事件。


# 概率图模型：HMM、MEMM、CRF

![Alt](https://pic3.zhimg.com/80/v2-714c1843f78b6aecdb0c57cdd08e1c6a_1440w.jpg?source=1940ef5c)
动态贝叶斯网络(dynamic Bayesian networks, DBN)用于处理随时 间变化的动态系统中的推断和预测问题。其中，隐马尔可夫模型 (hidden Markov model, HMM)在语音识别、汉语自动分词与词性标注 和统计机器翻译等若干语音语言处理任务中得到了广泛应用;卡尔曼滤 波器则在信号处理领域有广泛的用途。马尔可夫网络(Markov network)又称马尔可夫随机场(Markov random field, MRF)。马尔可夫 网络下的条件随机场(conditional random field, CRF)广泛应用于自然语 言处理中的序列标注、特征选择、机器翻译等任务，波尔兹曼机 (Boltzmann machine)近年来被用于依存句法分析[Garg and Henderson, 2011]和语义角色标注[庄涛，2012]等。
![Alt](https://pica.zhimg.com/80/v2-376fd85a490e161978130ddd759244d4_1440w.jpg?source=1940ef5c)

横向:由点到线(序列结 构)、到面(图结构)。以朴素贝叶斯模型为基础的隐马尔可夫模型用 于处理线性序列问题，有向图模型用于解决一般图问题;以逻辑回归模 型(即自然语言处理中ME模型)为基础的线性链式条件随机场用于解 决“线式”序列问题，通用条件随机场用于解决一般图问题。纵向:在一定条件下生成式模型(generative model)转变为判别式模型 (discriminative model)，朴素贝叶斯模型演变为逻辑回归模型，隐马 尔可夫模型演变为线性链式条件随机场，生成式有向图模型演变为通用 条件随机场。

上图可以看到, 贝叶斯网络(信念网络) 都是有向的, 马尔科夫网络无向。所以, 贝叶斯网络适合为有单向依赖的数据建模, 马尔科夫网络适合实体之间互相依赖的建模。具体地, 他们的核心差 异表现在如何求  $P=(Y)$, 即怎么表示  $Y=\left(y_{1}, \cdots, y_{n}\right)$  这个的联合概率。

## 有向图&无向图模型
$\text { 对于有向图模型,这么求联合概率: } \quad P\left(x_{1}, \cdots, x_{n}\right)=\prod_{i=0} P\left(x_{i} \mid \pi\left(x_{i}\right)\right)$

对于无向图，一般就指马尔科夫网络}

(团与最大团）:无向图G中任何两个结点均有边连接的结点子集称为团。若C是无向图G的一个团，并且不能再加进任何一个G的结点使其成为一个更大的团，则称此C为最大团。


$P(Y)=\frac{1}{Z(x)} \prod_{c} \psi_{c}\left(Y_{c}\right)$ 其中$Z(x)=\sum_{Y} \prod_{c} \psi_{c}\left(Y_{c}\right)$

其中,  $\psi_{c}\left(Y_{c}\right)$  是一个最大团  C  上随机变量们的联合概率, 一般取指数函数的:

$\psi_{c}\left(Y_{c}\right)=e^{-E\left(Y_{c}\right)}=e^{\sum_{k} \lambda_{k} f_{k}(c, y \mid c, x)}$


最后可以得到


$P(Y)=\frac{1}{Z(x)} \prod_{c} \psi_{c}\left(Y_{c}\right)=\frac{1}{Z(x)} \prod_{c} e^{\sum_{k} \lambda_{k} f_{k}(c, y \mid c, x)}=\frac{1}{Z(x)} e^{\sum_{c} \sum_{k} \lambda_{k} f_{k}\left(y_{i}, y_{i-1}, x, i\right)}$


## 马尔科夫假设&马尔科夫性
1. 马尔科夫假设额应该是齐次马尔科夫假设，这样假设：马尔科夫链里的总是只受一个人的影响。马尔科夫假设这里相当于就是个2-gram。马尔科夫过程呢？即在一个过程中，每个状态的转移只依赖于前n个状态，并且只是个n阶的模型。最简单的马尔科夫过程是一阶的，即只依赖于器哪一个状态。

2. 马尔科夫性马尔科夫性是是保证或者判断概率图是否为概率无向图的条件。三点内容：a. 成对，b. 局部，c. 全局。



## HMM
![Alt](https://pic3.zhimg.com/80/v2-d4077c2dbd9899d8896751a28490c9c7_1440w.jpg?source=1940ef5c)

**HMM的5要素**

N , 隐藏状态集  $N=\left\{q_{1}, \cdots, q_{N}\right\}$，我的隐藏节点不能随意取, 只能限定取包含在隐藏状态 集中的符号。

M , 观测集$M=\left\{v_{1}, \cdots, v_{M}\right\}$，同样我的观测节点不能随意取, 只能限定取包含在观测状 态集中的符号。

A , 状态转移概率矩阵, 这个就是其中一个概率分布。他是个矩阵,  $A=\left[a_{i j}\right]_{N \times N}$(N为隐 藏状态集元素个数) , 其中$a_{i j}=P\left(i_{t+1} \mid i_{t}\right), i_{t}$即第$\mathrm{i}$个隐状态节点,即所谓的状态转移嘛。

B , 观测概率矩阵, 这个就是另一个概率分布。他是个矩阵,$B=\left[b_{i j}\right]_{N \times M}$(N为隐藏状态 集元素个数,  $\mathrm{M}$  为观测集元素个数), 其中  $b_{i j}=P\left(o_{t} \mid i_{t}\right), o_{t}$  即第  $\mathrm{i}$  个观测节点,  $i_{t}$  即第  $\mathrm{i}$  个 隐状态节点,即所谓的观测概率(发射概率)嘛。

$\pi$ , 在第一个隐状态节点  $i_{t}$ , 我得人工单独赋予, 我第一个隐状态节点的隐状态是  N  中的每一 个的概率分别是多少, 然后  $\pi$  就是其概率分布。

1. 根据概率图分类, 可以看到HMM属于有向图, 并且是生成式模型, 直接对联合概率分布建模  $P(O, I)=\sum_{t=1}^{T} P\left(O_{t} \mid O_{t-1}\right) P\left(I_{t} \mid O_{t}\right)$  (注意, 这个公式不在模型运行的任何阶段能体现出 来, 只是我们都去这么来表示HMM是个生成式模型, 他的联合概率  P(O, I)  就是这么计算 的)。
2. 并且B中  $b_{i j}=P\left(o_{t} \mid i_{t}\right)$ , 这意味着o对i有依赖性。
3. 在A中$a_{i j}=P\left(i_{t+1} \mid i_{t}\right)$ , 也就是说只遵循了一阶马尔科夫假设, 1-gram。试想, 如果数据 的依赖超过1-gram, 那肯定HMM肯定是考虑不进去的。这一点限制了HMM的性能。





### 解决了三个问题
1. **求概率问题，已知模型参数（A，B, π）和观测序列O，可以计算当前模型下的观测序列概率。即在已知模型的基础下，出现这样的观测序列可能性有多大。前向后向算法。**

2. **训练学习问题，学习HMM模型，已知观测序列O，计算使该观测序列出现最大可能性的HMM模型。这是一个通过序列得到模型的过程，不需要知道隐状态，得到模型自然也得到了隐状态。Baum-Welch算法**

3. **预测问题，根据HMM模型和观测序列，计算状态序列。Viterbi算法，预测最大可能序列。**

### Baum-Welch 算法（也就是 EM算法）

输入: 观测数据  $O=\left(o_{1}, o_{2}, \cdots, o_{T}\right)$ ;
输出：隐马尔可夫模型参数。
（1）初始化。对  n=0 , 选取  $a_{i j}^{(0)}, b_{j}(k)^{(0)}, \pi_{i}^{(0)}$ , 得到模型  $\lambda^{(0)}=\left(A^{(0)}, B^{(0)}, \pi^{(0)}\right)$  。

（2）递推。对  $n=1,2, \cdots ,$

$$\begin{aligned}
a_{i j}^{(n+1)} &=\frac{\sum_{t=1}^{T-1} \xi_{t}(i, j)}{\sum_{t=1}^{T-1} \gamma_{t}(i)} \\
b_{j}(k)^{(n+1)} &=\frac{\sum_{t=1, o_{t}=v_{k}}^{T} \gamma_{t}(j)}{\sum_{t=1}^{T} \gamma_{t}(j)} \\
\pi_{i}^{(n+1)} &=\gamma_{1}(i)
\end{aligned}$$

右端各值按观测  $O=\left(o_{1}, o_{2}, \cdots, o_{T}\right)  和模型  \lambda^{(n)}=\left(A^{(n)}, B^{(n)}, \pi^{(n)}\right)$  计算。式中  $\gamma_{t}(i), \xi_{t}(i, j)$  由式 
$$\gamma_{t}(i)=\frac{\alpha_{t}(i) \beta_{t}(i)}{P(O \mid \lambda)}=\frac{\alpha_{t}(i) \beta_{t}(i)}{\sum_{j=1}^{N} \alpha_{t}(j) \beta_{t}(j)}$$ 
和式 
$$\begin{array}{c}
\xi_{t}(i, j)=\frac{P\left(i_{t}=q_{i}, i_{t+1}=q_{j}, O \mid \lambda\right)}{P(O \mid \lambda)}=\frac{P\left(i_{t}=q_{i}, i_{t+1}=q_{j}, O \mid \lambda\right)}{\sum_{i=1}^{N} \sum_{j=1}^{N} P\left(i_{t}=q_{i}, i_{t+1}=q_{j}, O \mid \lambda\right)}
\end{array}$$ 
给出，其中
$P\left(i_{t}=q_{i}, i_{t+1}=q_{j}, O \mid \lambda\right)=\alpha_{t}(i) a_{i j} b_{j}\left(o_{t+1}\right) \beta_{t+1}(j)$。

（3）终止。得到模型参数  $\lambda^{(n+1)}=\left(A^{(n+1)}, B^{(n+1)}, \pi^{(n+1)}\right)$  。

### 最大熵模型
最大熵模型的基本原理是:在只掌握关于未知分布的部分信息的情况下，符合已知知识的概率分布可能有多个，但使熵值最大的概率分布最真实地反映了事件的分布情况，因为熵定义了随机变量的不确定性，当熵最大时，随机变量最不确定，最难准确地预测其行为。也就是说，在已知部分信息的前提下，关于未知分布最合理的推断应该是符合已知信息最不确定或最大随机的推断。

对一个随机事件的概率分布进行预测时，预测应当满足全部已知的约束，而对未知的情况不要做任何主观假设。在这种情况下，概率分布最均匀，预测的风险最小，因此得到的概率分布的熵是最大

$$\hat{p}(x)=\frac{count(x)}{N} $$
## MEMM 最大熵马尔科夫模型/条件马尔可夫模型
![Alt](https://pica.zhimg.com/80/v2-cb2cc25593fcaf06e682191d551ba03b_1440w.jpg?source=1940ef5c)

MEMM是判别模型,是能够直接允许“定义特征”，直接学习条件概率，为此, 提出来的MEMM模型就是能够直接允许"定义特征", 直接学习条件概率, 即  $P\left(i_{i} \mid i_{i-1}, o_{i}\right)(i=1, \cdots, n)$  ，总体为:

$P(I \mid O)=\prod_{t=1}^{n} P\left(i_{i} \mid i_{i-1}, o_{i}\right), i=1, \cdots, n$

并且,  $P\left(i \mid i^{\prime}, o\right)$  这个概率通过最大樀分类器建模（取名MEMM的原因）:

$P\left(i \mid i^{\prime}, o\right)=\frac{1}{Z\left(o, i^{\prime}\right)} \exp \left(\sum_{a}\right) \lambda_{a} f_{a}(o, i)$

$Z\left(o, i^{\prime}\right)$  这部分是归一化;  $f_{a}(o, i)$  是特征函数,具体点,这个函数是需要去定义的;  $\lambda$  是特征函数的权重, 这是个末知参数, 需要从训练阶段学习而得。

MEMM的建模公式
$P(I \mid O)=\prod_{t=1}^{n} \frac{\exp \left(\sum_{a}\right) \lambda_{a} f_{a}(o, i)}{Z\left(o, i_{i-1}\right)}, i=1, \cdots, n$

与HMM的  $o_{i}$  依赖  $i_{i}$  不一样, MEMM当前隐藏状态  $i_{i}$  应该是依赖当前时刻的观测节点  $o_{i}$  和上一时刻的隐藏节点  $i_{i-1}$

## CRF 条件随机场
![Alt](https://pica.zhimg.com/80/v2-c5e2e782e35f6412ed65e58cdda0964e_1440w.jpg?source=1940ef5c)

条件随机场是在给定的随机变量  X  (具体, 对应观测序列  $o_{1}, \cdots, o_{i}$ ) 条件下, 随机变量  Y  (具体, 对应隐 状态序列  $i_{1}, \cdots, i_{i}$)  的马尔科夫随机场。
广义的CRF的定义是: 满足  $P\left(Y_{v} \mid X, Y_{w}, w \neq v\right)=P\left(Y_{v} \mid X, Y_{w}, w \sim v\right)$  的马尔科夫随机场 叫做条件随机场 (CRF)。

**CRF基于条件分布建模是指在给定的随机变量  X  (具体, 对应观测序列  $o_{1}, \cdots, o_{i}$ ) 条件下。**

CRF的建模公式如下：
$$
\begin{aligned}
P(I \mid O)&=\frac{1}{Z(O)} \prod_{i} \psi_{i}\left(I_{i} \mid O\right)\\
&=\frac{1}{Z(O)} \prod_{i} e^{\sum_{k} \lambda_{k} f_{k}\left(O, I_{i-1}, I_{i}, i\right)}\\
&=\frac{1}{Z(O)} e^{\sum_{i} \sum_{k} \lambda_{k} f_{k}\left(O, I_{i-1}, I_{i}, i\right)}
\end{aligned}
$$
## 总结
1. HMM -> MEMM： 
   HMM模型中存在两个假设：一是输出观察值之间严格独立，二是状态的转移过程中当前状态只与前一状态有关。但实际上序列标注问题不仅和单个词相关，而且和观察序列的长度，单词的上下文，等等相关。**MEMM解决了HMM输出独立性假设的问题**。因为HMM只限定在了观测与状态之间的依赖，而MEMM引入自定义特征函数，不仅可以表达观测之间的依赖，**还可表示当前观测与前后多个状态之间的复杂依赖**。
2. MEMM -> CRF:
   CRF不仅解决了HMM输出独立性假设的问题，还解决了MEMM的**标注偏置问题**，MEMM容易陷入局部最优是因为只在局部做归一化，而CRF统计了全局概率，在做归一化时考虑了数据在全局的分布，而不是仅仅在局部归一化，这样就解决了MEMM中的标记偏置的问题。使得序列标注的解码变得最优解。HMM、MEMM属于有向图，所以考虑了x与y的影响，但没讲x当做整体考虑进去（这点问题应该只有HMM）。CRF属于无向图，没有这种依赖性，克服此问题。



# TF-IDF
TF-IDF(Term Frequency-Inverse Document Frequency, 词频-逆文件频率)是一种用于资讯检索与资讯探勘的常用加权技术。TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。

简单来说就是：一个词语在一篇文章中出现次数越多, 同时在所有文档中出现次数越少, 越能够代表该文章。这也就是TF-IDF的含义。
## TF(Term Frequency)
词频（TF）表示词条（关键字）在文本中出现的频率。这个数字通常会被归一化(一般是词频除以文章总词数), 以防止它偏向长的文件。

$$TF_w = \frac{ 在某一类中词条 w 出现的次数 }{该类中所有的词条数目}$$

## IDF(Inverse Document Frequency)
逆向文件频率 (IDF) ：某一特定词语的IDF，可以由总文件数目除以包含该词语的文件的数目，再将得到的商取对数得到。

$$IDF_w = \log(\frac{语料库的文档总数}{包含词条w的文档数+1}) $$

如果包含词条t的文档越少, IDF越大，则说明词条具有很好的类别区分能力。

TF-IDF=TF*IDF

某一特定文件内的高词语频率，以及该词语在整个文件集合中的低文件频率，可以产生出高权重的TF-IDF。因此，TF-IDF倾向于过滤掉常见的词语，保留重要的词语。

IDF 的简单结构并不能有效地反映单词的重要程度和特征词的分布情况，使其无法很好地完成对权值调整的功能，所以 TF-IDF 算法的精度并不是很高，尤其是当文本集已经分类的情况下。

（1）没有考虑特征词的位置因素对文本的区分度，词条出现在文档的不同位置时，对区分度的贡献大小是不一样的。

（2）按照传统TF-IDF，往往一些生僻词的IDF(反文档频率)会比较高、因此这些生僻词常会被误认为是文档关键词。

（3）传统TF-IDF中的IDF部分只考虑了特征词与它出现的文本数之间的关系，而忽略了特征项在一个类别中不同的类别间的分布情况。

（4）对于文档中出现次数较少的重要人名、地名信息提取效果不佳。

## 代码

```py
import math

class TfIdf:
    def __init__(self):
        self.num_docs = 0
        self.vocab = {}

    def add_corpus(self, corpus):
        self._merge_corpus(corpus)

        tfidf_list = []
        for sentence in corpus:
            tfidf_list.append(self.get_tfidf(sentence))
        return tfidf_list

    def _merge_corpus(self, corpus):
        """
        统计语料库，输出词表，并统计包含每个词的文档数。
        """
        self.num_docs = len(corpus)
        for sentence in corpus:
            words = sentence.strip().split()
            words = set(words)
            for word in words:
                self.vocab[word] = self.vocab.get(word, 0.0) + 1.0

    def _get_idf(self, term):
        """
        计算 IDF 值
        """
        return math.log(self.num_docs / (self.vocab.get(term, 0.0) + 1.0))

    def get_tfidf(self, sentence):
        tfidf = {}
        terms = sentence.strip().split()
        terms_set = set(terms)
        num_terms = len(terms)
        for term in terms_set:
            # 计算 TF 值
            tf = float(terms.count(term)) / num_terms
            # 计算 IDF 值，在实际实现时，可以提前将所有词的 IDF 提前计算好，然后直接使用。
            idf = self._get_idf(term)
            # 计算 TF-IDF 值
            tfidf[term] = tf * idf
        return tfidf

corpus = [
    "What is the weather like today",
    "what is for dinner tonight",
    "this is question worth pondering",
    "it is a beautiful day today"
]

tfidf = TfIdf()
tfidf_values = tfidf.add_corpus(corpus)
for tfidf_value in tfidf_values:
    print(tfidf_value)
```
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

TPR越高，同时FPR越低（即ROC曲线越陡），那么模型的性能就越好。

用 ROC 曲线来表示分类器的性能很直观很好用。但是人们更希望能有一个数值来表示分类器的好坏。于是 Area Under ROC Curve(AUC) 就出现了。

AUC被定义为ROC曲线下的面积，完全随机的二分类器的AUC为0.5，因此与之相对的相对面积更大，更靠近左上角的曲线代表着一个更加稳健的二分类器。

首先 AUC 值是一个概率值，当你随机挑选一个正样本以及负样本，当前的分类算法根据计算得到的 Score 值能够区分正样本和负样本的概率就是 AUC 值，AUC 值越大，当前分类算法越有可能区分正样本负样本，从而能够更好地分类。

AUC越接近 1 越好是肯定的，但是并不是越接近 0 就越差，最差的是接近 0.5，如果 AUC 很接近 0 的话，只需要把模型预测的结果加个负号就能让 AUC 接近 1。

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


# BLUE BiLingual Evaluation Understudy（双语评价替补）
BLEU是IBM在2002提出的，用于机器翻译任务的评价。

BLEU还有许多变种。根据n-gram可以划分成多种评价指标，常见的指标有BLEU-1、BLEU-2、BLEU-3、BLEU-4四种，其中n-gram指的是连续的单词个数为n。BLEU-1衡量的是单词级别的准确性，更高阶的bleu可以衡量句子的流畅性。它的总体思想就是准确率(精准率、查准率)。
$$
\begin{aligned}
&p_{n}=\frac{\sum_{C \in\{\text { Candidates }\}} \sum_{n-\text { grame }\in C} \text { Count }_{\text {clip }}(n-\text { gram })}{\sum_{C' \in\{\text { Candidates }\}} \sum_{n-\text { gram' } \in C'} \operatorname{Count}(n-\text { gram })}\\
&\operatorname{Count}_{\text {clip }}(n-\operatorname{gram})=\min (\operatorname{Count}(n-\operatorname{gram}), \operatorname{Ref}(n-\operatorname{gram}))\\
&\operatorname{Ref}(n-g r a m)=\max \left(\operatorname{Ref}^{j}(n-g r a m)\right), j=1,2, \ldots, M
\end{aligned}
$$
Count是N-gram在机器翻译译文中的出现次数，Ref是参考译文中的N-gram出现次数。
reference是参考译文，candidates是预测句。

n的总数为N，一般N取4，用$\omega_n$表示权重，一般为1/4，那么综合的$p_{avg}$为
$$
p_{\text {avg }}=e^{\sum_{n=1}^{N} \omega_{n} \log \left(p_{n}\right)}
$$
BP（Brevity Penalty）指标，由于在modified N-gram precision已经对Candidate过长进行惩罚（过长则分母较大），此处仅需要对Candidate过短进行惩罚。令c表示Candidate的长度，r表示reference中长度最接近Candidate的句子的长度，当c小于r时进行惩罚
$$
B P=\left\{\begin{array}{rl}
1 & i f c>r \\
e^{1-r / c} & \text { ifc } \leq r
\end{array}\right.
$$
将BP和modified N-gram precision进行整合，可以得到BLEU值
$$
B L E U=B P \cdot p_{\text {avg }}=B P \cdot e^{\sum_{n=1}^{N} \omega_{n} \log \left(p_{n}\right)}
$$
# Subword

## tokenization技术的对比
1. Word-based tokenization

    传统词表示方法无法很好的处理未知或罕见的词汇（OOV out of vocabulary 问题）

    传统词tokenization方法不利于模型学习词缀之间的关系

    E.g. 模型学到的“old”, “older”, and “oldest”之间的关系无法泛化到“smart”, “smarter”, and “smartest”。

2. Character-based tokenization

    Character embedding作为OOV的解决方法粒度太细，句子非常长

    单个字符没有意义

3. Subword-based tokenization
    Subword粒度在词与字符之间，能够较好的平衡OOV问题

    高频词不应该被切分成更小的subwords，如dog 不应该切分成 do 和 ##g (##为非单词开头的标志)

    生僻词应该被切分成有意义的subwords，如tokenization 切分成 token 和 ##ization

    Subword tokenizaiton 算法可以标记单词的起始，如tokenization 切分成 token 和 ##ization，token表示单词的起始，有点英语中的词根、词缀的意思。

    词表既不是特别大，又能有效减少OOV问题，低频词也能很好的训练



## Byte Pair Encoding BPE
BPE(字节对)编码或二元编码是一种简单的数据压缩形式，其中最常见的一对连续字节数据被替换为该数据中不存在的字节。 后期使用时需要一个替换表来重建原始数据。OpenAI GPT-2 与Facebook RoBERTa均采用此方法构建subword vector。

优点    可以有效地平衡词汇表大小和步数(编码句子所需的token数量)。

缺点    基于贪婪和确定的符号替换，不能提供带概率的多个分片结果。

### 算法

1. 准备足够大的训练语料
2. 确定期望的subword词表大小
3. 将单词拆分为字符序列并在末尾添加后缀“ </ w>”，统计单词频率。 本阶段的subword的粒度是字符。 例如，“ low”的频率为5，那么我们将其改写为“ l o w </ w>”：5
4. 统计每一个连续字节对的出现频率，选择最高频者合并成新的subword
5. 重复第4步直到达到第2步设定的subword词表大小或下一个最高频的字节对出现频率为1

每次合并后词表可能出现3种变化：

+1，表明加入合并后的新字词，同时原来的2个子词还保留（2个字词不是完全同时连续出现）

+0，表明加入合并后的新字词，同时原来的2个子词中一个保留，一个被消解（一个字词完全随着另一个字词的出现而紧跟着出现）

-1，表明加入合并后的新字词，同时原来的2个子词都被消解（2个字词同时连续出现）

实际上，随着合并的次数增加，词表大小通常先增加后减小。

### 编码和解码

在之前的算法中，我们已经得到了subword的词表，对该词表按照子词长度由大到小排序。编码时，对于每个单词，遍历排好序的子词词表寻找是否有token是当前单词的子字符串，如果有，则该token是表示单词的tokens之一。

我们从最长的token迭代到最短的token，尝试将每个单词中的子字符串替换为token。 最终，我们将迭代所有tokens，并将所有子字符串替换为tokens。 如果仍然有子字符串没被替换但所有token都已迭代完毕，则将剩余的子词替换为特殊token，如< unk >。

解码就是将所有的tokens拼在一起。

## WordPiece

WordPiece算法可以看作是BPE的变种。不同点在于，WordPiece基于概率生成新的subword而不是下一最高频字节对。Bert使用的是WordPiece。

1. 准备足够大的训练语料
2. 确定期望的subword词表大小
3. 将单词拆分成字符序列
4. 基于第3步数据训练语言模型
5. 从所有可能的subword单元中选择加入语言模型后能最大程度地增加训练数据概率的单元作为新的单元。计算token对的得分，合并得分最高的token对。得分为token对的频率除以两个token分别的频率（互信息）如h 出现了100次，e出现了20次，he出现了5次，则得分为5/（100*20）=1/400
6. 重复第5步直到达到第2步设定的subword词表大小或概率增量低于某一阈值


假设把相邻位置的x和y两个子词进行合并，合并后产生的子词记为z，此时句子$S$似然值的变化可表示为：
$$
log P(t_z)-(logP(t_x)+logP(t_y))=\frac{logP(t_z)}{logP(t_x)P(t_y)} 
$$
似然值的变化就是两个子词之间的互信息。简而言之，WordPiece每次选择合并的两个子词，他们具有最大的互信息值，也就是两子词在语言模型上具有较强的关联性，它们经常在语料中以相邻方式同时出现。

## Unigram Language Model ULM
它能够输出带概率的多个子词分段。它引入了一个假设：所有subword的出现都是独立的，并且subword序列由subword出现概率的乘积产生。WordPiece和ULM都利用语言模型建立subword词表。ULM算法考虑了句子的不同分词可能，因而能够输出带概率的多个子词分段。

1. 准备足够大的训练语料
2. 确定期望的subword词表大小
3. 建立一个足够大的词表。一般，可用语料中的所有字符加上常见的子字符串初始化词表，也可以通过BPE算法初始化。
4. 针对当前词表，用EM算法求解每个子词在语料上的概率。
5. 对于每个子词，计算当该子词被从词表中移除时，总的loss降低了多少，记为该子词的loss。
6. 将子词按照loss大小进行排序，丢弃一定比例loss最小的子词(比如20%)，保留下来的子词生成新的词表。为了避免OOV，建议保留字符级的单元
7. 重复第4至第6步直到达到第2步设定的subword词表大小或第5步的结果不再变化


与WordPiece一样，Unigram Language Model(ULM)同样使用语言模型来挑选子词。

不同之处在于，BPE和WordPiece算法的词表大小都是从小到大变化，属于增量法。

而Unigram Language Model则是减量法,即先初始化一个大词表，根据评估准则不断丢弃词表，直到满足限定条件。ULM算法考虑了句子的不同分词可能，因而能够输出带概率的多个子词分段。

对于句子S，挑选似然值最大的作为分词结果

初始时，词表V并不存在。因而，ULM算法采用不断迭代的方法来构造词表以及求解分词概率

在实际应用中，词表大小有上万个，直接罗列所有可能的分词组合不具有操作性。针对这个问题，可通过维特比算法得到来解决

ULM通过EM算法来估计每个子词的概率。

可以看出，ULM会保留那些以较高频率出现在很多句子的分词结果中的子词，因为这些子词如果被丢弃，其损失会很大。


## 总结

|算法|	BPE|	WordPiece|	ULM|
|---|---|---|---|
|训练方式|	从一个小词汇表开始，学习合并token的规则|	从一个小词汇表开始，学习合并token的规则|	从大量词汇开始（可以通过BPE获得），学习删除token的规则|
|是否使用语言模型|	否|	是|	是|
|规则|	合并频次最高的token对如有两个token对，（h,e）出现10次，（t, e）出现6次，则合并(h,e)|	计算token对的得分，合并得分最高的token对。得分为token对的频率除以两个token分别的频率（互信息）如h 出现了100次，e出现了20次，he出现了5次，则得分为5/（100*20）=1/400，也可依据交叉熵，能够提升语言模型概率最大的相邻子词加入词表|用EM算法估计词的概率，在全部语料中计算删除该token时的损失，丢弃一定比例loss最小的子词|
|学习结果|	合并规则及词表|	词表|	词表，每个token有对应的分数|
|使用|	将单词切分成单个字符，使用学到的合并规则从前往后对token进行合并|	从单词的起始位置开始，找到最长的且位于词表中的subword，重复该步骤。	|使用训练过程中学到的分数，找到最可能的切分结果。一般偏向将单词切分成尽可能少的token|


1. subword可以平衡词汇量和对未知词的覆盖。 极端的情况下，我们只能使用26个token（即字符）来表示所有英语单词。一般情况，建议使用16k或32k子词足以取得良好的效果，Facebook RoBERTa甚至建立的多达50k的词表。
2. 对于包括中文在内的许多亚洲语言，单词不能用空格分隔。 因此，初始词汇量需要比英语大很多。

GPT-2和RoBERTa使用的Subword算法都是BPE，Bert使用的Subword算法是WordPiece。



# 生成式模型&判别式模型
生成式模型表示了给定输入X产生输出Y的生产关系， $P(X,Y)=P(Y)*P(X|Y)$ 

对联合分布概率p(x,y)进行建模，$P(Y)、P(X|Y)$均由样本直接计算得出，$P(X)$由样本的全部输入计算得出，可以得出一个输入X为Y的概率。可以直接由$P(Y)*P(X|Y)$ 判断分类，也可根据$P(Y \mid X)=\frac{P(X, Y)}{P(X)}$。

判别式模型是有数据直接学习决策函数或者条件概率分布的模型，直接对条件概率$P(Y|X)$或$P(X)$进行建模。

条件概率分布$P(Y|X)$可以理解为：在已知某样本的特征为X的条件下，计算该样本类别为类别Y1、Y2、Y3的概率，并选择概率最大的类别为该样本的预测类别。

生成式模型(或称产生式模型)与区分式模型(或称判别式模型) 的本质区别在于模型中观测序列x和状态序列y之间的决定关系，前者假 设y决定x，后者假设x决定y。生成模型以“状态(输出)序列y按照一定 的规律生成观测(输入)序列x”为假设，针对联合分布p(x, y)进行建 模，并且通过估计使生成概率最大的生成序列来获取y。生成式模型是 所有变量的全概率模型，因此可以模拟(“生成”)所有变量的值。在这类模型中一般都有严格的独立性假设，特征是事先给定的，并且特征之间的关系直接体现在公式中。这类模型的优点是:处理单类问题时比较灵活，模型变量之间的关系比较清楚，模型可以通过增量学习获得，可用于数据不完整的情况。

判别式模型则符合传统的模式分类思想，认为y由x决定，直接对后 验概率p(y|x)进行建模，它从x中提取特征，学习模型参数，使得条件 概率符合一定形式的最优。在这类模型中特征可以任意给定，一般特征是通过函数表示的。这种模型的优点是:处理多类问题或分辨某一类与 其他类之间的差异时比较灵活，模型简单，容易建立和学习。其弱点在于模型的描述能力有限，变量之间的关系不清楚，而且大多数区分式模型是有监督的学习方法，不能扩展成无监督的学习方法。



## 总体特征

判别式模型的特征总结如下：

1. 对P(y|x)建模，直接学习得到P(y|x)，利用MAP得到 y。或者直接学得一个映射函数 y=f(x)。
2. 对所有的样本只构建一个模型，确认总体判别边界
3. 观测到输入什么特征，就预测最可能的label
4. 判别式的优点是：对数据量要求没生成式的严格，速度也会快，小数据量下准确率也会好些。

生成式模型的特征总结如下：
1. 对P(x,y)建模，继而得到 P(y|x)。预测时应用最大后验概率法（MAP）得到预测类别 y。
2. 这里我们主要讲分类问题，所以是要对每个label都需要建模，最终选择最优概率的label为结果，所以没有什么判别边界。（对于序列标注问题，那只需要构件一个model）
3. 中间生成联合分布，并可生成采样数据。
4. 生成式模型的优点在于，所包含的信息非常齐全，我称之为“上帝信息”，所以不仅可以用来输入label，还可以干其他的事情。生成式模型关注结果是如何产生的。但是生成式模型需要非常充足的数据量以保证采样到了数据本来的面目，所以速度相比之下，慢。
5. 对每一种样本属性都需要建模，这样就需要很多的模型，而且模型的描述能力有限，变量之间的关系不清楚，而且大多数区分式模型是有监督的学习方法，不能扩展成无监督的学习方法。

判别式模型从历史数据中学习到模型，然后通过提取需要判断的样本的特征来预测出属于A、B....等的概率。

生成模型是根据样本属性的特征首先学习出多个模型，然后从需要判断的样本中提取特征，放到A模型中看概率是多少，在放到B模型中看概率是多少，哪个大就是哪个。

判别模型： 关注类别之间的差别
生成模型： 关注数据是如何生成的
## 优缺点
生成式模型优缺点

（1）生成式模型的优点：过拟合的几率比较小，尤其是当你采集的数据的分布与真实世界整体数据集的分布是相近的适合，基本上不用担心过拟合问题。

（2）生成式模型的缺点：因为生成式模型需要生成Y的分布函数，而这个分布函数可能会受到一些异常点的影响变得不那么准确。

（3）为了使生成的分布函数与真实世界中的分布函数尽可能接近，需要大量的数据来生成模型。

（4）生成式模型比判别式模型计算量更大。

判别式模型的优缺点

（1）优点：在小数据集上表现效果很好，但是要注意过拟合问题。另外，计算量比生成式模型小。

## 生成式模型
判别式分析
朴素贝叶斯
混合高斯模型
隐马尔科夫模型（HMM）
贝叶斯网络
Sigmoid Belief Networks
马尔科夫随机场（Markov Random Fields）
贝叶斯网络/信念网络（DBN）

***seq2seq模型***

## 判别式模型
线性回归（Linear Regression）
K近邻（KNN）
逻辑斯蒂回归（Logistic Regression）
神经网络（NN）
支持向量机（SVM）
高斯过程（Gaussian Process）
条件随机场（CRF）
CART（Classification and Regression Tree）



