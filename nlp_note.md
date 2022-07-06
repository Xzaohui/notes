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
$$\begin{aligned}
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



## 互信息
两个随机变量的互信息，是变量间相互依赖性的量度，不同于相关系数，互信息不限于实值随机变量，其更加一般。
$\begin{aligned} I(X ; Y) &=\sum \sum-p(x, y) \log \left(\frac{p(x) p(y)}{p(x, y)}\right) \\ I(X ; Y) &=H(X)-H(X \mid Y) \\ &=H(Y)-H(Y \mid X) \\ &=H(X)+H(Y)-H(X, Y) \\ &=H(X, Y)-H(X \mid Y)-H(Y \mid X) \end{aligned}$



## 信息增益
又称互信息，信息散度（information divergence)

信息熵-条件熵
$$\operatorname{I}(X,Y)=H(X)-H(X \mid Y)$$

![Alt](https://img-blog.csdn.net/20170907220115224)

$$\begin{aligned} 
X\cap Y&=X+Y-(X\cup Y) \\
I(X ; Y) &=H(X)-H(X \mid Y) \\ &=H(Y)-H(Y \mid X) \\ &=H(X)+H(Y)-H(X, Y) \\ &=H(X, Y)-H(X \mid Y)-H(Y \mid X) 
\end{aligned}$$

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
$$\begin{aligned}
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

$\text {对于无向图，一般就指马尔科夫网络}$ 

(团与最大团）:无向图G中任何两个结点均有边连接的结点子集称为团。若C是无向图G的一个团，并且不能再加进任何一个G的结点使其成为一个更大的团，则称此C为最大团。


$P(Y)=\frac{1}{Z(x)} \prod_{c} \psi_{c}\left(Y_{c}\right)$ 其中$Z(x)=\sum_{Y} \prod_{c} \psi_{c}\left(Y_{c}\right)$

其中,  $\psi_{c}\left(Y_{c}\right)$  是一个最大团  C  上随机变量们的联合概率, 一般取指数函数的:

$\psi_{c}\left(Y_{c}\right)=e^{-E\left(Y_{c}\right)}=e^{\sum_{k} \lambda_{k} f_{k}(c, y \mid c, x)}$


最后可以得到


$P(Y)=\frac{1}{Z(x)} \prod_{c} \psi_{c}\left(Y_{c}\right)=\frac{1}{Z(x)} \prod_{c} e^{\sum_{k} \lambda_{k} f_{k}(c, y \mid c, x)}=\frac{1}{Z(x)} e^{\sum_{c} \sum_{k} \lambda_{k} f_{k}\left(y_{i}, y_{i-1}, x, i\right)}$


## 马尔科夫假设&马尔科夫性
1. 马尔科夫假设额应该是齐次马尔科夫假设，这样假设：马尔科夫链里的总是只受一个人的影响。马尔科夫假设这里相当于就是个2-gram。马尔科夫过程呢？即在一个过程中，每个状态的转移只依赖于前n个状态，并且只是个n阶的模型。最简单的马尔科夫过程是一阶的，即只依赖于器哪一个状态。

2. 马尔科夫性马尔科夫性是是保证或者判断概率图是否为概率无向图的条件。三点内容：a. 成对，b. 局部，c. 全局。


## 生成式模型&判别式模型
生成式模型表示了给定输入X产生输出Y的生产关系， $P(X,Y)=P(Y)*P(X|Y)$ 

对联合分布概率p(x,y)进行建模，$P(Y)、P(X|Y)$均由样本直接计算得出，$P(X)$由样本的全部输入计算得出，可以得出一个输入X为Y的概率。可以直接由$P(Y)*P(X|Y)$ 判断分类，也可根据$P(Y \mid X)=\frac{P(X, Y)}{P(X)}$。

判别式模型是有数据直接学习决策函数或者条件概率分布的模型，直接对条件概率$P(Y|X)$或$P(X)$进行建模。

条件概率分布$P(Y|X)$可以理解为：在已知某样本的特征为X的条件下，计算该样本类别为类别Y1、Y2、Y3的概率，并选择概率最大的类别为该样本的预测类别。

生成式模型(或称产生式模型)与区分式模型(或称判别式模型) 的本质区别在于模型中观测序列x和状态序列y之间的决定关系，前者假 设y决定x，后者假设x决定y。生成模型以“状态(输出)序列y按照一定 的规律生成观测(输入)序列x”为假设，针对联合分布p(x, y)进行建 模，并且通过估计使生成概率最大的生成序列来获取y。生成式模型是 所有变量的全概率模型，因此可以模拟(“生成”)所有变量的值。在这类模型中一般都有严格的独立性假设，特征是事先给定的，并且特征之间的关系直接体现在公式中。这类模型的优点是:处理单类问题时比较灵活，模型变量之间的关系比较清楚，模型可以通过增量学习获得，可用于数据不完整的情况。

判别式模型则符合传统的模式分类思想，认为y由x决定，直接对后 验概率p(y|x)进行建模，它从x中提取特征，学习模型参数，使得条件 概率符合一定形式的最优。在这类模型中特征可以任意给定，一般特征是通过函数表示的。这种模型的优点是:处理多类问题或分辨某一类与 其他类之间的差异时比较灵活，模型简单，容易建立和学习。其弱点在于模型的描述能力有限，变量之间的关系不清楚，而且大多数区分式模型是有监督的学习方法，不能扩展成无监督的学习方法。



### 总体特征

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

判别模型： 关注类别之间的差别
生成模型： 关注数据是如何生成的
### 优缺点
生成式模型优缺点

（1）生成式模型的优点：过拟合的几率比较小，尤其是当你采集的数据的分布与真实世界整体数据集的分布是相近的适合，基本上不用担心过拟合问题。

（2）生成式模型的缺点：因为生成式模型需要生成Y的分布函数，而这个分布函数可能会受到一些异常点的影响变得不那么准确。

（3）为了使生成的分布函数与真实世界中的分布函数尽可能接近，需要大量的数据来生成模型。

（4）生成式模型比判别式模型计算量更大。

判别式模型的优缺点

（1）优点：在小数据集上表现效果很好，但是要注意过拟合问题。另外，计算量比生成式模型小。

### 生成式模型
判别式分析
朴素贝叶斯
混合高斯模型
隐马尔科夫模型（HMM）
贝叶斯网络
Sigmoid Belief Networks
马尔科夫随机场（Markov Random Fields）
贝叶斯网络/信念网络（DBN）

### 判别式模型
线性回归（Linear Regression）
K近邻（KNN）
逻辑斯蒂回归（Logistic Regression）
神经网络（NN）
支持向量机（SVM）
高斯过程（Gaussian Process）
条件随机场（CRF）
CART（Classification and Regression Tree）

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


前向后向算法 计算序列整体概率

Viterbi算法 预测最大可能序列

### Baum-Welch 算法（也就是 EM算法）

输入: 观测数据  $O=\left(o_{1}, o_{2}, \cdots, o_{T}\right)$ ;
输出：隐马尔可夫模型参数。
（1）初始化。对  n=0 , 选取  $a_{i j}^{(0)}, b_{j}(k)^{(0)}, \pi_{i}^{(0)}$ , 得到模型  $\lambda^{(0)}=\left(A^{(0)}, B^{(0)}, \pi^{(0)}\right)$  。

（2）递推。对  $n=1,2, \cdots ,$

$\begin{aligned}
a_{i j}^{(n+1)} &=\frac{\sum_{t=1}^{T-1} \xi_{t}(i, j)}{\sum_{t=1}^{T-1} \gamma_{t}(i)} \\
b_{j}(k)^{(n+1)} &=\frac{\sum_{t=1, o_{t}=v_{k}}^{T} \gamma_{t}(j)}{\sum_{t=1}^{T} \gamma_{t}(j)} \\
\pi_{i}^{(n+1)} &=\gamma_{1}(i)
\end{aligned}$

右端各值按观测  $O=\left(o_{1}, o_{2}, \cdots, o_{T}\right)  和模型  \lambda^{(n)}=\left(A^{(n)}, B^{(n)}, \pi^{(n)}\right)$  计算。式中  $\gamma_{t}(i), \xi_{t}(i, j)$  由式 $\gamma_{t}(i)=\frac{\alpha_{t}(i) \beta_{t}(i)}{P(O \mid \lambda)}=\frac{\alpha_{t}(i) \beta_{t}(i)}{\sum_{j=1}^{N} \alpha_{t}(j) \beta_{t}(j)}$  和式 $\begin{array}{c}
\xi_{t}(i, j)=\frac{P\left(i_{t}=q_{i}, i_{t+1}=q_{j}, O \mid \lambda\right)}{P(O \mid \lambda)}=\frac{P\left(i_{t}=q_{i}, i_{t+1}=q_{j}, O \mid \lambda\right)}{\sum_{i=1}^{N} \sum_{j=1}^{N} P\left(i_{t}=q_{i}, i_{t+1}=q_{j}, O \mid \lambda\right)}
\end{array}$ 给出，其中$P\left(i_{t}=q_{i}, i_{t+1}=q_{j}, O \mid \lambda\right)=\alpha_{t}(i) a_{i j} b_{j}\left(o_{t+1}\right) \beta_{t+1}(j)$。

（3）终止。得到模型参数  $\lambda^{(n+1)}=\left(A^{(n+1)}, B^{(n+1)}, \pi^{(n+1)}\right)$  。

### 最大熵模型
最大熵模型的基本原理是:在只掌握关于未知分布的部分信息的情
况下，符合已知知识的概率分布可能有多个，但使熵值最大的概率分布
最真实地反映了事件的分布情况，因为熵定义了随机变量的不确定性，
当熵最大时，随机变量最不确定，最难准确地预测其行为。也就是说，
在已知部分信息的前提下，关于未知分布最合理的推断应该是符合已知
信息最不确定或最大随机的推断。

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

CRF的建模公式如下：

$P(I \mid O)=\frac{1}{Z(O)} \prod_{i} \psi_{i}\left(I_{i} \mid O\right)=\frac{1}{Z(O)} \prod_{i} e^{\sum_{k} \lambda_{k} f_{k}\left(O, I_{i-1}, I_{i}, i\right)}=\frac{1}{Z(O)} e^{\sum_{i} \sum_{k} \lambda_{k} f_{k}\left(O, I_{i-1}, I_{i}, i\right)}$

## 总结
1. HMM -> MEMM： 
   HMM模型中存在两个假设：一是输出观察值之间严格独立，二是状态的转移过程中当前状态只与前一状态有关。但实际上序列标注问题不仅和单个词相关，而且和观察序列的长度，单词的上下文，等等相关。MEMM解决了HMM输出独立性假设的问题。因为HMM只限定在了观测与状态之间的依赖，而MEMM引入自定义特征函数，不仅可以表达观测之间的依赖，还可表示当前观测与前后多个状态之间的复杂依赖。
2. MEMM -> CRF:
   CRF不仅解决了HMM输出独立性假设的问题，还解决了MEMM的标注偏置问题，MEMM容易陷入局部最优是因为只在局部做归一化，而CRF统计了全局概率，在做归一化时考虑了数据在全局的分布，而不是仅仅在局部归一化，这样就解决了MEMM中的标记偏置的问题。使得序列标注的解码变得最优解。HMM、MEMM属于有向图，所以考虑了x与y的影响，但没讲x当做整体考虑进去（这点问题应该只有HMM）。CRF属于无向图，没有这种依赖性，克服此问题。


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
word2vec工具是为了解决上述问题而提出的。它将每个词映射到一个固定长度的向量，这些向量能更好地表达不同词之间的相似性和类比关系。word2vec工具包含两个模型，即跳元模型（skip-gram） [Mikolov et al., 2013b]和连续词袋（CBOW） [Mikolov et al., 2013a]。对于在语义上有意义的表示，它们的训练依赖于条件概率，条件概率可以被看作是使用语料库中一些词来预测另一些单词。由于是不带标签的数据，因此跳元模型和连续词袋都是自监督模型。

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

## 负采样
根据word2vec论文中的建议，将噪声词的采样概率设置为其在字典中的相对频率，其幂为0.75。

