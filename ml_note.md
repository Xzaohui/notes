# 决策树
## ID3算法 熵

信息熵    $H(D)=-\sum_{k=1}^{K} \frac{\left|C_{k}\right|}{|D|} \log _{2} \frac{\left|C_{k}\right|}{|D|}$

每个决策的条件熵 $\begin{aligned} H(D \mid A) &=\sum_{i=1}^{n} \frac{\left|D_{i}\right|}{|D|} H\left(D_{i}\right)  =-\sum_{i=1}^{n} \frac{\left|D_{i}\right|}{|D|}\left(\sum_{k=1}^{K} \frac{\left|D_{i k}\right|}{\left|D_{i}\right|} \log _{2} \frac{\left|D_{i k}\right|}{\left|D_{i}\right|}\right) \end{aligned}$

其中$D_i$表示 D 中特征 A 取第 i 个值的样本子集，$D_{i,k} $表示 $D_i$ 中属于第 k 类的样本子集

信息熵减去条件熵得出信息增益 $\operatorname{Gain}(D,A)=H(D)-H(D \mid A)$

### 缺点

没有剪枝策略，容易过拟合

信息增益准则对可取值数目较多的特征有所偏好

只能处理离散分布的特征

没有考虑缺失值

## C4.5算法

引入悲观剪枝策略进行后剪枝；

引入信息增益率/比作为划分标准； 

将连续特征离散化，假设 n 个样本的连续特征 A 有 m 个取值，C4.5 将其排序并取相邻两样本值的平均数共 m-1 个划分点，分别计算以该划分点作为二元分类点时的信息增益，并选择信息增益最大的点作为该连续特征的二元离散分类点； 

对于缺失值的处理可以分为两个子问题：问题一：在特征值缺失的情况下进行划分特征的选择？（即如何计算特征的信息增益率）问题二：选定该划分特征，对于缺失该特征值的样本如何处理？（即到底把这个样本划分到哪个结点里）

针对问题一，C4.5 的做法是：对于具有缺失值特征，用没有缺失的样本子集所占比重来折算；  
针对问题二，C4.5 的做法是：将样本同时划分到所有子节点，不过要调整样本的权重值，其实也就是以不同概率划分到不同节点中。

$\operatorname{Gain}_{radio}(D,A)=\frac{\operatorname{Gain}(D,A)}{H(D)} $

这里需要注意，信息增益率对可取值较少的特征有所偏好（分母越小，整体越大），因此 C4.5 并不是直接用增益率最大的特征进行划分，而是使用一个启发式方法：先从候选划分特征中找到信息增益高于平均值的特征，再从中选择增益率最高的。

## CART 分类与回归树
$\begin{aligned} \operatorname{Gini}(D) =\sum_{k=1}^{K} \frac{\left|C_{k}\right|}{|D|}\left(1-\frac{\left|C_{k}\right|}{|D|}\right) =1-\sum_{k=1}^{K}\left(\frac{\left|C_{k}\right|}{|D|}\right)^{2} \end{aligned}$

$\begin{aligned}  \operatorname{Gini}(D \mid A) &=\sum_{i=1}^{n} \frac{\left|D_{i}\right|}{|D|} \operatorname{Gini}\left(D_{i}\right) \end{aligned}$

分裂：分裂过程是一个二叉递归划分过程，其输入和预测特征既可以是连续型的也可以是离散型的，CART 没有停止准则，会一直生长下去； 

剪枝：采用代价复杂度剪枝，从最大树开始，每次选择训练数据熵对整体性能贡献最小的那个分裂节点作为下一个剪枝对象，直到只剩下根节点。CART 会产生一系列嵌套的剪枝树，需要从中选出一颗最优的决策树； 

树选择：用单独的测试集评估每棵剪枝树的预测性能（也可以用交叉验证）



## 差异

划分标准的差异：ID3 使用信息增益偏向特征值多的特征，C4.5 使用信息增益率克服信息增益的缺点，偏向于特征值小的特征，CART 使用基尼指数克服 C4.5 需要求 log 的巨大计算量，偏向于特征值较多的特征。

使用场景的差异：ID3 和 C4.5 都只能用于分类问题，CART 可以用于分类和回归问题；ID3 和 C4.5 是多叉树，速度较慢，CART 是二叉树，计算速度很快；

样本数据的差异：ID3 只能处理离散数据且缺失值敏感，C4.5 和 CART 可以处理连续性数据且有多种方式处理缺失值；
从样本量考虑的话，小样本建议 C4.5、大样本建议 CART。C4.5 处理过程中需对数据集进行多次扫描排序，处理成本耗时较高，而 CART 本身是一种大样本的统计方法，小样本处理下泛化误差较大 ；样本特征的差异：ID3 和 C4.5 层级之间只使用一次特征，CART 可多次重复使用特征；

剪枝策略的差异：ID3 没有剪枝策略，C4.5 是通过悲观剪枝策略来修正树的准确性，而 CART 是通过代价复杂度剪枝。


# 支持向量机

原始问题

$\min_{w,b} 1/2||w||^2$ 

s.t. $y_i(w^Tx_i+b)-1>0$

拉格朗日乘子法
对偶问题将minmax转换为$max_a min_{w,b}$

$L(w,b,a)=1/2||w||-\sum_{i=1}^N a_iyi(w*x_i+b)+\sum_{i=1}^Na_i$

对w、b求偏导等于0 $\quad \ \ w-\sum_{i=1}^N a_iy_ix_i=0$  $\quad \ \ \sum_{i=1}^N a_iy_i=0$

$max_a -1/2\sum _{i=1}^N\sum _{j=1}^Na_ia_jy_iy_j(x_i*x_j)+\sum_{i=1}^Na_i$


等价于
$min_a 1/2\sum _{i=1}^N\sum _{j=1}^Na_ia_jy_iy_j(x_i*x_j)-\sum_{i=1}^Na_i$

s.t.  $\sum_{i=1}^N a_iy_i=0$

$\quad \ \ a_i\geq 0$

核函数$K(x_i,x_j)=\phi(x_i)*\phi(x_j)$

映射函数$\phi$将原本的输入空间映射到新的空间中，将输入空间的内积$x_i*x_j$映射到新的空间$\phi(x_i)*\phi(x_j)$中


# K近邻 K-NN
在训练集种找到与输入X最近的k个点，然后投票决定输入X的类别。可以用欧氏距离也可以用其他如曼哈顿距离、Minkowski距离等。

## kd树

### 切分

kd树的每轮切分点的选择策略比较简单，是将带切分平面上的所有数据点按照切分域的维度进行大小排序，选择正中间的点作为切分点。

### 查找

1 寻找近似点-寻找最近邻的叶子节点作为目标数据的近似最近点。

2 回溯-以目标数据和最近邻的近似点的距离沿子节点到父节点方向进行回溯和迭代。

# K均值聚类 K-Means

算法接受参数 k ；然后将事先输入的n个数据对象划分为k个聚类以便使得所获得的聚类满足：同一聚类中的对象相似度较高；而不同聚类中的对象相似度较小。聚类相似度是利用各聚类中对象的均值所获得一个“中心对象”（引力中心）来进行计算的。

（1）适当选择c个类的初始中心；

（2）在第k次迭代中，对任意一个样本，求其到c个中心的距离，将该样本归到距离最短的中心所在的类；

（3）利用均值等方法更新该类的中心值；

（4）对于所有的c个 聚类中心，如果利用（2）（3）的 迭代法更新后，值保持不变，则迭代结束，否则继续迭代。

## 代码
    
```python
for t in range(self.times):
    for index, x in enumerate(X):
        dis = np.sqrt(np.sum(x-self.cluster_centers)**2, axis=1)
        # 将最小距离的所有赋值给标签数组，索引的值就说当前点所属的簇
        self.labels_[index] = dis.argmin()
    # 循环遍历每个簇
    for i in range(self.k):
        # 计算每个簇内的所有的点均值，更新聚类中心
        self.cluster_centers[i] = np.mean(X[self.labels_ == i], axis=0)
```
## 区别
KMeans与KNN的区别：

(1)KMeans是无监督学习算法，KNN是监督学习算法。

(2)KMeans算法的训练过程需要反复迭代的操作（寻找新的质心），但是KNN不需要。

(3)KMeans中的K代表的是簇中心，KNN的K代表的是选择与新测试样本距离最近的前K个训练样本数。
    
(4)KMeans算法的训练过程中，每次迭代都需要遍历所有的训练样本，而KNN算法的训练过程中，只需要遍历前K个训练样本即可。

# 朴素贝叶斯

朴素贝叶斯算法是假设各个特征之间相互独立
$P(Y|X)= \frac{P(X|Y)*P(Y)}{P(X)}=\frac{\prod P(x_i|Y)*P(Y)}{\prod P(x_i)} $ 

优点：

（1） 算法逻辑简单,易于实现

（2）分类过程中时空开销小

缺点：

理论上，朴素贝叶斯模型与其他分类方法相比具有最小的误差率。但是实际上并非总是如此，这是因为朴素贝叶斯模型假设属性之间相互独立，这个假设在实际应用中往往是不成立的，在属性个数比较多或者属性之间相关性较大时，分类效果不好。
而在属性相关性较小时，朴素贝叶斯性能最为良好。对于这一点，有半朴素贝叶斯之类的算法通过考虑部分关联性适度改进。


# 逻辑斯蒂回归 LR

$\begin{array}{l}
P(Y=1 \mid x)=\frac{\exp (w \cdot x)}{1+\exp (w \cdot x)} ~~~~~~
P(Y=0 \mid x)=\frac{1}{1+\exp (w \cdot x)}
\end{array}$

现在考查逻辑斯谛回归模型的特点。一个事件的几率 （odds）是指该事件发生的概率与该事件不发生的概率的比值。如果事件发生的概率是p，那么该事件的几率是p/1-p，该事件的对数几率 （log odds）或 logit 函数是:

$\operatorname{logit}(p)=\log \frac{p}{1-p}=\log \frac{P(Y=1 \mid x)}{1-P(Y=1 \mid x)}=w \cdot x$

这就是说，在逻辑斯谛回归模型中，输出Y=1 的对数几率是输入x的线性函数。或者说，输出Y=1的对数几率是由输入x的线性函数表示的模型，即逻辑斯谛回归模型。

https://www.bilibili.com/video/BV17r4y137bW?spm_id_from=333.999.0.0&vd_source=cda54caf7daa0c1a91d08c632937453b 

令
$P(Y=1 \mid x)=\pi(x), \quad P(Y=0 \mid x)=1-\pi(x)$

似然函数为
$\prod_{i=1}^{N}\left[\pi\left(x_{i}\right)\right]^{y_{i}}\left[1-\pi\left(x_{i}\right)\right]^{1-y_{i}}$


取对数
$\begin{aligned} 
L(w) &=\sum_{i=1}^{N}\left[y_{i} \log \pi\left(x_{i}\right)+\left(1-y_{i}\right) \log \left(1-\pi\left(x_{i}\right)\right)\right] \\
&=\sum_{i=1}^{N}\left[y_{i} \log \frac{\pi\left(x_{i}\right)}{1-\pi\left(x_{i}\right)}+\log \left(1-\pi\left(x_{i}\right)\right)\right] \\
&=\sum_{i=1}^{N}\left[y_{i}\left(w \cdot x_{i}\right)-\log \left(1+\exp \left(w \cdot x_{i}\right)\right]\right.
\end{aligned}$

对L(w)取极大得到w


多项逻辑斯谛回归模型

$P(Y=k \mid x)=\frac{\exp \left(w_{k} \cdot x\right)}{1+\sum_{k=1}^{K-1} \exp \left(w_{k} \cdot x\right)}, \quad k=1,2, \cdots, K-1 \\
P(Y=K \mid x)=\frac{1}{1+\sum_{k=1}^{K-1} \exp \left(w_{k} \cdot x\right)}$

## 逻辑回归的优点

直接对分类可能性进行建模，无需实现假设数据分布，这样就避免了假设分布不准确所带来的问题。

形式简单，模型的可解释性非常好，特征的权重可以看到不同的特征对最后结果的影响。

除了类别，还能得到近似概率预测，这对许多需利用概率辅助决策的任务很有用。

## 逻辑回归的缺点

准确率不是很高，因为形势非常的简单，很难去拟合数据的真实分布。

本身无法筛选特征。

## 逻辑回归用一句话总结

逻辑回归假设数据服从伯努利分布，通过极大似然函数的方法，运用梯度下降来求解参数，来达到将数据二分类的目的。
# EM算法
E步,求期望(expectation);M步,求极大(maximization）

https://www.bilibili.com/video/BV1RT411G7jJ?spm_id_from=333.999.0.0&vd_source=cda54caf7daa0c1a91d08c632937453b

输入：观测变量数据Y，隐变量数据Z，联合分布$P(Y,Z|\theta)$，条件分布$P(Z|Y,\theta)$；

输出：模型参数$\theta$。

(1)选择参数的初值$\theta^0$，开始迭代；

(2)E步：记$\theta^i$ 为第i次迭代参数$\theta$的估计值，在第i十1次迭代的E步，计算

$\begin{aligned}
Q\left(\theta, \theta^{(i)}\right) &=E_{Z}\left[\log P(Y, Z \mid \theta) \mid Y, \theta^{(i)}\right] \\
&=\sum_{Z} \log P(Y, Z \mid \theta) P\left(Z \mid Y, \theta^{(i)}\right)
\end{aligned}$

(3) M 步: 求使  $Q\left(\theta, \theta^{(i)}\right)$ 极大化的  $\theta$ , 确定第  i+1  次迭代的参数的估计值  $\theta^{(i+1)}$

$\theta^{(i+1)}=\arg \max _{\theta} Q\left(\theta, \theta^{(i)}\right)$

(4)重复第(2)步和第(3)步，直到收敛。

# 概率图模型：HMM、MEMM、CRF

![Alt](https://pic3.zhimg.com/80/v2-714c1843f78b6aecdb0c57cdd08e1c6a_1440w.jpg?source=1940ef5c)

![Alt](https://pica.zhimg.com/80/v2-376fd85a490e161978130ddd759244d4_1440w.jpg?source=1940ef5c)

上图可以看到, 贝叶斯网络(信念网络) 都是有向的, 马尔科夫网络无向。所以, 贝叶斯网络适 合为有单向依赖的数据建模, 马尔科夫网络适合实体之间互相依赖的建模。具体地, 他们的核心差 异表现在如何求  $P=(Y)$, 即怎么表示  $Y=\left(y_{1}, \cdots, y_{n}\right)$  这个的联合概率。

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
1. 马尔科夫假设额应该是齐次马尔科夫假设，这样假设：马尔科夫链 里的  总是只受一个人的影响。马尔科夫假设这里相当于就是个2-gram。马尔科夫过程呢？即在一个过程中，每个状态的转移只依赖于前n个状态，并且只是个n阶的模型。最简单的马尔科夫过程是一阶的，即只依赖于器哪一个状态。

2. 马尔科夫性马尔科夫性是是保证或者判断概率图是否为概率无向图的条件。三点内容：a. 成对，b. 局部，c. 全局。


## 生成式模型&判别式模型
生成式模型表示了给定输入X产生输出Y的生产关系， $P(X,Y)=P(Y)*P(X|Y)$ 

对联合分布概率p(x,y)进行建模，$P(Y)、P(X|Y)$均由样本直接计算得出，$P(X)$由样本的全部输入计算得出，可以得出一个输入X为Y的概率。可以直接由$P(Y)*P(X|Y)$ 判断分类，也可根据$P(Y \mid X)=\frac{P(X, Y)}{P(X)}$。

判别式模型是有数据直接学习决策函数或者条件概率分布的模型，直接对条件概率$P(Y|X)$或$P(X)$进行建模。

条件概率分布$P(Y|X)$可以理解为：在已知某样本的特征为X的条件下，计算该样本类别为类别Y1、Y2、Y3的概率，并选择概率最大的类别为该样本的预测类别。


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

（2）生成式模型的缺点：因为生成式模型需要生成Y的分布函数，而这个分布函数可能会受到一些异常点的影响变得不那么准确，如下图所示，有两个黄色小球乱入了绿军阵营，有两个绿色小球混进了黄球阵营。

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
深度信念网络（DBN）

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

### B

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

## MEMM 最大熵马尔科夫模型
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
# 监督学习总结
![Alt](./picture/1.png)

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
&=\left.\frac{1}{2}|| w\right|^{2}+C \sum_{i} L_{H i n g e}
\end{aligned}$

SVM的损失函数可以看做是L2正则化与Hinge loss之和。