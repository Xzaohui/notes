# 决策树
## ID3算法 熵

信息熵    
$$H(D)=-\sum_{k=1}^{K} \frac{\left|C_{k}\right|}{|D|} \log _{2} \frac{\left|C_{k}\right|}{|D|}$$

每个决策的条件熵 
$$\begin{aligned} H(D \mid A) &=\sum_{i=1}^{n} \frac{\left|D_{i}\right|}{|D|} H\left(D_{i}\right)  =-\sum_{i=1}^{n} \frac{\left|D_{i}\right|}{|D|}\left(\sum_{k=1}^{K} \frac{\left|D_{i k}\right|}{\left|D_{i}\right|} \log _{2} \frac{\left|D_{i k}\right|}{\left|D_{i}\right|}\right) \end{aligned}$$

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

$\operatorname{Gain}_{radio}(D,A)=\frac{\operatorname{Gain}(D,A)}{H(D)}$

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

取对数，结构类似交叉熵，即

$$\begin{aligned} 
L(w) &=\sum_{i=1}^{N}\left[y_{i} \log \pi\left(x_{i}\right)+\left(1-y_{i}\right) \log \left(1-\pi\left(x_{i}\right)\right)\right] \\
&=\sum_{i=1}^{N}\left[y_{i} \log \frac{\pi\left(x_{i}\right)}{1-\pi\left(x_{i}\right)}+\log \left(1-\pi\left(x_{i}\right)\right)\right] \\
&=\sum_{i=1}^{N}\left[y_{i}\left(w \cdot x_{i}\right)-\log \left(1+\exp \left(w \cdot x_{i}\right)\right]\right.
\end{aligned}$$

对$L(w)$取极大得到$w$


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

# 高斯混合模型
高斯混合模型是指具有如下形式的概率分布模型：
$$
p(x \mid \theta)=\sum_{k=1}^{K} \alpha_{k} \phi\left(x \mid \theta_{k}\right)
$$
其中, $\alpha_{k}$ 是系数, 且 $\alpha_{k} \geq 0, \sum_{k=1}^{K} \alpha_{k}=1$, 而 $\phi\left(y \mid \theta_{k}\right)$ 是 高斯分布 $Q$ 密度, $\theta_{k}=\left(\mu_{k}, \sigma_{k}^{2}\right)$, 对于随机变量 $\mathrm{y}$ 是一维数据时,
$$
\phi\left(x \mid \theta_{k}\right)=\frac{1}{\sqrt{2 \pi \sigma_{k}^{2}}} \exp \left(\frac{\left(x-\mu_{k}\right)^{2}}{2 \sigma_{k}^{2}}\right)
$$
称为第k个分模型。

GMM是多个高斯分布叠加而成的加权平均的结果，理论上只要分模型足够多，并且各分模型的系数设置合理，就能够产生任意分布的样本。公式可简写为：
$p(x)=\sum_{z} p(x, z)=\sum_{k=1}^{K} p(x, z)=\sum_{k=1}^{K} p(z) \cdot p(x \mid z)=\sum_{k=1}^{K} p_{k} \cdot N\left(\mu_{k}, \sum_{k}\right)$

$$
p(x)=\sum_{k=1}^{K} \alpha_{k} \cdot N\left(\mu_{k}, \sum_{k}\right), \sum_{k=1}^{K} \alpha_{k}=1(\alpha_{k} \geq 0)
$$



高斯混合模型属于生成模型, 可以设想观测数据 $y_{j}, j=1,2, \ldots, N$, 是这样生成的: 首先以概率 $\alpha_{k}$ 选择第 $\mathrm{k}$ 个分模型, 然后由第 $\mathrm{k}$ 个分模 型的概率分布生成观测数据 $y_{j}$ 。这里我们先约定: $x^{(i)}$ 表示为第 $\mathrm{i}$ 个样本的观测变量; $z^{(i)}$ 表示第 $\mathrm{i}$ 个样本所属的分模型, 是隐变量; 为了方 便, 统一用 $\theta$ 表示模型参数 $\alpha 、 \mu$ 和 $\sigma^{2}$ (对于一维数据）。观测数据是能直接观测到的, 已知的; 而反映第 $\mathrm{i}$ 个样本来自第 $\mathrm{k}$ 个分模型的数据 是末知的, 称为隐随机变量。一般地, 用 $x$ 表示观测随机变量的数据; $z$ 表示隐随机变量的数据。 $x$ 称为不完全数据, 而 $y$ 和z连在一起称为 完全数据。
为求模型参数, 先计算其似然函数：
$$
\begin{aligned}
L(\theta)&=\prod_{i=1}^{n} p\left(x^{(i)} \mid \theta\right)=\prod_{i=1}^{n} \sum_{k=1}^{K} \alpha_{k} \phi\left(x^{(i)} \mid \mu_{k} ; \sigma_{k}\right) \\
&=\prod_{i=1}^{n} \sum_{k=1}^{K} p\left(z^{(i)}=k \mid \theta\right) p\left(x^{(i)} \mid z^{(i)}=k ; \theta\right) \\
&=\prod_{i=1}^{n} \sum_{k=1}^{K} p\left(x^{(i)}, z^{(i)}=k \mid \theta\right)
\end{aligned}
$$
对数似然为
$$
L L(\theta)=\ln L(\theta)=\sum_{i=1}^{n} \ln \sum_{k=1}^{K} p\left(z^{(i)}=k \mid \theta\right) p\left(x^{(i)} \mid z^{(i)}=k ; \theta\right)
$$
可以看到, GMM通过求不完全数据的边缘概率来得到完全数据的似然函数。其中, $\alpha_{k}$ 对应于 $p\left(z^{(i)}=k \mid \theta\right), \phi\left(x^{(i)} \mid \mu_{k} ; \sigma_{k}\right)$ 对应于 $p\left(x^{(i)} \mid z^{(i)}=k ; \theta\right)$ 。因此
$$
\theta=\underset{\theta}{\arg \max } L L(\theta)
$$

(1) 初始化模型的参数值。EM算法对初始值较敏感, 不同的初始值可能得到不同的参数估计值。

(2) E-step：依据当前模型参数, 计算分模型k对观测样本数据的响应度。
$$
\varpi_{k}^{(i)}=\frac{\alpha_{k} \phi\left(x^{(i)} \mid \theta_{k}\right)}{\sum_{k=1}^{K} \alpha_{k} \phi\left(x^{(i)} \mid \theta_{k}\right)}
$$
(3) M-step: 计算新一轮迭代的模型参数
求模型参数：固定 $\varpi^{(i)}$ 后最大化 
$$
\sum_{i=1}^{n} \sum_{k=1}^{K} \varpi_{k}^{(i)} \ln \frac{p\left(z^{(i)}=k \mid \theta\right) p\left(x^{(i)} \mid z^{(i)}=k ; \theta\right)}{\varpi_{k}^{(i)}}=\sum_{i=1}^{n} \sum_{k=1}^{K} \varpi_{k}^{(i)} \ln \frac{\alpha_{k} \phi\left(x^{(i)} \mid \mu_{k} ; \sigma_{k}\right)}{\varpi^{(i)}}
$$
求解模型参数, 即
$$
\theta^{t+1}=\arg \max _{\theta} \sum_{i=1}^{n} \sum_{k=1}^{K} \varpi_{k}^{(i)} \ln \frac{\alpha_{k} \phi\left(x^{(i)} \mid \mu_{k} ; \sigma_{k}\right)}{c s \varpi_{k}^{(i)} }
$$
(4) 重复步骤 (2) (3) , 直至达到收敛。

# 监督学习总结
![Alt](./picture/1.png)


# 无监督学习
![Alt](./picture/2.png)
![Alt](./picture/3.png)
![Alt](./picture/4.png)


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



