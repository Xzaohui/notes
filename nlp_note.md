# 决策树
## ID3算法 熵

总熵    $I\left(s_{1}, s_{2}, \ldots, s_{m}\right)=-\sum_{i=1}^{m} P_{i} \log _{2} P_{i}$

每个决策的熵 $E(A)=\sum_{j=1}^{k}\left[\frac{s_{1 j}, s_{2 j}, \ldots, s_{m j}}{S} \times I\left(s_{1 j}, s_{2 j}, \ldots, s_{m j}\right)\right]$

每个决策的判别 $\operatorname{Gain}(A)=I\left(s_{1}, s_{2}, \ldots, s_{m}\right)-E(A)$

## 基尼系数 Gini
$Gini=1-\sum_{i=1}^{m} P_{i}^2$

$\operatorname{Gain}(A)=\sum_{j=1}^{k}\left[\frac{s_{1 j}, s_{2 j}, \ldots, s_{m j}}{S} \times (1-\sum_{i=1}^{m} P_{i}^2) \right]$

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






