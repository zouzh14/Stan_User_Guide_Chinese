## 1.7 参数化中心向量 

定义一个中心化的参数向量$\beta$通常很方便，即$\beta$满足各分量和为零的约束

$$\sum_{k=1}^{K} \beta_k = 0.$$

这样的参数向量可用于识别multi-logit回归参数向量（详细信息请参阅[multi-logit](https://mc-stan.org/docs/2_27/stan-users-guide/multi-logit-section.html#multi-logit.section)一节），或用于IRT模型中的能力或难度参数（但不能同时用于两者，IRT模型请参阅[项目反应模型](https://mc-stan.org/docs/2_27/stan-users-guide/item-response-models.section)一节）。

### K − 1 自由度

有不止一种方法可以对参数向量执行总和为零的约束，其中最有效的方法是将第K个元素定义为前K-1元素之和的负值。

```
parameters { 
  vector[K-1] beta_raw; 
  ...
transformed parameters { 
  vector[K] beta = append_row(beta_raw, -sum(beta_raw)); 
  ...
```

在参数化中给`beta_raw`加上一个先验后得到的后验，与没有零和限制时给`beta`加相同先验下得到的后验有微小的差异。值得注意的是，把简单的先验加到每个分量上后，`beta_raw`得到的结果，和无约束相同先验下K向量`beta`的结果不同。例如，将`normal(0, 5)`作为各分量的先验，`beta`与`beta_raw`后验不同。 

### 零和分量的边缘分布

在 Stan 论坛上，Aaron Goodman 提供了以下代码，构建出在`beta`各分量上有正态边缘分布的先验

``` 
model { 
  beta ~ normal(0, inv(sqrt(1 - inv(K))))); 
  ...
```

各分量不是独立的，因为它们的总和必须为零。这里不需要雅可比行列式，因为求和与非门是线性运算（因此具有常量雅可比行列式）。

为了生成具有标准正态外的边缘分布，`beta`可以通过某个因子`sigma`进行缩放，并平移到某个新位置`mu`。 

### QR 分解

Stan论坛上的Aaron Goodman也提供了这种方法，在变换后数据块中计算QR分解，然后转换为和为零的参数`x`，

```
transformed data { 
  matrix[K, K] A = diag_matrix(rep_vector(1,K)); 
  matrix[K, K-1] A_qr; 
  for (i in 1:K-1) A[K,i] = -1; 
  A[K,K] = 0; 
  A_qr = qr_Q(A)[ , 1:(K-1)]; 
} 
parameters { 
  vector[K-1] beta_raw; 
} 
transformed parameters { 
  vector[K] beta = A_qr * beta_raw; 
} 
model { 
  beta_raw ~ normal(0, inv(sqrt(1 - inv(K)))); 
}
```

这里在`beta`上生成了边缘正态分布，通过QR分解确保总和为零。

### 变换与缩放后的单纯形

另一种效率较低，但适合于对称先验的方法是变换与缩放一个单纯形。

```
parameters {
  simplex[K] beta_raw;
  real beta_scale;
  ...
transformed parameters {
  vector[K] beta;
  beta = beta_scale * (beta_raw - inv(K));
  ...
```

这里`inv(K)`是`1.0 / K`的简写。给定`beta_raw`为一个单纯形，它各元素和为1，逐元素减去`inv(K)`保证和为零。因为单纯形元素大小是有界的，所以需要一个比例因子来为`beta`提供K自由度，以使其可以取到各个和为零的可能值。

有了这里的参数化，可以给`beta_raw`一个Dirichlet先验，有可能是均匀的，而在`beta_scale`上放置另一个先验，通常是为了“收缩”。

### 软中心化

加上一个像$\beta \sim normal(0,\sigma)$这样的先验将会带来参数$\beta$的软中心化，相较于等价的$\sum_{k=1}^K \beta_k = 0$做法。这种方法只有在给定条件才能保证大致中心化，即$\beta$和$\beta+c$（c是标量常数）制造出相同的似然函数（也许通过另一个向量$\alpha$来变换到$\alpha-c$，像在IRT模型中那样）。这是另一种制造对称先验的方法。