# 逻辑斯蒂回归和概率单位回归

对于二元结果问题，可以应用逻辑斯蒂回归或概率单位回归模型。这些广义线性模型的区别仅在于，它们将线性预测值从$(-\infty, \infty)$映射到$(0,1)$上所用的链接函数不同。它们的链接函数，即逻辑斯蒂函数和标准正态累积分布函数都是sigmoid函数（即都是S形的）。 

有一个预测变量和一个截距的逻辑斯蒂回归模型代码如下。

```
data { 
int<lower=0> N; 
vector[N] x; 
int<lower=0,upper=1> y[N]; 
} 
paramters { 
real alpha;
real beta; 
} 
model { 
y ~ bernoulli_logit(alpha + beta * x); 
} 
```

噪声参数内置在伯努利表达式里，而不是特别给出。

逻辑斯蒂回归是一种输出二元结果的广义线性模型，使用log odds链接函数，定义为  

$$logit(v) = log (\frac{v}{1-v}).$$
  
链接函数的反函数： 

$$logit^{-1}(u)=inv\_logit(u)= \frac{1}{1+exp(-u)}.$$

上面的模型中使用了伯努利分布的逻辑斯蒂参数化版本，定义为 

$$ bernoulli\_logit (y | \alpha) = bernoulli (y | logit^{−1}(\alpha)).$$

该表达式中`alpha`和`beta`是标量，而`x`是向量，因此`alpha + beta * x`是向量。该向量化表达式和下面这个低效形式等价：

``` 
for (n in 1:N) 
  y[n] ~ bernoulli_logit(alpha + beta * x[n]); 
```

把伯努利逻辑斯蒂表达式展开，模型和下面这个版本等价，它更明确，但更低效，数值计算上更不稳定

```
for (n in 1:N) 
y[n] ~ bernoulli(inv_logit(alpha + beta * x[n])); 
```

其他链接函数以相同的方式调用。例如，概率单位回归使用的累积正态分布函数，通常写为

$$\Phi(x) = \int_{\infty}^x normal(y|0,1) dy.$$

标准正态累积分布函数$\Phi$在Stan中为函数`Phi`。将逻辑斯蒂语句的抽样函数换成如下所示后，Stan也可以构建出概率单位回归模型。

```
y[n] ~ bernoulli(Phi(alpha + beta * x[n]));
```

标准正态累积分布函数$\Phi$的快速近似在Stan中实现为函数`Phi_approx`。近似概率单位回归模型的代码为
```
y[n] ~ bernoulli(Phi_approx(alpha + beta * x[n]));
``` 

---------------------

Phi_approx函数对logit逆函数进行了重新缩放，因此数值规模和$\Phi$有出入时，尾部不匹配。