## 1.1 线性回归

最简单的线性回归模型如下，有一个预测因子、斜率系数和截距系数，以及服从正态分布的噪声。这个模型可以写成标准的回归形式

$$y_n = \alpha + \beta x_n + \epsilon_n \quad where \quad \epsilon_n \sim normal(0,\sigma).$$

这相当于以下带残差的抽样

$$y_n - (\alpha + \beta x_n) \sim normal(0,\sigma),$$

化简后

$$y_n \sim normal(\alpha + \beta x_n,\sigma).$$

在Stan里，后面形式的模型为

```
data {
  int<lower=0> N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  y ~ normal(alpha + beta * x, sigma);
}
```

这里有`N`个观测值，对观测值$n \in N$，我们有预测变量`x[n]`和结局变量`y[n]`。截距和斜率为`alpha`和`beta`。模型假设了一个正态噪声，规模为`sigma`。模型对两个回归系数有不合适的先验。

### 矩阵符号化和向量化

在前面的模型中，抽样陈述被向量化了，即

```
y \sim normal(alpha + beta * x, sigma)
```

相同模型不向量化的版本为

```
for (n in 1:N)
  y[n] ~ normal(alpha + beta * x[n], sigma);
```

除了更简洁外，向量化版本也更快。

一般来说，Stan允许分布的参数是向量，例如`normal`分布。如果任何其他参数是向量或数组，它们必须相同大小。如果其他参数中的任何一个是标量，它将被用来填充整个向量。关于概率函数矢量化的更多信息，请参见[向量化](https://mc-stan.org/docs/2_27/stan-users-guide/linear-regression.html#vectorization.section)。

这样做的另一个原因是，Stan的算术运算符被重载，可以对矩阵进行矩阵运算。在这种情况下，因为`x`是`vector`类型，`beta`是`real`类型，所以表达式`beta * x`是向量类型。因为Stan支持向量化，一个有多个预测因子的回归模型可以直接用矩阵符号来写。

```
data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors
  matrix[N, K] x;   // predictor matrix
  vector[N] y;      // outcome vector
}
parameters {
  real alpha;           // intercept
  vector[K] beta;       // coefficients for predictors
  real<lower=0> sigma;  // error scale
}
model {
  y ~ normal(x * beta + alpha, sigma);  // likelihood
}
```

参数`sigma`声明中的约束项`lower=0`限制该值大于或等于0。模型块中没有先验时，该约束效果相当于对非负实数的不恰当先验。尽管可以添加一个信息量更大的先验，但只要能导致适当的后验，不适当的先验是可以接受的。

在上面的模型中，`x`是一个$N \times K$预测因子矩阵，`beta`是一个系数的$K-$向量，所以`x * beta`是一个  预测值的$N-$向量，每一维对应着`N`个样本中的一个。输出的预测结果被写成$N-$向量`y`，所以整个模型可以用矩阵运算来写。可以在数据矩阵`x`中加入一列1，以去除`alpha`参数。

上述模型中的取样语句只是一种更有效的、基于向量的方法。下面用循环来编码模型，一个统计学等效的模型。

```
model {
  for (n in 1:N)
    y[n] ~ normal(x[n] * beta, sigma);
}
```

按Stan的矩阵索引方案，`x[n]`选取矩阵`x`的第`n`行；因为`beta`是一个列向量，所以积`x[n] * beta`是一个`real`类型的标量。

### 作为输入的截距

在如下模型中

```
y ~ normal(x * beta, sigma);
```

不再有一个截距系数`alpha`。相反，我们假设输入矩阵`x`的第一列是一列1。这样一来，`beta[1]`就扮演了截距的角色。如果截距和斜率项先验不同，那么把它拆出来会更清楚。在截距形式明确的情况下，单列出来的效率也会稍高一些，因为少了一次乘法；不过，这对速度的影响应该不大，所以选择应该以清晰为基础。

------------------------------------------

1. 与Python和R不同的是，Stan被翻译成C++并被编译，所以循环和赋值语句都很快。向量化代码在Stan中更快，因为(a)用于计算导数的表达式树可以被简化，导致更少的虚拟函数调用，以及(b)在循环版本中重复的计算，如上述模型中的`log(sigma)`，只会计算一次并重新使用。