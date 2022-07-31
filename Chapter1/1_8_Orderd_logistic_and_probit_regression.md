## 1.8 Ordered logistic回归和probit回归

对输出$y_n \in \{ 1, ..., k \}$，预测变量$x_n \in R^D$，Ordered回归由单个系数向量 $\beta \in R^D$带上排序好的切点序列 $c \in R^{K-1}$ 决定，$c_d < c_{d+1}$ . 当$x_n \beta$落到$c_{k-1}$和$c_k$之间时，输出离散值${k}$，这里假设$c_0 = -\infin, c_K = \infin$。噪声项根据回归形式是固定的，这里有ordered Logistic和ordered Probit模型的例子。

### Ordered Logistic回归
在Stan中，可以使用`ordered`数据类型编码切点，和内置的`ordered_logistic`分布来建立ordered logistic模型。

```
data {
  int<lower=2> K;
  int<lower=0> N;
  int<lower=1> D;
  int<lower=1,upper=K> y[N];
  row_vector[D] x[N];
}
parameters {
  vector[D] beta;
  ordered[K-1] c;
}
model {
  for (n in 1:N)
    y[n] ~ ordered_logistic(x[n] * beta, c);
}
```

切点向量`c`被声明为`ordered[K-1]`，它保证`c[k]`小于`c[k+1]`。

如果切点被分配了独立的先验，那么这个约束就有效地把联合先验截断到了那些满足ordering约束的支撑点上。幸运的是，Stan不需要计算约束对归一化项的影响，因为需要的概率只用算一个比例。

### Ordered probit
一个ordered probit模型可以用完全相同的方式进行编码，只要将累积Logistic（`inv_logit`）替换成累积正态（`Phi`）。

```
data {
  int<lower=2> K;
  int<lower=0> N;
  int<lower=1> D;
  int<lower=1,upper=K> y[N];
  row_vector[D] x[N];
}
parameters {
  vector[D] beta;
  ordered[K-1] c;
}
model {
  vector[K] theta;
  for (n in 1:N) {
    real eta;
    eta = x[n] * beta;
    theta[1] = 1 - Phi(eta - c[1]);
    for (k in 2:(K-1))
      theta[k] = Phi(eta - c[k-1]) - Phi(eta - c[k]);
    theta[K] = Phi(eta - c[K-1]);
    y[n] ~ categorical(theta);
  }
}
```

Logistic模型也可以这样编码，用`inv_logit`代替`Phi`，不过基于`softmax`变换的内置编码效率更高，数值也更稳定。计算一次`Phi(eta-c[k])`的值并将其存储起来以便重复使用，可以获得较少的效率提升。