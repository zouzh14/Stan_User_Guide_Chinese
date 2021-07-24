## 1.2 QR 重参数化

在前面的例子中，线性预测值可以写成$\eta=x \beta$，其中$\eta$是代表预测值的N-向量，$x$是$N \times K$矩阵，$\beta$是用K-向量保存的系数。假设$N \ge K$，我们可以利用任意设计矩阵$x$都能用thin QR分解成正交矩阵$Q$和上三角矩阵$R$，即$x=QR$。

函数`qr_thin_Q`和`qr_thin_R`实现了thin QR分解，它们要优于fat QR分解`qr_Q`和`qr_R`，因为fat QR分解函数运行中容易超出内存(请查看Stan Functions Reference中关于`qr_thin_Q`和`qr_thin_R`的更多信息)。实践中最好写成$x=Q^*R^*$，其中$Q^* = Q \times \sqrt{n-1}$，$R^* = \sqrt{1}{\sqrt{n-1}}R$。故我们可以等价地写出$\eta = x \beta = QR \beta = Q^* R^* \beta$。如果我们记$\theta = R^* \beta$，那么有$\eta = Q^* \theta$，$\beta = R^{*-1} \theta$。前面的Stan程序代码变成

```
data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors
  matrix[N, K] x;   // predictor matrix
  vector[N] y;      // outcome vector
}
transformed data {
  matrix[N, K] Q_ast;
  matrix[K, K] R_ast;
  matrix[K, K] R_ast_inverse;
  // thin and scale the QR decomposition
  Q_ast = qr_thin_Q(x) * sqrt(N - 1);
  R_ast = qr_thin_R(x) / sqrt(N - 1);
  R_ast_inverse = inverse(R_ast);
}
parameters {
  real alpha;           // intercept
  vector[K] theta;      // coefficients on Q_ast
  real<lower=0> sigma;  // error scale
}
model {
  y ~ normal(Q_ast * theta + alpha, sigma);  // likelihood
}
generated quantities {
  vector[K] beta;
  beta = R_ast_inverse * theta; // coefficients on x
}
```

因为这段Stan程序和之前的程序比，生成的$y$预测值相等，$\alpha$、$\beta$和$\sigma$的先验分布也相同，许多人想知道为什么QR重参数化在使用中表现要好这么多，不论是从wall time还是从有效样本数目上看。这有以下的原因：

1. $Q^*$的列都是正交的，而$x$的不是。因此比起$\beta$空间中，在$\theta$空间中马尔科夫链跳转更容易。

2. $Q^*$的列在数值范围上相同，而$x$一般不是。对哈密顿蒙特卡罗(HMC)算法，它可以在这样的参数空间中用更少的步数、更大的步长移动。

3. 因为$Q^*$列之间的协方差矩阵是单位阵，如果$y$的单位选的合理，那么$\theta$的数值范围一般也合理。这可以在保证数值精确的情况下，帮助HMC更有效率的跳转。

综上，在Stan中当$K>1$且对$\beta$缺乏有信息量的先验时，非常推荐对线性模型和广义线性模型使用QR重参数化。在QR重参数化前对$x$的每一列减去各自的均值也是可以的，这不会影响到$\theta$和$\beta$的后验分布，但会影响到$\alpha$的后验。在线性模型中，可以将$\alpha$解释成$y$的期望。