# 第一章 回归模型

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

## 1.3 系数和数值范围的先验

对回归参数先验分布，请参阅我们[对先验的一般讨论](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations)一文。 

后面的章节会讨论[单变量分层先验](https://mc-stan.org/docs/2_27/stan-users-guide/hierarchical-priors-section.html#hierarchical-priors.section)和[多变量分层先验](https://mc-stan.org/docs/2_27/stan-users-guide/multivariate-hierarchical-priors-section.html#multivariate-hierarchical-priors.section)，以及[用于辨识模型的先验](https://mc-stan.org/docs/2_27/stan-users-guide/priors-for-identification-section.html#priors-for-identification.section)。 

但是，如[QR重参数化](https://mc-stan.org/docs/2_27/stan-users-guide/QR-reparameterization-section.html#QR-reparameterization.section)章节所述，如果没有回归系数location的有信息量的先验认识，最好重参数化模型，以便回归系数是生成的数值。在这种情况下，重参数化的回归系数上使用什么先验通常并不重要，几乎任何和输出值数值范围相似的弱先验都可以。 

## 1.4 稳健的噪声模型 

线性回归的标准方法是对噪声项$\epsilon$用正态分布建模。从Stan的角度看，正态噪声没有什么特别。例如，可以通过给噪声项学生t分布来构建稳健回归模型。在Stan中，采样分布对应的代码如下。
```
data { 
... 
real<lower=0> nu; 
} 
... 
model{ 
y ~ student_t(nu, alpha + beta * x, sigma); 
}
```
数据中自由度常数`nu`是给定的。

## 1.5 Logistic回归和probit回归

对于二元结果问题，可以应用Logistic回归或probit回归模型。这些广义线性模型的区别仅在于，它们将线性预测值从$(-\infty, \infty)$映射到$(0,1)$上所用的链接函数不同。它们的链接函数，即Logistic函数和标准正态累积分布函数都是sigmoid函数（即都是S形的）。 

有一个预测变量和一个截距的Logistic回归模型代码如下。

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

Logistic回归是一种输出二元结果的广义线性模型，使用log odds链接函数，定义为  

$$logit(v) = log (\frac{v}{1-v}).$$
  
链接函数的反函数： 

$$logit^{-1}(u)=inv\_logit(u)= \frac{1}{1+exp(-u)}.$$

上面的模型中使用了伯努利分布的Logistic参数化版本，定义为 

$$ bernoulli\_logit (y | \alpha) = bernoulli (y | logit^{−1}(\alpha)).$$

该表达式中`alpha`和`beta`是标量，而`x`是向量，因此`alpha + beta * x`是向量。该向量化表达式和下面这个低效形式等价：

``` 
for (n in 1:N) 
  y[n] ~ bernoulli_logit(alpha + beta * x[n]); 
```

把伯努利Logistic表达式展开，模型和下面这个版本等价，它更明确，但更低效，数值计算上更不稳定

```
for (n in 1:N) 
y[n] ~ bernoulli(inv_logit(alpha + beta * x[n])); 
```

其他链接函数以相同的方式调用。例如，probit回归使用的累积正态分布函数，通常写为

$$\Phi(x) = \int_{\infty}^x normal(y|0,1) dy.$$

标准正态累积分布函数$\Phi$在Stan中为函数`Phi`。将Logistic语句的抽样函数换成如下所示后，Stan也可以构建出probit回归模型。

```
y[n] ~ bernoulli(Phi(alpha + beta * x[n]));
```

标准正态累积分布函数$\Phi$的快速近似在Stan中实现为函数`Phi_approx`。近似probit回归模型的代码为
```
y[n] ~ bernoulli(Phi_approx(alpha + beta * x[n]));
``` 

---------------------

2. Phi_approx函数对logit逆函数进行了重新缩放，因此数值规模和$\Phi$有出入时，尾部不匹配。

## 1.6 多类Logistic回归

多类输出情况下的Logistic回归可以直接在Stan中进行。记对每个输出变量$y_n$都有$K$个可能结果，它对应的$x_n$是一个D维向量。系数先验为$normal(0,5)$的多类Logistic回归代码如下。

```
data { 
  int K; 
  int N; 
  int D; 
  int y[N]; 
  matrix[N, D] x; 
} 
parameters { 
  matrix[D, K] beta; 
} 
model { 
  matrix[N, K] x_beta = x * beta; 
  
  to_vector(beta) ~ normal(0, 5); 
  
  for (n in 1:N) 
    y[n] ~ categorical_logit(x_beta[n]'); 
}
```

其中 `x_beta[n]' `是`x_beta[n]`的转置。`beta`的先验也是向量形式。从Stan 2.18开始，categorical-logit分布参数未向量化，编程中需要循环。这里用矩阵乘法来定义一个对应所有预测变量的局部变量，来提高效率。与Bernoulli-logit一样，categorical-logit分布在内部应用softmax将任意向量转换为单纯形，

$$categorical\_logit (y | \alpha) = categorical (y | softmax(\alpha)),$$
 
其中

$$softmax(u) = exp(u)/ sum(exp(u)).$$

上面使用log-odds (logit)缩放参数的分布也等价于下面的代码： 

```
y[n] ~ categorical(softmax(x[n] * beta));
```

### 数据声明中的约束项

上述模型的数据模块里没有给`K`、`N`、`D`和`y`加约束条件。数据声明中的约束项帮助在读取数据（或定义数据变换）时检查是否有错误，该流程在整个抽样工作开始之前。数据声明中的约束也使模型作者的意图更加明确，有助于提高可读性。上述模型的声明可以收紧为

```
int<lower = 2> K; 
int<lower = 0> N; 
int<lower = 1> D; 
int<lower = 1, upper = K> y[N]; 
```

之所以出现这些约束，是因为类别数`K`必须至少为两个，才能使分类模型有意义。样本数目`N`可以为零，但不能为负数；与R语言不同，Stan的for循环总是向前移动，因此N为0时，循环范围为1:N，循环体不会执行。预测变量的数量D至少为1，以便`beta * x[n]`为`softmax()`生成适当的参数。分类结果`y[n]`必须介于1和K之间，以便离散采样结果符合定义。

数据声明中的约束是可选的。而在`parameters`部分是必选的——它需要确保所有参数的支撑集满足约束条件。在变换后数据、变换后参数和生成量部分，约束也是可选的。

### 可识别性

因为在向softmax输入值的每个分量添加一个常数时，输出不变，所以模型通常只有系数有合适的先验时才有可识别性。 

另一种方法是使用(K −1)向量，将一个分量固定为零。[部分已知参数](https://mc-stan.org/docs/2_27/stan-users-guide/partially-known-parameters-section.html#partially-known-parameters.section)一节讨论了如何在向量中混合参数和固定常量。在多类Logistic中，参数块中将重定义出(K −1)向量：

```
parameters { 
matrix[K - 1, D] beta_raw; 
}
```

然后它会被变换为参数，在模型中使用。首先，在参数块之前添加一个变换后数据块来定义一个全零行向量，

```
transformed data { 
row_vector[D] zeros = rep_row_vector(0, D); 
}
```

然后接到`beta_row`上来产生系数矩阵`beta`，

```
transformed parameters { 
matrix[K，d] beta; 
beta = append_row(beta_raw, zeros); 
} 
```

命令`rep_row_vector(0,D)` 创建一个长度为`D`的全零行向量。然后将零行向量`zeros`作为一个新行，附在`beta_row`末尾来定义矩阵`beta`；行向量`zeros`定义上属于变换后数据，因此不需要每次使用从头开始构建。

这与使用K向量作为参数的模型不同，因为现在只对(K − 1)向量有先验。在实践中，使用以零为中心的先验时，这将导致最大似然解不同，后验也略有不同。这种先验对回归系数很典型。

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