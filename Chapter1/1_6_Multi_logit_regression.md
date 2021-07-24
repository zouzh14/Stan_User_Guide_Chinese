## 1.6 多类逻辑蒂克回归

多类输出情况下的逻辑蒂克回归可以直接在Stan中进行。记对每个输出变量$y_n$都有$K$个可能结果，它对应的$x_n$是一个D维向量。系数先验为$normal(0,5)$的多类逻辑蒂克回归代码如下。

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

另一种方法是使用(K −1)向量，将一个分量固定为零。[部分已知参数](https://mc-stan.org/docs/2_27/stan-users-guide/partially-known-parameters-section.html#partially-known-parameters.section)一节讨论了如何在向量中混合参数和固定常量。在多类逻辑蒂克中，参数块中将重定义出(K −1)向量：

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
