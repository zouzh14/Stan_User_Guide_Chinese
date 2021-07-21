# 1.4 稳健的噪声模型 

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