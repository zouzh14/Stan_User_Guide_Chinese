## 1.3 系数和数值范围的先验

对回归参数先验分布，请参阅我们[对先验的一般讨论](https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations)一文。 

后面的章节会讨论[单变量分层先验](https://mc-stan.org/docs/2_27/stan-users-guide/hierarchical-priors-section.html#hierarchical-priors.section)和[多变量分层先验](https://mc-stan.org/docs/2_27/stan-users-guide/multivariate-hierarchical-priors-section.html#multivariate-hierarchical-priors.section)，以及[用于辨识模型的先验](https://mc-stan.org/docs/2_27/stan-users-guide/priors-for-identification-section.html#priors-for-identification.section)。 

但是，如[QR重参数化](https://mc-stan.org/docs/2_27/stan-users-guide/QR-reparameterization-section.html#QR-reparameterization.section)章节所述，如果没有回归系数location的有信息量的先验认识，最好重参数化模型，以便回归系数是生成的数值。在这种情况下，重参数化的回归系数上使用什么先验通常并不重要，几乎任何和输出值数值范围相似的弱先验都可以。 
