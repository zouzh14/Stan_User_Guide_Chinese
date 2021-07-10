# 概述
## 关于本用户指南

这是Stan的官方用户指南。它提供了在Stan中编码统计模型的示例模型和编程技术。

+ 第一部分给出了几类重要模型的Stan代码和相关讨论。

+ 第二部分讨论了多种一般性Stan编程技术，这些技术并不与任何特定的模型相联系。

+ 第三部分介绍了校准和模型检查的算法，它们需要多次运行Stan。

+ 附录为BUGS和JAGS用户提供了指南和建议。

除了本用户指南外，还有两本Stan语言和算法的参考手册。[*Stan Reference Manual*](https://mc-stan.org/docs/2_27/reference-manual/index.html)详述了Stan编程语言和推断算法。[*Stan Function Reference*](https://mc-stan.org/docs/2_27/functions-reference/index.html)详述了Stan编程语言中内置的函数。

还有一份单独的安装和入门指南，适用于各Stan调用接口（R、Python、Julia、Stata、MATLAB、Mathematica和命令行）。

我们建议借助*Bayesian Data Analysis*和*Bayesian Data Analysis and Statistical Rethinking: A Bayesian Course with Examples in R and Stan*这两本教科书来学习本指南，必要时使用[*Stan Reference Manual*](https://mc-stan.org/docs/2_27/reference-manual/index.html)来阐明编程问题。

## 网络资源

Stan是一个开源软件项目，它的资源托管在不同的网站上。

+ [Stan网站](https://mc-stan.org/)为用户和开发者组织了Stan项目的所有资源。它包含了Stan官方发布的链接、源代码、安装说明和完整的文档，包括本手册的最新版本、每个界面的用户指南和入门指南、教程、案例研究和开发人员的参考材料。

+ [Stan论坛](https://discourse.mc-stan.org/)为用户和开发者提供了成体系的留言板，用于与Stan相关的问题、讨论和公告。

+ [Stan GitHub](https://github.com/stan-dev)承载了Stan的所有代码、文档、维基和网站，错误报告和功能请求，交互的代码审查。

## 鸣谢
没有开发人员、用户和资助，Stan项目就不可能存在。Stan是一个高度协作的项目。Stan开发者对编程的贡献通过GitHub来记录，对设计的共享通过维基和论坛。

用户以案例研究、教程、甚至书籍的方式对文档做出了广泛的贡献。他们还报告了代码和文档中的许多错误。

Stan获得的资助包括对Stan及其开发者的拨款、公司和个人贡献出的开发时间，以及对开源科学软件非营利组织NumFOCUS的捐款。关于项目的资金的细节，见Stan开发者的网站和项目页面。

## 版权、商标和许可

This book is copyright 2011–2019, Stan Development Team and their assignees.

文本内容在CC-BY ND 4.0许可下发布。用户指南R和Stan程序在BSD 3-clause许可证下分发。

Stan的名称和标识是NumFOCUS的注册商标。Stan名称和标识的使用受[Stan标识使用指南](https://mc-stan.org/about/logo/)的约束。