# 深度学习 (Java 版本 )

该项目是对原始的 [Dive Into Deep Learning ](https://d2l.ai) 一书的修改，这本书是由阿斯顿·张（Aston Zhang），扎卡里·立普顿（Zachary C. Lipton），穆力，亚历克斯·J·斯莫拉（Alex J. Smola）和所有社区贡献者撰写的。  
原始书籍的GitHub地址: [https://github.com/d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en) 。
我们改编了本书以使用Java和[Deep Java Library(DJL)](https://djl.ai)。

这里所有笔记本均可下载并使用Java运行。同时你也可以参阅我们的[网站](https://d2l.djl.ai)。

[DJL community](https://github.com/awslabs/djl) -> DJL(中文)社区。

## 如何在Java中运行Jupyter Notebook

### 线上
您可以通过以下方式在线运行: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aws-samples/d2l-java/master?urlpath=lab)

或者: [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aws-samples/d2l-java/blob/colab/)

### 本地
请按照[此处](https://d2l.djl.ai/chapter_installation/index.html) 的说明操作，以了解如何使用Java运行Notebook。


## 如何为这本书做贡献

请在[此处](documentation/contribute.md)遵循贡献者指南


我们实施了以下章节
* [前言](chapter_preface/index.ipynb)
* [安装](chapter_installation/index.ipynb)
* [符号](chapter_notation/index.ipynb)
* [第一章 简介](chapter_introduction/index.ipynb)
* [第二章 预备知识](chapter_preliminaries/)
* [第三章 线性网络](chapter_linear-networks/)
* [第四章 多层感知器](chapter_multilayer-perceptrons/)
* [第五章 深度学习计算](chapter_deep-learning-computation/)
* [第六章 卷积神经网络](chapter_convolutional-neural-networks/)
* [第七章 现代卷积神经网络](chapter_convolutional-modern/)
* [第八章 优化算法](chapter_optimization/)
* [第九章 计算性能](chapter_computational-performance/)
* [第十章 计算机视觉](chapter_computer-vision/)

## 关于Deep Java Library

[Deep Java Library (DJL)](https://djl.ai) 是用Java编写的深度学习框架，同时支持训练和推理。DJL建立在现代深度学习框架（TenserFlow，PyTorch，MXNet等）之上。您可以轻松地使用DJL训练模型或从各种引擎部署您喜欢的模型，而无需进行任何其他转换。它包含一个强大的ModelZoo设计，使您可以管理训练有素的模型并将其加载到一行中。内置的ModelZoo目前支持来自GluonCV，HuggingFace，TorchHub和Keras的70多种预训练并可以使用的模型。

请关注我们的 [GitHub](https://github.com/awslabs/djl/tree/master/docs), [demo repository](https://github.com/aws-samples/djl-demo), [Slack channel](https://join.slack.com/t/deepjavalibrary/shared_invite/zt-ar91gjkz-qbXhr1l~LFGEIEeGBibT7w) and [twitter](https://twitter.com/deepjavalibrary) ，以获取DJL的更多文档和示例！
