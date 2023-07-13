#### 卷积网络和全连接结合对视频评分进行拟合demo

- 模型

两层卷积层外加全连接，将输入的16维向量输出成为最终评分，模型比较简单可以任意更改，在valuemodel.py中给出demo

- 数据处理

1. datapreprocess_one_hot 是对数据进行one hot编码

2. datapreprocess_random_vector 是对数据进行random vector编码

3. 数据具体格式参照MovieLens数据集

4. jiebafenci.py中提供对于中文的支持函数。