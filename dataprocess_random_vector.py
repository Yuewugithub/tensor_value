import numpy as np
from sklearn.model_selection import train_test_split


def preprocess_data(data_path, embedding_dim=16, test_size=0.2, random_state=42):

    data = np.loadtxt(data_path, delimiter='::', dtype=np.int32)

    # 分割为特征和标签
    user_ids = data[:, 0]
    movie_ids = data[:, 1]
    ratings = data[:, 2]

    # 创建用户和电影的嵌入矩阵
    num_users = np.max(user_ids) + 1
    num_movies = np.max(movie_ids) + 1
    user_embeddings = np.random.randn(num_users, embedding_dim)
    movie_embeddings = np.random.randn(num_movies, embedding_dim)


    #随机化范围（0,1）
    # user_embeddings = np.random.randn(num_users, embedding_dim)
    # movie_embeddings = np.random.randn(num_movies, embedding_dim)


    # 获取用户和视频的嵌入向量
    user_features = user_embeddings[user_ids]
    movie_features = movie_embeddings[movie_ids]

    # 将评分转换为范围在1到5之间的整数
    ratings = np.clip(ratings, 1, 5)

    # 将特征组合成输入向量
    features = np.concatenate((user_features, movie_features), axis=1)

    # 将标签转换为范围在0到4之间的整数
    labels = ratings - 1

    # 划分训练集和测试集
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state)

    # 调整输入形状为适应卷积神经网络的要求
    features_train = np.expand_dims(features_train, axis=-1)
    features_test = np.expand_dims(features_test, axis=-1)

    return features_train, features_test, labels_train, labels_test