import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def preprocess_data(data_path, test_size=0.2, random_state=42):

    data = np.loadtxt(data_path, delimiter='::', dtype=np.int32)

    # 分割为特征和标签
    user_ids = data[:, 0]
    movie_ids = data[:, 1]
    ratings = data[:, 2]

    # 将用户ID和电影ID转换为One-Hot编码
    num_users = np.max(user_ids) + 1
    num_movies = np.max(movie_ids) + 1
    user_one_hot = to_categorical(user_ids, num_classes=num_users)
    movie_one_hot = to_categorical(movie_ids, num_classes=num_movies)


    ratings = np.clip(ratings, 1, 5)


    features = np.concatenate((user_one_hot, movie_one_hot), axis=1)

    # 将标签转换为One-Hot编码
    labels = to_categorical(ratings - 1, num_classes=5)

    # 划分训练集和测试集
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state)

    # 调整输入形状为适应卷积神经网络的要求
    features_train = np.expand_dims(features_train, axis=-1)
    features_test = np.expand_dims(features_test, axis=-1)

    return features_train, features_test, labels_train, labels_test