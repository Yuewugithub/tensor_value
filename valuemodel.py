import tensorflow as tf
from tensorflow.keras import layers


def create_model():
    model = tf.keras.Sequential()

    # 第一个卷积层
    model.add(layers.Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(16, 1)))
    model.add(layers.MaxPooling1D(pool_size=2))

    # 第二个卷积层
    #model.add(layers.Conv1D(filters=32, kernel_size=3, activation='relu'))
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation='tanh'))
    model.add(layers.MaxPooling1D(pool_size=2))

    # 展平层
    model.add(layers.Flatten())

    # 全连接层
    model.add(layers.Dense(64, activation='relu'))
    #model.add(layers.Dense(1, activation='tanh'))

    # 输出层
    #model.add(layers.Dense(1, activation='sigmoid'))
    model.add(layers.Dense(1, activation='linear'))

    return model

model = create_model()

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_data, train_labels, epochs=10, batch_size=32)