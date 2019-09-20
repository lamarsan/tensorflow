from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()

# 数据集包含一些未知值。
dataset.isna().sum()

# 为了保持本教程的初始简单，请删除这些行。
dataset = dataset.dropna()

# “origin”列实际上是分类的，而不是数字的。所以把它转换成一个one-hot：
origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
dataset.tail()

'''
现在将数据集拆分为一个训练集和一个测试集。

我们将在模型的最终评估中使用测试集。
'''
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

'''
检查数据

快速查看训练集中几对列的联合分布。
'''
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()

# 看总体统计数据：
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats

'''
从标签拆分要素

将目标值或“标签”与特征分开。这个标签是您将训练模型以预测的值。
'''
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

'''
规范化数据

再看上面的train_stats块，注意每个特性的范围有多大。

规范化使用不同比例和范围的特征是一种很好的做法。尽管该模型可能在没有特征规范化的情况下收敛，但它使训练变得更加困难，并且使生成的模型依赖于输入中使用的单元的选择。
'''


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# 建立模型
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer,
                  metrics=['mean_absolute_error', 'mean_squared_error'])
    return model


model = build_model()

# 观察模型
model.summary()

# 预测10个数据
example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result


# 训练1000次迭代
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()])

# 使用存储在history对象中的统计数据可视化模型的训练进度。
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


plot_history(history)

# 这张图显示了大约100个阶段后验证错误几乎没有改善，甚至没有降低。让我们更新model.fit调用，以便在验证分数没有提高时自动停止培训。我们将使用EarlyStopping回调来测试每个时代的训练条件。如果一个设定的时间段过去了而没有表现出改善，那么自动停止训练。
model = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

'''
图表显示，在验证集上，平均误差通常约为+/-2 mpg。这样好吗？我们把决定权留给你。

让我们看看通过使用测试集（我们在训练模型时没有使用测试集），模型的泛化程度如何。这告诉我们，当我们在现实世界中使用该模型时，我们可以很好地预测它。
'''
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

# 预测
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

# 训练结果还不错，让我们看看错误分布
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()

'''
均方误差（mse）是一种常见的用于回归问题的损失函数（不同的损失函数用于分类问题）。

同样，用于回归的评估指标与分类不同。常见的回归指标是平均绝对误差（mae）。

当数字输入数据特征具有不同范围的值时，每个特征应独立地缩放到相同范围。

如果训练数据不多，一种方法是选择一个隐藏层较少的小网络，以避免过度拟合。

早期停车是防止过度训练的有效技术。
'''
