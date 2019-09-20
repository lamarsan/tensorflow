from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

tf.__version__

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# 定义模型
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


# Create a basic model instance
model = create_model()
model.summary()

# 检查点回调用法
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()

model.fit(train_images, train_labels, epochs=10,
          validation_data=(test_images, test_labels),
          callbacks=[cp_callback])

# 上述代码将创建一个 TensorFlow 检查点文件集合，这些文件在每个周期结束时更新：

'''
创建一个未经训练的全新模型。仅通过权重恢复模型时，您必须有一个与原始模型架构相同的模型。由于模型架构相同，因此我们可以分享权重（尽管是不同的模型实例）。

现在，重新构建一个未经训练的全新模型，并用测试集对其进行评估。未训练模型的表现有很大的偶然性（准确率约为 10%）：
'''

model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))

# 然后从检查点加载权重，并重新评估：
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

'''
该回调提供了多个选项，用于为生成的检查点提供独一无二的名称，以及调整检查点创建频率。

训练一个新模型，每隔 5 个周期保存一次检查点并设置唯一名称：
'''
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

model = create_model()
model.fit(train_images, train_labels,
          epochs=50, callbacks=[cp_callback],
          validation_data=(test_images, test_labels),
          verbose=0)

# 要进行测试，请重置模型并加载最新的检查点：
latest = tf.train.latest_checkpoint(checkpoint_dir)
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

'''
上述代码将权重存储在检查点格式的文件集合中，这些文件仅包含经过训练的权重（采用二进制格式）。检查点包括： * 包含模型权重的一个或多个分片。 * 指示哪些权重存储在哪些分片中的索引文件。

如果您仅在一台机器上训练模型，则您将有 1 个后缀为 .data-00000-of-00001 的分片。

在上文中，您了解了如何将权重加载到模型中。

手动保存权重的方法同样也很简单，只需使用 Model.save_weights 方法即可。
'''
# Save the weights
model.save_weights('checkpoints/my_checkpoint')

# Restore the weights
model = create_model()
model.load_weights('checkpoints/my_checkpoint')

loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

'''
保存整个模型
整个模型可以保存到一个文件中，其中包含权重值、模型配置乃至优化器配置。这样，您就可以为模型设置检查点，并稍后从完全相同的状态继续训练，而无需访问原始代码。

在 Keras 中保存完全可正常使用的模型非常有用，您可以在 TensorFlow.js 中加载它们，然后在网络浏览器中训练和运行它们。

Keras 使用 HDF5 标准提供基本的保存格式。对于我们来说，可将保存的模型视为一个二进制 blob。
'''
model = create_model()

model.fit(train_images, train_labels, epochs=5)

# Save entire model to a HDF5 file
model.save('my_model.h5')

# 重建
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

# 检查准确率
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

'''
此技巧可保存以下所有内容：

权重值
模型配置（架构）
优化器配置

Keras 通过检查架构来保存模型。目前，它无法保存 TensorFlow 
优化器（来自 tf.train）。使用此类优化器时，您需要在加载模型
后对其进行重新编译，使优化器的状态变松散。
'''
