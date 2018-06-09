# 在tensorflow中实现一个RNN
#----------------------------------
#
# 通过RNN实现识别垃圾邮件/非垃圾邮件的问题
#

# 首先从载入必要的包开始:
import os
import re
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from zipfile import ZipFile  # 读取zip文件
from tensorflow.python.framework import ops

# 初始化默认图
ops.reset_default_graph()

# 启动一个会话
sess = tf.Session()

# 设置RNN参数
epochs = 20
batch_size = 250
# 超过25长度的序列将会被剪短到25，不足25的序列会做zero padding
max_sequence_length = 25
rnn_size = 10
# 嵌入维度比较小，方便计算
embedding_size = 50
min_word_frequency = 10
learning_rate = 0.0005
# 在学习过程中dropout比例控制在0.5，在evaluation过程中dropout比例控制成1
dropout_keep_prob = tf.placeholder(tf.float32)


# 下载或者打开数据
data_dir = 'temp'
data_file = 'text_data.txt'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 如果目标文件不存在
if not os.path.isfile(os.path.join(data_dir, data_file)):
    # 给出爬取路径
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    # 通过爬虫爬取url链接, 可以看到目标文件是.zip文件
    r = requests.get(zip_url)    
    # 用io读取爬取到的内容, zipfile默认是读取
    z = ZipFile(io.BytesIO(r.content))
    # 定义read文件为file
    file = z.read('SMSSpamCollection')
    # 格式化数据
    text_data = file.decode()  # 先解码
    text_data = text_data.encode('ascii',errors='ignore')  # 然后编码成ASCII
    text_data = text_data.decode().split('\n')  # 解码并且用换行符分割

    # 将爬取到并且格式化的数据写入, 并且用换行符隔开
    with open(os.path.join(data_dir, data_file), 'w') as file_conn:
        for text in text_data:
            file_conn.write("{}\n".format(text))
# 如果存在
else:
    # 读取数据, 先定义一个空的列表
    text_data = []
    with open(os.path.join(data_dir, data_file), 'r') as file_conn:
        for row in file_conn:
            text_data.append(row)
    text_data = text_data[:-1] # 切片到倒数第一个, 且不包含倒数第一个, 最后一个是标签

# 用制表符分开
text_data = [x.split('\t') for x in text_data if len(x)>=1]
# zip是压缩, zip(*)是解压缩 
# zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
# 如果各个迭代器的元素个数不一致，则返回列表长度与最短的对象相同，利用 * 号操作符，可以将元组解压为列表。
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]


# 定义一个清洗数据的函数 去掉数字 下划线 和不可见符号 全部小写
def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return(text_string)

# 清洗训练数据
text_data_train = [clean_text(x) for x in text_data_train]

# 把文本转化成数值型的向量 词汇表模型 低于最小词频的词不会被收录到词汇表中
# 首先定义processor 然后fit_transform数据 最后用np.array转换类型
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length,
                                                                     min_frequency=min_word_frequency)
text_processed = np.array(list(vocab_processor.fit_transform(text_data_train)))

# 随机打乱 并且分割数据
# 先把训练数据变成ndarray
text_processed = np.array(text_processed)
# 把标签转化成数值型, 非垃圾邮件是1, 其余情况是0
text_data_target = np.array([1 if x=='ham' else 0 for x in text_data_target])  # 制作标签
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))  # 制作一个长度相同但顺序打乱的表
x_shuffled = text_processed[shuffled_ix]  # 利用打乱的表制作训练集x
y_shuffled = text_data_target[shuffled_ix]  # 利用打乱的表制作训练集y

# 划分训练集和测试集
ix_cutoff = int(len(y_shuffled)*0.80)  # 确定划分点
# 按照划分点分割训练集和验证集
x_train, x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
y_train, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
# 词汇表容量
vocab_size = len(vocab_processor.vocabulary_)
print("Vocabulary Size: {:d}".format(vocab_size))
print("80-20 Train Test split: {:d} -- {:d}".format(len(y_train), len(y_test)))

# 创建占位符 x是batch_size * 序列长度 y是batch_size
x_data = tf.placeholder(tf.int32, [None, max_sequence_length])
y_output = tf.placeholder(tf.int32, [None])

# 创建嵌入矩阵, 首先是初始化, 由于这是Variable, 在训练过程中会不断优化
embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))  # 参数分别是最小值和最大值
# 把输入的文件转化成词嵌入的形式
embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data)
#embedding_output_expanded = tf.expand_dims(embedding_output, -1)  #可以给output增加一个维度 -1表示最后一个维度

# 定义RNN细胞
# 大于1.0版本的tensorflow已经把RNN放在tensorflow.contrib里面了, 小于的版本在tf.nn
if tf.__version__[0]>='1':
    cell=tf.contrib.rnn.BasicRNNCell(num_units = rnn_size)
else:
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units = rnn_size)

# rnn_cell 会输出两个, 一个是状态一个没有经过softmax的输出, embedding_output作为输入
output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)
# 随机丢弃的比例, 比例之前已经设置为一个placeholder
output = tf.nn.dropout(output, dropout_keep_prob)

# 获得RNN序列的输出
output = tf.transpose(output, [1, 0, 2])  # 把第一维度batch_size和第二维度time_step互换
# 定义一个last, 选择的是output转化维度后的第一个维度, 也就是time_step的最后一个
last = tf.gather(output, int(output.get_shape()[0]) - 1)  # tf.gather 可以对张量进行切片 get_shape()对象是tensor, 返回是tuple

# 设置权重, 这里的2是分成两类的意思
weight = tf.Variable(tf.truncated_normal([rnn_size, 2], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[2]))
# 计算最后的输出
logits_out = tf.matmul(last, weight) + bias

# 损失函数, 注意这里的sparse版本和softmax是一样的, 区别在于这里的label是一个整数, softmax那里是一个one_hot的向量
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=y_output) # logits=float32, labels=int32
# 计算平均的损失
loss = tf.reduce_mean(losses)

# 计算准确率
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out, 1), tf.cast(y_output, tf.int64)), tf.float32))

# 定义优化器和训练步
optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

# 开始训练
init = tf.global_variables_initializer()
sess.run(init)

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

# 开始训练
for epoch in range(epochs):

    # 随机打乱顺序
    shuffled_ix = np.random.permutation(np.arange(len(x_train)))
    
    x_train = x_train[shuffled_ix]
    y_train = y_train[shuffled_ix]
    num_batches = int(len(x_train)/batch_size) + 1   # 注意这里取整后+1 数据会有多
    # 生成minibatch
    for i in range(num_batches):
        # 选择训练数据
        min_ix = i * batch_size
        max_ix = np.min([len(x_train), ((i+1) * batch_size)])  # 最后一个minibatch上限取len() 
        x_train_batch = x_train[min_ix:max_ix]
        y_train_batch = y_train[min_ix:max_ix]
        
        # sess.run 注意dropoutkeepprob是放在train_dict里面的
        train_dict = {x_data: x_train_batch, y_output: y_train_batch, dropout_keep_prob:0.5}
        sess.run(train_step, feed_dict=train_dict)
        
    # 在运行时获得损失和准确率
    temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
    train_loss.append(temp_train_loss)
    train_accuracy.append(temp_train_acc)
    
    # test dict里面dopout比例是1
    test_dict = {x_data: x_test, y_output: y_test, dropout_keep_prob:1.0}
    temp_test_loss, temp_test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
    test_loss.append(temp_test_loss)
    test_accuracy.append(temp_test_acc)
    print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch+1, temp_test_loss, temp_test_acc))
    
# 画两张图分别是train loss和test loss
epoch_seq = np.arange(1, epochs+1)
plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
plt.plot(epoch_seq, test_loss, 'r-', label='Test Set')
plt.title('Softmax Loss')
plt.xlabel('Epochs')
plt.ylabel('Softmax Loss')
plt.legend(loc='upper left')
plt.show()

# 画两张图分别是train accuracy和test accuracy
plt.plot(epoch_seq, train_accuracy, 'k--', label='Train Set')
plt.plot(epoch_seq, test_accuracy, 'r-', label='Test Set')
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()
