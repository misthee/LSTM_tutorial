# -*- coding: utf-8 -*-
#
# 实现LSTM RNN 模型
#------------------------------
#  在这里我们在莎士比亚的作品集上实现LSTM模型
#
#
#

import os  # os模块主要用来连接路径, 许多命令类似命令行命令
import re  # 正则表达式模块
import string
import requests  # 爬虫模块
import numpy as np
import collections  # 统计词频模块
import random
import pickle  
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

# 清除默认图的堆栈,并设置全局图为默认图. 类似于tensorflow版本的rm(list = ls())
ops.reset_default_graph()

# 新建一个会话
sess = tf.Session()

# 设置RNN参数
min_word_freq = 5 # 出现次数这个值的词将会被忽略
rnn_size = 128 # 每一个Cell里面包含的神经元的个数
embedding_size = 100 # 词嵌入的维数
epochs = 10
batch_size = 100 
learning_rate = 0.001
training_seq_len = 50 # 训练的句子的长度
embedding_size = rnn_size
save_every = 500 # 保存模型到ckpt的频率
eval_every = 50 # 评价测试数据的频率 sess.run用来训练 eval用来测试和验证
# 定义文本生成的起始句
prime_texts = ['thou art more', 'to be or not to', 'wherefore art thou']

# 数据存储的目录 以及 模型存储的目录 
data_dir = 'temp'
data_file = 'shakespeare.txt'
model_path = 'shakespeare_model'
# 模型的目录是放在数据的目录之下
full_model_dir = os.path.join(data_dir, model_path)  #path.join 可以做一个更长的目录，此处为"../temp/shakespeare_model"

# 把除开连字符和撇号以外的所有标点都去掉
punctuation = string.punctuation  # string.punctuation 可以得到所有的标点
punctuation = ''.join([x for x in punctuation if x not in ['-', "'"]])  # 用空格链接所有(除了-和')的标点

# 生成模型的目录
if not os.path.exists(full_model_dir):  # os.path.exists 返回的结果是 bool 型，如果没有就生成一个
    os.makedirs(full_model_dir) 

# 生成数据的目录
if not os.path.exists(data_dir):  # 注意是 exists 和 makedirs 都有s
    os.makedirs(data_dir)

# 开始载入莎士比亚的数据
print('Loading Shakespeare Data')  
# 确认文件是否存在
if not os.path.isfile(os.path.join(data_dir, data_file)):  # isfile 函数, 函数中的变量是包括了文件名的路径
    print('Not found, downloading Shakespeare texts from www.gutenberg.org')  # 如果没有 就说没有
    shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'  # 给出下载链接
    # 获取莎士比亚的数据
    response = requests.get(shakespeare_url)  # 用request.get 爬取链接
    shakespeare_file = response.content  # response.content
    # 爬取到的为计算机可读的binary文件. 需要解码成utf-8. utf-8编码可以显示简繁中英日韩. 比较通用.
    s_text = shakespeare_file.decode('utf-8')
    # 前7675个词 非正文 舍弃
    s_text = s_text[7675:]
    # 把\r 回车符 \n 换行符 换成 空格
    s_text = s_text.replace('\r\n', '')
    s_text = s_text.replace('\n', '')
    
    # 把s_text 写成文件 shakespear.txt
    with open(os.path.join(data_dir, data_file), 'w') as out_conn:
        out_conn.write(s_text)
# 如果存在
else:
    # 读取文件 "r"
    with open(os.path.join(data_dir, data_file), 'r') as file_conn:
        s_text = file_conn.read().replace('\n', '')

# 开始清理文本
print('Cleaning Text')
s_text = re.sub(r'[{}]'.format(punctuation), ' ', s_text)  # re.sub 利用正则表达式, 把标点换成空格
s_text = re.sub('\s+', ' ', s_text ).strip().lower()  # 把任意空白字符  \f\n\r\t\v 都替换为空格 

# 定义一个建立词汇表的 函数
def build_vocab(text, min_word_freq):
    # collection.Counter 可以统计词频 由于之前标点都已经换成了空格 直接用空格分开即可 返回结果是一个字典
    word_counts = collections.Counter(text.split(' '))  
    # 做成一个字典 索引是词 key是词 val是频次 加入最低频次限制
    word_counts = {key:val for key, val in word_counts.items() if val>min_word_freq} 
    # 把key提取成list
    words = word_counts.keys()
    # enumerate 方法可以提取list中的index key是索引 val是list元素 重新做一个字典 索引从1开始 所以此处key是词 val是index
    vocab_to_ix_dict = {key:(ix+1) for ix, key in enumerate(words)}
    # 索引为0的未登录词
    vocab_to_ix_dict['unknown']=0
    # 将上面的dict的key val互换 key是index val是词
    ix_to_vocab_dict = {val:key for key,val in vocab_to_ix_dict.items()}
    
    return(ix_to_vocab_dict, vocab_to_ix_dict)

# 建立莎士比亚词汇表
print('Building Shakespeare Vocab')
# 用上文的函数获得两个词典
ix2vocab, vocab2ix = build_vocab(s_text, min_word_freq)
# 字典长度 ? 为什么 +1
vocab_size = len(ix2vocab) + 1
print('Vocabulary Length = {}'.format(vocab_size))
# 合理性检查
# assert 如果后面的语句为False 会报错 Assertation Error
assert(len(ix2vocab) == len(vocab2ix))

# 把文本中的每个词汇换成它在词典中的索引
s_text_words = s_text.split(' ')
# 先做成一个空的list
s_text_ix = []
for ix, x in enumerate(s_text_words):
    # try except 句式 首先尝试 其次尝试
    try:
        s_text_ix.append(vocab2ix[x])  # 添加上面的词对应的索引
    except:
        s_text_ix.append(0)  # 如果词没有索引, 就添加0, 即记为未登录词
s_text_ix = np.array(s_text_ix)  # 将list变换成ndarray



# 定义LSTM 模型
class LSTM_Model():
    # 定义init方法
    def __init__(self, embedding_size, rnn_size, batch_size, learning_rate,
                 training_seq_len, vocab_size, infer_sample=False):
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.vocab_size = vocab_size
        self.infer_sample = infer_sample
        self.learning_rate = learning_rate
        
        if infer_sample:
            self.batch_size = 1
            self.training_seq_len = 1
        else:
            self.batch_size = batch_size
            self.training_seq_len = training_seq_len
        
        # 定义LSTM细胞 参数是rnn_size
        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
        # 定义初始状态
        self.initial_state = self.lstm_cell.zero_state(self.batch_size, tf.float32)
        
        # 定义x和y的占位符
        self.x_data = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
        self.y_output = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
        
        # 定义变量, scope相当于定义一个文件夹上级目录, 方便参数共享
        with tf.variable_scope('lstm_vars'):
            # 为Softmax输出设置变量，列数是词汇表的大小
            W = tf.get_variable('W', [self.rnn_size, self.vocab_size], tf.float32, tf.random_normal_initializer())
            # b的维度是词汇表的大小
            b = tf.get_variable('b', [self.vocab_size], tf.float32, tf.constant_initializer(0.0))
        
            # 设置词嵌入, 注意是variable
            embedding_mat = tf.get_variable('embedding_mat', [self.vocab_size, self.embedding_size],
                                            tf.float32, tf.random_normal_initializer())
            
            # tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。tf.nn.embedding_lookup（tensor, id）
            embedding_output = tf.nn.embedding_lookup(embedding_mat, self.x_data)
            # tf.split(dimension, num_split, input)：dimension的意思就是输入张量的哪一个维度，如果是0就表示对第0维度进行切割。num_split就是切割的数量，如果是2就表示输入张量被切成2份，每一份是一个列表。
            rnn_inputs = tf.split(axis=1, num_or_size_splits=self.training_seq_len, value=embedding_output)
            # tf.squeeze 去掉维数为1的维度
            rnn_inputs_trimmed = [tf.squeeze(x, [1]) for x in rnn_inputs]
        
        # 如果我们要推断(生成文本),  我们设置一个loop函数
        # 确定怎么样从第i次的输入产生i+1次的输入
        def inferred_loop(prev, count):
            # 前一层的输出和W矩阵相乘, 再加上b
            prev_transformed = tf.matmul(prev, W) + b
            # argmax沿着一号轴线也就是列方向找最大值, stop_gradient可以不求某个参数的倒数
            prev_symbol = tf.stop_gradient(tf.argmax(prev_transformed, 1))
            # 得到嵌入的向量
            output = tf.nn.embedding_lookup(embedding_mat, prev_symbol)
            return(output)
        
        decoder = tf.contrib.legacy_seq2seq.rnn_decoder
        # 得到推断的文本
        outputs, last_state = decoder(rnn_inputs_trimmed,
                                      self.initial_state,
                                      self.lstm_cell,
                                      loop_function=inferred_loop if infer_sample else None)
        # 把结果拼接起来, axis = 1 是横向的
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.rnn_size])
        # 得到RNN
        self.logit_output = tf.matmul(output, W) + b
        self.model_output = tf.nn.softmax(self.logit_output)
        
        # 这个损失函数相当于是在序列上运用cross_entropy
        loss_fun = tf.contrib.legacy_seq2seq.sequence_loss_by_example
        # tf.ones 是全1
        loss = loss_fun([self.logit_output],[tf.reshape(self.y_output, [-1])],[tf.ones([self.batch_size * self.training_seq_len])])
        self.cost = tf.reduce_sum(loss) / (self.batch_size * self.training_seq_len)
        self.final_state = last_state
        # 为了防止梯度爆炸或者梯度消失, 对全局梯度进行限制
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()), 4.5)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # 把梯度应用到变量上去
        self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
        
    
    # 定义一个初始采样的函数
    def sample(self, sess, words=ix2vocab, vocab=vocab2ix, num=10, prime_text='thou art'):
        # initial_state = LSTMStateTuple(c_state, h_state) state
        # 定义初始0状态
        state = sess.run(self.lstm_cell.zero_state(1, tf.float32))
        # word list是初始语句分割的结果
        word_list = prime_text.split()
        for word in word_list[:-1]:
            # 初始化x
            x = np.zeros((1, 1))
            # 用word替换x中的元素
            x[0, 0] = vocab[word]
            feed_dict = {self.x_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed_dict=feed_dict)

        out_sentence = prime_text
        # 出来
        word = word_list[-1]
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[word]
            feed_dict = {self.x_data: x, self.initial_state:state}
            [model_output, state] = sess.run([self.model_output, self.final_state], feed_dict=feed_dict)
            sample = np.argmax(model_output[0])
            if sample == 0:
                break
            word = words[sample]
            out_sentence = out_sentence + ' ' + word
        return(out_sentence)

# 定义LSTM模型
lstm_model = LSTM_Model(embedding_size, rnn_size, batch_size, learning_rate,
                        training_seq_len, vocab_size)

# 重新使用之前的variable scope, 也就是变量的集合, 用于测试
with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    test_lstm_model = LSTM_Model(embedding_size, rnn_size, batch_size, learning_rate,
                                 training_seq_len, vocab_size, infer_sample=True)


# 保存参数
saver = tf.train.Saver(tf.global_variables())

# 制作minibatch
num_batches = int(len(s_text_ix)/(batch_size * training_seq_len)) + 1
# 按照batch的数量分割
batches = np.array_split(s_text_ix, num_batches)
# 一维变成两维, 按照长度和batch_size来划分
batches = [np.resize(x, [batch_size, training_seq_len]) for x in batches]

# 初始化所有变量
init = tf.global_variables_initializer()
sess.run(init)

# 训练模型
train_loss = []
iteration_count = 1
for epoch in range(epochs):
    # 随机打乱batch的索引
    random.shuffle(batches)
    # np.roll 把x沿着1轴线滚动-1长度 因为我们在第i个位置要训练的target实际上就是i+1个词
    targets = [np.roll(x, -1, axis=1) for x in batches]
    # 训练一个epoch
    print('Starting Epoch #{} of {}.'.format(epoch+1, epochs))
    # 每个epoch之前都重置lstm的初始状态
    state = sess.run(lstm_model.initial_state)
    
    for ix, batch in enumerate(batches):
        # 定义训练用的数据词典
        training_dict = {lstm_model.x_data: batch, lstm_model.y_output: targets[ix]}
        # 定义两个state
        c, h = lstm_model.initial_state
        training_dict[c] = state.c
        training_dict[h] = state.h
        
        temp_loss, state, _ = sess.run([lstm_model.cost, lstm_model.final_state, lstm_model.train_op],
                                       feed_dict=training_dict)
        train_loss.append(temp_loss)
        
        # 每10次生成打印一次
        if iteration_count % 10 == 0:
            summary_nums = (iteration_count, epoch+1, ix+1, num_batches+1, temp_loss)
            print('Iteration: {}, Epoch: {}, Batch: {} out of {}, Loss: {:.2f}'.format(*summary_nums))
        
        # 保存模型
        if iteration_count % save_every == 0:
            # 在目标路径下保存为文件名model
            model_file_name = os.path.join(full_model_dir, 'model')
            # 保存sess
            saver.save(sess, model_file_name, global_step = iteration_count)
            print('Model Saved To: {}'.format(model_file_name))
            # 把词汇到索引 索引到词典都保存起来
            dictionary_file = os.path.join(full_model_dir, 'vocab.pkl')
            with open(dictionary_file, 'wb') as dict_file_conn:
                pickle.dump([vocab2ix, ix2vocab], dict_file_conn)
        
        # 测试
        if iteration_count % eval_every == 0:
            for sample in prime_texts:
                print(test_lstm_model.sample(sess, ix2vocab, vocab2ix, num=10, prime_text=sample))
                
        iteration_count += 1


# 画图
plt.plot(train_loss, 'k-')
plt.title('Sequence to Sequence Loss')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
