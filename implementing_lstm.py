# -*- coding: utf-8 -*-
#
# 实现LSTM RNN 模型
#------------------------------
#  在这里我们在莎士比亚的作品集上实现LSTM模型
#
#
#

import os  # 这个模块里面的很多命令和cmd命令行类似
import re
import string
import requests
import numpy as np
import collections
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
training_seq_len = 50 # how long of a word group to consider 
embedding_size = rnn_size
save_every = 500 # 保存模型的频率
eval_every = 50 # How often to evaluate the test sentences
prime_texts = ['thou art more', 'to be or not to', 'wherefore art thou']

# 数据存储的目录 以及 模型存储的目录 
data_dir = 'temp'
data_file = 'shakespeare.txt'
model_path = 'shakespeare_model'
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
    # 前7675行 非正文 舍弃
    s_text = s_text[7675:]
    # 把\r 回车符 \n 换行符 换成 空格
    s_text = s_text.replace('\r\n', '')
    s_text = s_text.replace('\n', '')
    
    # 写入文件 "w"
    with open(os.path.join(data_dir, data_file), 'w') as out_conn:
        out_conn.write(s_text)
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

# 制作one-hot词向量
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
        
        # 定义x
        self.x_data = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
        self.y_output = tf.placeholder(tf.int32, [self.batch_size, self.training_seq_len])
        
        with tf.variable_scope('lstm_vars'):
            # Softmax Output Weights
            W = tf.get_variable('W', [self.rnn_size, self.vocab_size], tf.float32, tf.random_normal_initializer())
            b = tf.get_variable('b', [self.vocab_size], tf.float32, tf.constant_initializer(0.0))
        
            # Define Embedding
            embedding_mat = tf.get_variable('embedding_mat', [self.vocab_size, self.embedding_size],
                                            tf.float32, tf.random_normal_initializer())
                                            
            embedding_output = tf.nn.embedding_lookup(embedding_mat, self.x_data)
            rnn_inputs = tf.split(axis=1, num_or_size_splits=self.training_seq_len, value=embedding_output)
            rnn_inputs_trimmed = [tf.squeeze(x, [1]) for x in rnn_inputs]
        
        # If we are inferring (generating text), we add a 'loop' function
        # Define how to get the i+1 th input from the i th output
        def inferred_loop(prev, count):
            # Apply hidden layer
            prev_transformed = tf.matmul(prev, W) + b
            # Get the index of the output (also don't run the gradient)
            prev_symbol = tf.stop_gradient(tf.argmax(prev_transformed, 1))
            # Get embedded vector
            output = tf.nn.embedding_lookup(embedding_mat, prev_symbol)
            return(output)
        
        decoder = tf.contrib.legacy_seq2seq.rnn_decoder
        outputs, last_state = decoder(rnn_inputs_trimmed,
                                      self.initial_state,
                                      self.lstm_cell,
                                      loop_function=inferred_loop if infer_sample else None)
        # Non inferred outputs
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, self.rnn_size])
        # Logits and output
        self.logit_output = tf.matmul(output, W) + b
        self.model_output = tf.nn.softmax(self.logit_output)
        
        loss_fun = tf.contrib.legacy_seq2seq.sequence_loss_by_example
        loss = loss_fun([self.logit_output],[tf.reshape(self.y_output, [-1])],
                [tf.ones([self.batch_size * self.training_seq_len])])
        self.cost = tf.reduce_sum(loss) / (self.batch_size * self.training_seq_len)
        self.final_state = last_state
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tf.trainable_variables()), 4.5)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
        
    def sample(self, sess, words=ix2vocab, vocab=vocab2ix, num=10, prime_text='thou art'):
        state = sess.run(self.lstm_cell.zero_state(1, tf.float32))
        word_list = prime_text.split()
        for word in word_list[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[word]
            feed_dict = {self.x_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed_dict=feed_dict)

        out_sentence = prime_text
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

# Define LSTM Model
lstm_model = LSTM_Model(embedding_size, rnn_size, batch_size, learning_rate,
                        training_seq_len, vocab_size)

# Tell TensorFlow we are reusing the scope for the testing
with tf.variable_scope(tf.get_variable_scope(), reuse=True):
    test_lstm_model = LSTM_Model(embedding_size, rnn_size, batch_size, learning_rate,
                                 training_seq_len, vocab_size, infer_sample=True)


# Create model saver
saver = tf.train.Saver(tf.global_variables())

# Create batches for each epoch
num_batches = int(len(s_text_ix)/(batch_size * training_seq_len)) + 1
# Split up text indices into subarrays, of equal size
batches = np.array_split(s_text_ix, num_batches)
# Reshape each split into [batch_size, training_seq_len]
batches = [np.resize(x, [batch_size, training_seq_len]) for x in batches]

# Initialize all variables
init = tf.global_variables_initializer()
sess.run(init)

# Train model
train_loss = []
iteration_count = 1
for epoch in range(epochs):
    # Shuffle word indices
    random.shuffle(batches)
    # Create targets from shuffled batches
    targets = [np.roll(x, -1, axis=1) for x in batches]
    # Run a through one epoch
    print('Starting Epoch #{} of {}.'.format(epoch+1, epochs))
    # Reset initial LSTM state every epoch
    state = sess.run(lstm_model.initial_state)
    for ix, batch in enumerate(batches):
        training_dict = {lstm_model.x_data: batch, lstm_model.y_output: targets[ix]}
        c, h = lstm_model.initial_state
        training_dict[c] = state.c
        training_dict[h] = state.h
        
        temp_loss, state, _ = sess.run([lstm_model.cost, lstm_model.final_state, lstm_model.train_op],
                                       feed_dict=training_dict)
        train_loss.append(temp_loss)
        
        # Print status every 10 gens
        if iteration_count % 10 == 0:
            summary_nums = (iteration_count, epoch+1, ix+1, num_batches+1, temp_loss)
            print('Iteration: {}, Epoch: {}, Batch: {} out of {}, Loss: {:.2f}'.format(*summary_nums))
        
        # Save the model and the vocab
        if iteration_count % save_every == 0:
            # Save model
            model_file_name = os.path.join(full_model_dir, 'model')
            saver.save(sess, model_file_name, global_step = iteration_count)
            print('Model Saved To: {}'.format(model_file_name))
            # Save vocabulary
            dictionary_file = os.path.join(full_model_dir, 'vocab.pkl')
            with open(dictionary_file, 'wb') as dict_file_conn:
                pickle.dump([vocab2ix, ix2vocab], dict_file_conn)
        
        if iteration_count % eval_every == 0:
            for sample in prime_texts:
                print(test_lstm_model.sample(sess, ix2vocab, vocab2ix, num=10, prime_text=sample))
                
        iteration_count += 1


# Plot loss over time
plt.plot(train_loss, 'k-')
plt.title('Sequence to Sequence Loss')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()
