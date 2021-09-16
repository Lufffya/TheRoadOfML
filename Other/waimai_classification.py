#
# 外卖好评差评分类
#

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split


n_sample = 10000
max_seq_len = 128
data = pd.read_csv(r"DataSet\\外卖评价.csv")
data = data.sample(n=n_sample)
data = data.loc[[True if len(item)<=max_seq_len else False for item in data["review"]]]
print(data.info())

# 构建词库
vocab = []
for review in data["review"]:
    vocab.extend(review)
vocab = list(set(vocab))

# tokenize
inputs, labels = [], []
for review in data["review"]:
    token_ids = [vocab.index(item) + 1 for item in review]
    if len(token_ids) < max_seq_len:
        token_ids = token_ids + [0] * (max_seq_len - len(token_ids))
    inputs.append(token_ids)
    
inputs = np.array(inputs)
labels = np.array(data["label"])

x_train, x_test, y_train, y_test = train_test_split(inputs, labels)

# print(len(vocab)+1)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=len(vocab)+1, output_dim=256, input_length=max_seq_len))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(64, activation="relu"))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(2, activation="softmax"))
model.compile(optimizer=tf.optimizers.Adam(), loss="sparse_categorical_crossentropy", metrics=["acc"])
model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)


def classification(n_show=20):
    pre = model.predict(x_test)
    indices = np.random.permutation(len(x_test))[:n_show]
    for index in indices:
        argmax = np.argmax(pre[index])
        review = ""
        for item in x_test[index]:
            if item == 0 : continue
            review += vocab[(item - 1)]
        print("softmax：{0} \t 预测: {1} \t 标签: {2} \t 评论: {3}".format(pre[index], argmax, y_test[index], review))


def semantic_search(target_label=None):
    def encoding(_inputs):
        inputs_embedding = []
        embedding_layer = model.layers[0]
        flatten = tf.keras.layers.GlobalAveragePooling1D()
        for i in range(0, len(_inputs), 128):
            batch_size = _inputs[i:i+128]
            # 传入embedding层进行处理，输出词向量，即词在embedding网络层的多维空间中的位置
            batch_word_embedding = flatten(embedding_layer(batch_size))
            inputs_embedding.extend(batch_word_embedding.numpy())
        return np.array(inputs_embedding)

    def get_review_by_token_ids(token_ids):
        review = ""
        for item in token_ids:
            if item == 0 : continue
            review += vocab[(item - 1)]
        return review
    
    if target_label == None:
        _index = np.random.permutation(len(inputs))[:1]
        x_target = inputs[_index]
        print("====== "+get_review_by_token_ids(x_target.flatten())+"======")
    else:
        token_ids = [vocab.index(item)+1 for item in target_label]
        token_ids = token_ids + [0] * (max_seq_len - len(token_ids))
        x_target = np.expand_dims(token_ids, axis=0)
        print("====== "+target_label+"======")

    # 获取当前测试句子的词向量
    x_target_embedding = encoding(x_target)

    # 获取所有训练句子的词向量
    inputs_embedding = encoding(inputs)

    # 该函数用于计算两个输入集合的距离,默认metric='euclidean'表示计算欧式距离
    # 返回两元素之间的距离
    euclidean_Distance = cdist(x_target_embedding, inputs_embedding)[0]

    # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
    results = zip(range(len(euclidean_Distance)), euclidean_Distance)

    # sorted函数对所有可迭代的对象进行排序操作
    # param1：可迭代对象：results
    # param2：定可迭代对象中的一个元素来进行排序
    # param3：reverse -- 排序规则，reverse = True 降序 ， reverse = False 升序（默认）
    results = sorted(results, key=lambda x: x[1], reverse=False)

    # 上述对距离进行升序排序，此时取出距离最近的元素，也相当于取出和预测词句意思最相近的词句
    for _index, distance in results[0:20]:
        review = ""
        for item in inputs[_index]:
            if item == 0 : continue
            review += vocab[(item - 1)]
        print("真实的标签为：{0} \t 词句: {1}".format(labels[_index],review),"(Distance: %.4f)" % (distance))



if __name__=='__main__':
    target = "味道好，送餐速度也快"
    while True:
        target = input("input:")
        semantic_search(target)
