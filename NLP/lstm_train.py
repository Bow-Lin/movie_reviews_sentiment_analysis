#! /bin/env python
# -*- coding: utf-8 -*-
"""
训练网络，并保存模型，其中LSTM的实现采用Python中的keras库
"""
import pandas as pd 
import numpy as np 
import jieba
import multiprocessing
import keras

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
np.random.seed(1337)  # For Reproducibility
import sys
sys.setrecursionlimit(1000000)
import yaml
import xlrd
# set parameters:
cpu_count = multiprocessing.cpu_count() # 4
vocab_dim = 100
n_iterations = 1  # ideally more..
n_exposures = 10 # 所有频数超过10的词语
window_size = 7
n_epoch = 1
input_length = 100
maxlen = 100
batch_size = 32

def loadfile():
    wb = xlrd.open_workbook('../Demo/data/dataSet.xlsx')  # 打开Excel文件
    sheet = wb.sheet_by_name('ceshi')  # 通过excel表格名称(rank)获取工作表
    sheet2 = wb.sheet_by_name('yanzheng')
    list = []
    labels = []
    trainData = []
    for a in range(sheet.nrows):  #循环读取表格内容（每次读取一行数据）
        line = sheet.row_values(a)  # 每行数据赋值给cells
        a=[]
        list.append(line[1])
        labels.append(line[0])
        trainData.append((line[0],line[1]))
    testData=[]
    for a in range(sheet2.nrows):  # 循环读取表格内容（每次读取一行数据）
        line = sheet2.row_values(a)  # 每行数据赋值给cells
        testData.append((line[0],line[1]))
    testData = np.array(testData)
    # print(testData[0])
    np.random.shuffle(testData)
    # print(testData[0])
    np.random.shuffle(trainData)
    combineData = np.concatenate((trainData,testData))
    lab = []
    data = []
    for i in range(len(combineData)):
        t = float(combineData[i][0])
        lab.append(int(t))
        data.append(combineData[i][1])
    lab = np.array(lab)
    data = np.array(data)
    return data,lab


#对句子经行分词，并去掉换行符
def tokenizer(text):
    ''' Simple Parser converting each document to lower-case, then
        removing the breaks for new lines and finally splitting on the
        whitespace
    '''
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text


def create_dictionaries(model=None,
                        allData=None):
    '''创建每个词语的索引，词向量，以及每个句子所对应的词语索引
    '''
    if (allData is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        #  freqxiao10->0 所以k+1
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引,(k->v)=>(v->k)
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量, (word->model(word))

        def parse_dataset(allData): # 闭包-->临时使用
            ''' Words become integers
            '''
            data=[]
            for sentence in allData:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0) # freqxiao10->0
                data.append(new_txt)
            return data # word=>index
        allData=parse_dataset(allData)
        allData= sequence.pad_sequences(allData, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,allData
    else:
        print('No data provided...')


#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(allData):

    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model.build_vocab(allData) # input: list
    model.train(allData, epochs=model.epochs, total_examples=model.corpus_count)
    model.save('../Demo/model/Word2vec_model.pkl')
    index_dict, word_vectors,allData = create_dictionaries(model=model,allData=allData)
    return index_dict, word_vectors,allData


def get_data(index_dict,word_vectors,allData,y):
    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim)) # 初始化 索引为0的词语，词向量全为0
    for word, index in index_dict.items(): # 从索引为1的词语开始，对每个词语对应其词向量1
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(allData, y, test_size=0.2)
    y_train = keras.utils.to_categorical(y_train,num_classes=2)
    y_test = keras.utils.to_categorical(y_test,num_classes=2)
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test
##定义网络结构
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(LSTM(activation="tanh", units=64))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Activation('softmax'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))  # Dense=>全连接层,输出维度=2
    model.add(Activation('softmax'))
    print('Compiling the Model...')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    print("Train...") # batch_size=32
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch,verbose=1)

    print("Evaluate...")
    score = model.evaluate(x_test, y_test,
                                batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('../Demo/model/lstm.yml', 'w') as outfile:
        outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
    model.save_weights('../Demo/model/lstm.h5')
    print('Test score:', score)


#训练模型，并保存
print('Loading Data...')
allData,y=loadfile()
print(len(allData),len(y))
print('Tokenising...')
allData = tokenizer(allData)
print('Training a Word2vec model...')
index_dict, word_vectors,allData=word2vec_train(allData)
# index_dict, word_vectors,testLists=word2vec_train(allData)
print('Setting up Arrays for Keras Embedding Layer...')
n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,allData,y)
print("x_train.shape and y_train.shape:")
print(x_train.shape,y_train.shape)
train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)