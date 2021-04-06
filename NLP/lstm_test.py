#! /bin/env python
# -*- coding: utf-8 -*-
"""
预测
"""
import jieba
import numpy as np
import xlrd
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
import keras

import yaml
from keras.models import model_from_yaml
np.random.seed(1337)  # For Reproducibility
import sys
sys.setrecursionlimit(1000000)

# define parameters
maxlen = 100
def loadfile():
    wb = xlrd.open_workbook('../data/dataSet.xlsx')  # 打开Excel文件
    sheet2 = wb.sheet_by_name('yanzheng')
    list = []
    labels = []
    for a in range(sheet2.nrows):  #循环读取表格内容（每次读取一行数据）
        line = sheet2.row_values(a)  # 每行数据赋值给cells
        print(line[1])
        a=[]
        list.append(line[1])
        labels.append(line[0])
        print(line[0])
        data=int(line[0])#因为表内可能存在多列数据，0代表第一列数据，1代表第二列，以此类推
    list = np.array(list)
    labels = np.array(labels)
def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        #  freqxiao10->0 所以k+1
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引,(k->v)=>(v->k)
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量, (word->model(word))

        def parse_dataset(combined): # 闭包-->临时使用
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0) # freqxiao10->0
                data.append(new_txt)
            return data # word=>index
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print ('No data provided...')


def input_transform(string):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load('../Demo/model/Word2vec_model.pkl')
    _,_,combined=create_dictionaries(model,words)
    return combined


def lstm_predict(string):
    print ('loading model......')
    keras.backend.clear_session()
    with open('../Demo/model/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f, Loader=yaml.FullLoader)
        #yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print ('loading weights......')
    model.load_weights('../Demo/model/lstm.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    data=input_transform(string)
    data.reshape(1,-1)
    result=model.predict_classes(data)
    result2 = model.predict(data)
    max = 0
    if(result2[0][0]>result2[0][1]):
        max = result2[0][0]
    else:
        max = result2[0][1]
    return result2[0][1]

if __name__=='__main__':
    string = ""
    keras.backend.clear_session()
    result = lstm_predict(string)
    print(result)
