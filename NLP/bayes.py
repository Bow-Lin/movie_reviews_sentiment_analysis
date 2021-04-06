# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 11:27:13 2019

@author: 86157
"""

import pickle
import jieba
import itertools
import re

stopword_path = '../Demo/data/stopwords.txt'
fea_pro_dict_pos = {}
fea_pro_dict_neg = {}

"""先验概率"""
CK_POS = 27629#训练集正类别数据量
CK_NEG = 21055#训练集负类别数据量
CK = 48684#训练集总样本数量
TEST = 12172#测试集总样本数量

P_CK_POS = CK_POS/CK#正向类别在样本集中概率
P_CK_NEG = CK_NEG/CK#负向类别在样本集中概率

def load_stopwords(file_path):
    stop_words = []
    with open(file_path, encoding='UTF-8') as words:
       stop_words.extend([i.strip() for i in words.readlines()])
    return stop_words


"""jieba分词+去停用词"""
def review_to_text(review):
    stop_words = load_stopwords(stopword_path)
    review = re.sub(r"[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", "", review) #去标点符号
    review = jieba.cut(review)
    all_stop_words = set(stop_words)
    review_words = [w for w in review if w not in all_stop_words]#去停用词
    return review_words

"""条件概率"""
def trainmodel():
    pos_review = pickle.load(open('./data/pos_review.pkl','rb'))
    neg_review = pickle.load(open('./data/neg_review.pkl','rb'))
        
    pos_words = []
    neg_words = []
    for i in pos_review:
        pos_words.append(review_to_text(i))
    for j in neg_review:
        neg_words.append(review_to_text(j))
        
    poswords=list(itertools.chain(*pos_words))
    negwords=list(itertools.chain(*neg_words))
    for i in poswords:
        if fea_pro_dict_pos.__contains__(i):
            fea_pro_dict_pos[i]=fea_pro_dict_pos[i]+1
        else:
            fea_pro_dict_pos[i]=1
    for j in negwords:
        if fea_pro_dict_neg.__contains__(j):
            fea_pro_dict_neg[j]=fea_pro_dict_neg[j]+1
        else:
            fea_pro_dict_neg[j]=1 
    for key,values in fea_pro_dict_pos.items():
        fea_pro_dict_pos[key] = values/CK_POS
    for key,values in fea_pro_dict_neg.items():
        fea_pro_dict_neg[key] = values/CK_NEG
    
    with open('bayesmodel.pkl', 'wb') as file:
        pickle.dump([fea_pro_dict_pos,fea_pro_dict_neg,P_CK_POS,P_CK_NEG], file)
    
def pred(text):   
    text_words = review_to_text(text)
    P_POS = 1
    P_NEG = 1
    
    for word in text_words:
        if fea_pro_dict_pos.__contains__(word):
            P_POS = P_POS * fea_pro_dict_pos[word]
        if fea_pro_dict_neg.__contains__(word):
            P_NEG = P_NEG * fea_pro_dict_neg[word] 
    
    P_POS = P_POS * P_CK_POS
    P_NEG = P_NEG * P_CK_NEG
    
    sentiment = "pos" if P_POS>P_NEG else "neg"
    return sentiment
    
def accuracy():
    test=pickle.load(open('./data/test_review.pkl','rb'))
    flag = 0
    
    for i in test:
        if pred(i[0])==i[1]:
            flag = flag+1
    
    acc = flag/TEST
    return acc
    
if __name__=="__main__":
    trainmodel()
    #text = '倍感失望的一部电影，真心没想到能差到这个地步。节奏的把控的失误和角色的定位模糊绝对是整部影片的硬伤。'

