# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 08:36:36 2019

@author: 86157
"""

import jieba
import csv
import pickle

jieba.load_userdict("./dict/userdict.txt")
pos_review_path='./data/pos_review.pkl'
neg_review_path='./data/neg_review.pkl'
test_review_path='./data/test_review.pkl'
train_path = './data/train.csv'
test_path='./data/test.csv'
stopword_path = './dict/stopwords.txt'


def load_train_corpus():
    with open(train_path, 'r', encoding='UTF-8-sig' ) as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    pos_review_list = []
    neg_review_list = []
    # 第一列为差评/好评， 第二列为评论
    for words in rows:
        if(words[0]=='0'):
           # neg_review_list.append(review_to_text(words[1]))
           neg_review_list.append(words[1])
        else:
            #pos_review_list.append(review_to_text(words[1]))
            pos_review_list.append(words[1])
    return pos_review_list, neg_review_list

def load_test_corpus():
    with open(test_path, 'r', encoding='UTF-8-sig' ) as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    review_list = []
    # 第一列为差评/好评， 第二列为评论
    for words in rows:
        #tup=(review_to_text(words[1]),'neg' if words[0]=='0' else 'pos')
        li=[words[1],'neg' if words[0]=='0' else 'pos']
        review_list.append(li)
    return review_list
    
def dump_file():
    review_pos,review_neg = load_train_corpus()
    review_test = load_test_corpus()
    with open(pos_review_path, 'wb') as file:
        pickle.dump(review_pos, file)
    with open(neg_review_path, 'wb') as file:
        pickle.dump(review_neg, file)
    with open(test_review_path, 'wb') as file:
        pickle.dump(review_test, file)

dump_file()
    