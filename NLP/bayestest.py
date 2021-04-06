# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 12:03:49 2019

@author: 86157
"""
from NLP.bayes import review_to_text
#from bayes import review_to_text
import pickle

def predict(text): 
    model = pickle.load(open("../Demo/data/bayesmodel.pkl","rb"))
    
    fea_pro_dict_pos = model[0]
    fea_pro_dict_neg = model[1]
    P_CK_POS = model[2]
    P_CK_NEG = model[3]
    
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
    return(P_POS/(P_POS+P_NEG))
    
#predict('倍感失望的一部电影，真心没想到能差到这个地步。节奏的把控的失误和角色的定位模糊绝对是整部影片的硬伤。')