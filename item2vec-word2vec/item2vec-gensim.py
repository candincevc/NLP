# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a file for item2vec data.

@ author candincevc

"""

import pandas as pd
import numpy as np
import gensim.models
from gensim.corpora import Dictionary

movies=pd.read_csv(r'./ml-latest-small/movies.csv')
data=pd.read_csv(r'./ml-latest-small/ratings.csv')

class item2vectest():
    def __init__(self,freq,sgval):
        self.freq=freq
        self.sgval=sgval
              

    def datapre(self,data):
        '''
        数据预处理
        '''
        print('数据维度：',data.shape)
        
        # 过滤到评分低于3.5分以下的
        data=data[data.rating>=3.5]
       
        # 数据集按用户、时间排序
        data=data.sort_values(['userId','timestamp'])
        
        # 行索引重排
        data.reset_index(drop=True,inplace=True)
        
        #movieid int to str
        data['movieId']=data['movieId'].apply(str)
        
        # 拼接到用户list
        mvlist=data.groupby('userId')['movieId'].apply(list)
        print(mvlist.head())
        
        corpus=list(mvlist.values)
        
        return corpus
    
    def getmodel(self,corpus):
    
        '''
        构建word2vec model
        '''
        # check 词典数
        dct = Dictionary(corpus) 
        print('语料中的词个数：',len(dct))
        
        # 训练word2vec模型
        # 设定min_count,sg参数
        model = gensim.models.Word2Vec(sentences=corpus,min_count=self.freq,sg=self.sgval)
        
    
    
        #获取词向量
        wordvec=model.wv.vectors
        word=model.wv.index2word
        
        # 获取模型词典个数
        print('模型中词典个数：',model.wv.vectors.shape)
        
        return model,wordvec,word
    
    def test(self,mv_name,model):
        mv_id=movies[movies.title==mv_name].movieId.values[0]
        m_s=model.most_similar(str(mv_id))
        print('用户评论电影：',movies[movies.movieId==int(mv_id)].values)
        for i in m_s:
            print('topN:',movies[movies.movieId==int(i[0])].values)
        print('-------------------------')
        return
    

item2vec_o=item2vectest(freq=1,sgval=1)
corpus=item2vec_o.datapre(data)
model,wordvec,word=item2vec_o.getmodel(corpus)

testmovies=movies.head(2)
for mv_name in list(testmovies.title):
    
    item2vec_o.test(mv_name,model)
    
    
    

