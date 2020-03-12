# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division 

import time, sys, os, re
from progressbar import *
from gensim.models import word2vec  
'''
jieba分词
'''
import jieba
import logging

def get_words_by_jieba(filePath,fileSegWordDonePath):
    # read the file by line
    fileTrainRead = []
    #fileTestRead = []
    with open(filePath) as fileTrainRaw:
        for line in fileTrainRaw:
            fileTrainRead.append(line)

    def PrintListChinese(list):
        for i in range(len(list)):
            print(list[i],)

    fileTrainSeg=[]
    progress=ProgressBar()
    print('====================step1 分词====================\n')
    for i in progress(range(len(fileTrainRead))):
#        time.sleep(0.01)
        fileTrainSeg.append([' '.join(list(jieba.cut(fileTrainRead[i][20:-4],cut_all=False)))])

    # test        
    # PrintListChinese(fileTrainSeg[10])

    # save the result
    with open(fileSegWordDonePath,'wb') as fW:
        for i in range(len(fileTrainSeg)):
            fW.write(fileTrainSeg[i][0].encode('utf-8'))


'''
停用词
'''
# 创建停用词list  
def stopwordslist(filepath):  
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]  
    return stopwords  

# 对句子进行分词  
def seg_sentence(word,stopwords):  
#    sentence_seged = jieba.cut(sentence.strip())  
#    stopwords = stopwordslist('./stopwords.txt')  # 加载停用词的路径  
#    print(stopwords)
    outstr = ''  
#    for word in sentence_seged:    
    if word not in stopwords:  
        if word != '\t': 
            outstr = word
#            outstr += word  
#            outstr += " "  
    return outstr  


def word_to_vec():
    '''
    word2vec训练词向量
    '''
    logging.basicConfig(format='%(asctime)s:%(levelname)s: %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(u"./Done_2.txt")  # 加载语料  
    model = word2vec.Word2Vec(sentences, sg=0, size=50, window=10, negative=5, hs=1, iter=10, workers=4)
    model.wv.save_word2vec_format('ans_train.vec',binary=False) # 保存为非二进制文件
    print('词向量训练完毕！\n')


if __name__ == "__main__":

    filePath='./data/train_data_complete.txt'
    fileSegWordDonePath ='./data/Done.txt'
    get_words_by_jieba(filePath,fileSegWordDonePath)
    inputs = open('./data/Done.txt', 'r', encoding='utf-8') 
    outputs = open('./data/Done_2.txt', 'w') 
    inputfileRead = inputs.read();
    inputfile = re.split(' ',inputfileRead)

    progress_2=ProgressBar()
    stopwords = stopwordslist('./stopwords.txt')  # 加载停用词的路径
    print('==================step2 删除停用词===================\n')
    for i in progress_2(range(len(inputfile))):
        line_seg = seg_sentence(inputfile[i],stopwords)  # 这里的返回值是字符串  
        outputs.write(line_seg + ' ')
#    p.finish()
    outputs.close()
#    inputs.close()

    print('分词完毕！接下来开始训练词向量（按回车键继续）：\n')
    run = input()
    if os.path.exists('./data/Done_2.txt'):
        word_to_vec()
    else:
        print('ERROR!')

