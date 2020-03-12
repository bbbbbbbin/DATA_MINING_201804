#-*-coding:utf-8-*-

import jieba
import numpy as np
from collections import defaultdict
from re import match

def loadWordVec(filename):
    """
    加载训练好的词向量文件
    Return:    
    embeddings-- 词向量
    wordindex -- 词索引字典

    """
    embeddings = []
    wordindex  = defaultdict(list)
    with open(filename, "r", encoding="utf-8") as vec:
        for line in vec:
            arr = line.replace('\n'," ").split(" ")
            embedding = [float(val) for val in arr[1: -1]]
            wordindex[arr[0]] = len(wordindex) #按顺序建立索引
            embeddings.append(embedding)
    return embeddings, wordindex


def sen_to_index(sentence, wordindex, max_len):
    """
    Paras:
    sentence  -- 句子
    wordindex -- 词索引字典
    max_len   -- 待解析句子的最大长度
    Return:
    句子的wordvec索引数组
    """
    # unknown = wordindex.get("UNKNOWN", 0)
    # num = wordindex.get("NUM", len(wordindex))
    unknown = 0
    num = 0
    sen_index = [unknown] * max_len
    i = 0
    for word in jieba.cut(sentence):
        if word in wordindex:
            sen_index[i] = wordindex[word]
        else:
            if match("\d+", word):
                sen_index[i] = num
            else:
                sen_index[i] = unknown
        if i >= max_len - 1:
            break
        i += 1
    return sen_index


def loadData(filename, wordindex, max_len):
    """
    Paras:
    wordindex -- 词索引字典
    max_len   -- 最大长度
    Return: 
    questions，answers，labels(0/1),question_IDs
    """
    question = ""
    q_num = 0
    questions, answers, labels, questionIds = [], [], [], []
    with open(filename, mode="r", encoding="utf-8") as rf:
        for line in rf.readlines():
            arr = line.replace('\n',"\t").split("\t")
            if question != arr[0]:
                question = arr[0]
                q_num += 1
            questionIdx = sen_to_index(arr[0].strip(), wordindex, max_len)
            answerIdx = sen_to_index(arr[1].strip(), wordindex, max_len)
            questions.append(questionIdx)
            answers.append(answerIdx)
            ans_label = int(arr[2])
            labels.append(ans_label)
            question_id = int(arr[3])
            questionIds.append(question_id)
    print('==========Total read: %d questions==========\n'%q_num)
    return questions, answers, labels, questionIds

