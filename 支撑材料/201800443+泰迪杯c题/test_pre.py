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
    questions，answers，answers_id
    """
    question = ""
    q_num = 0
    questions, answers, answers_id = [], [], []
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
            ans_id = int(arr[2])
            answers_id.append(ans_id)
    print('==========Total read: %d questions==========\n'%q_num)
    return questions, answers, answers_id

def predict_fun(model, f, question, answer, id):
    question = np.array(question)
    question.shape = (1,100)
    answer = np.array(answer)
    answer.shape = (1,100)
    predict_label = model.predict([question, answer],verbose=0)
    print(predict_label[:,0][0],predict_label[:,1][0])
    print(type(predict_label[:,0][0]))
    pred = (2-(predict_label[:,0][0]>predict_label[:,1][0]))-1
    predict = str(id) + ',' + str(pred) + '\n'
    f.write(predict)
    

if __name__ == '__main__':
	
    filename = './ans_train.vec'
    testfile = './test_data_complete_done.txt'

    print('Loading data...')
    embeddings, word2idx = loadWordVec(filename)
    questions, answers, answers_id = loadData(testfile, word2idx, 100)
    questions = np.array(questions)
    answers = np.array(answers)

    print('Load_data done!')

    # 从磁盘载入模型结构
    from keras.models import model_from_json
    import h5py
    json_file = open('./final_lstm_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # 从磁盘读入模型权重
    model.load_weights("./final_lstm_model.h5")
    print("载入模型完毕")
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print('模型编译完毕...')
    
    f = open('predict_label.txt', mode='w', encoding='utf-8')
    for question, answer, id in zip(questions, answers, answers_id):
        predict_fun(model, f, question, answer,id)

    f.close()
    print("finish")
