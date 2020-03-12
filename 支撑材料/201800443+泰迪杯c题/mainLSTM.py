# -*- coding: UTF-8 -*-
import numpy as np
import dataproc as dp
from progressbar import *
'''
def get_train_data(embeddings,wordindex,questions,answers):
    ques_train = [[] for i in range(len(questions))]
    ans_train = [[] for i in range(len(answers))]
    for i in range(len(questions)):
        ques_t = []
        for j in range(100):
            if questions[i][j]!=0:
                ques_t = np.mean(embeddings[questions[i][j]])
            else:
                ques_t = 0
            ques_train[i].append(ques_t)
    for i in range(len(answers)):
        ans_t = []
        for j in range(100):
            if answers[i][j]!=0:
                ans_t = np.mean(embeddings[answers[i][j]])
            else:
                ans_t = 0
            ans_train[i].append(ans_t)
    return ques_train,ans_train
'''

def get_train_data(embeddings,questions,answers):
    ques_train = np.zeros((len(questions),50))
    ans_train = np.zeros((len(answers),50))
    progress = ProgressBar()
    for i in progress(range(len(questions))):
        ques_t, ans_t = [], []
        for j in range(50):
            ques_mean, ans_mean = [], []
            for k in range(100):
                ques_m = embeddings[questions[i][k]][j]
                ques_mean.append(ques_m)    # ques_mean:100
                ans_m = embeddings[answers[i][k]][j]
                ans_mean.append(ans_m)    # ans_mean:100
            ques_t.append(np.mean(ques_mean))  # ques_t:50
            ans_t.append(np.mean(ans_mean))  # ans_t:50
        
        ques_train[i] = ques_t  # ques_train:(len(questions)*50
        ans_train[i] = ans_t  # ans_train:(len(answers)*50

    return ques_train,ans_train

'''
def generate_batch_data_random(x, y, batch_size):
    """逐步提取batch数据到显存，降低对显存的占用"""
    ylen = len(y)
    loopcount = ylen // batch_size
    while (True):
        i = randint(0,loopcount)
        yield x[i * batch_size:(i + 1) * batch_size], y[i * batch_size:(i + 1) * batch_size]
'''

if __name__ == '__main__':

    import keras
    from keras.models import Sequential, Model
    from keras.layers import Input, Dense, Activation, Dropout, Embedding, Reshape, Dot, Concatenate, Multiply, Merge
    from keras.layers import LSTM
    from keras.models import model_from_json
    from keras.utils.np_utils import to_categorical
    
    vec_file = './data/ans_train.vec'
    # vec_file = './ans_train.word2vec'
    txt_file = './data/train_data_complete_done.txt'
    # txt_file = './train_data_done.txt'

    ques_len = 100
    ans_len  = 100
    wordvec_dim=50
    VALIDATION_SPLIT = 0.16 #验证集比例
    TEST_SPLIT = 0.2        #测试集比例
    
    # 加载数据集
    print('Loading dataset...(may cost 30 mins)\n')
    embeddings,wordindex = dp.loadWordVec(vec_file)
    questions,answers,labels, questionIds = dp.loadData(txt_file, wordindex, 100)
    ques_train,ans_train = get_train_data(embeddings,questions,answers)
    questions = np.array(questions)   #转为narray
    answers   = np.array(answers)
    labels    = np.array(labels)
    cat_labels = to_categorical(labels,num_classes=None)
    # 划分数据集
    p1 = int(len(questions)*(1-VALIDATION_SPLIT-TEST_SPLIT))
    p2 = int(len(questions)*(1-TEST_SPLIT))
    ques_tr = questions[:p1]
    ans_tr  = answers[:p1]
    y_tr    = labels[:p1]
    ques_val= questions[p1:p2]
    ans_val = answers[p1:p2]
    y_val   = labels[p1:p2]
    ques_te = questions[p2:]
    ans_te  = answers[p2:]
    y_te    = labels[p2:]
        
    print('Load done! Press Enter to continue...\n')
    run = input()

    # 网络结构
    q_embedding=Embedding(len(questions),
                        wordvec_dim,
                        weights=[ques_train],
                        input_length=ques_len,
                        trainable=False)
    a_embedding=Embedding(len(answers),
                        wordvec_dim,
                        weights=[ans_train],
                        input_length=ans_len,
                        trainable=False)
    
    model_q = Sequential()
    model_q.add(q_embedding)
    model_q.add(LSTM(50,dropout=0.2))
    model_q.add(Dropout(0.2))
    model_q.add(Dense(10,input_shape=(None,50)))
    
    model_a = Sequential()
    model_a.add(a_embedding)
    model_a.add(LSTM(50,dropout=0.2))
    model_a.add(Dropout(0.2))
    model_a.add(Dense(10,input_shape=(None,50)))
    
    model = Sequential()
    model.add(Merge([model_q,model_a],mode='dot'))
    model.add(Dense(2,input_shape=(None,1),activation='softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    
    #begin to train
    print('Begin to train...\n')
    model.fit([questions, answers],cat_labels, batch_size=256, epochs=10,shuffle=True,verbose=1,validation_split=0.2)
    # model.fit([ques_tr, ans_tr], y_tr,validation_data=([ques_val,ans_val], y_val), batch_size=256, epochs=20)
    
    # save model
    model_json = model.to_json()
    with open("./data/final_lstm_model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("./data/final_lstm_model.h5")
    print("Model has been writen to disk.")
