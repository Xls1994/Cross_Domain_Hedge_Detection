#coding: utf-8
import math
import pickle as pkl
import string
import time
from tqdm import tqdm
from decimal import Decimal
from keras.utils import to_categorical
import tensorflow as tf
from keras.layers import Embedding, Dropout, BatchNormalization, Activation
from keras.layers import Input, concatenate, Dense, Conv1D, LSTM
from keras.layers import AveragePooling1D, GlobalMaxPool1D, GlobalAveragePooling1D, Lambda, Flatten, Reshape
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Model, load_model, Sequential
from keras.optimizers import RMSprop, Adam, SGD
from keras.utils import plot_model
from keras.regularizers import l2
import keras.backend as K
import numpy as np
from collections import OrderedDict

config =tf.ConfigProto()
config.gpu_options.allow_growth =True
sess =tf.Session(config=config)
K.set_session(sess)


def _shared_layer(concat_input):
    '''
    共享不同任务的Embedding层和bilstm层
    :param input_data:
    :return:
    '''
    output = Bidirectional(LSTM(units=100, return_sequences=False, kernel_regularizer=l2(1e-4),
                                     bias_regularizer=l2(1e-4)), name='bilstm1')(concat_input)
    output = Dropout(0.5)(output)
    return output


def _private_layer(input_data, output_shared, modelName=None):
    # 参考对抗中文分词
    private = Bidirectional(LSTM(units=100, return_sequences=False, kernel_regularizer=l2(1e-4),
                                bias_regularizer=l2(1e-4)), merge_mode='concat')(input_data)
    # private = Dropout(0.5)(private)
    private = concatenate([private, output_shared], axis=-1)

    output = Dense(100, activation='tanh')(private)
    output = Dense(2, activation='softmax')(output)
    return output


def discriminator(output_shared, n_corpus, state=True):
    '''
    classification layer
    :param output_pub:
    :return:
    '''
    dr_gan = Dropout(0.5, name='dr_gan')(output_shared)
    pool_gan = GlobalAveragePooling1D(name='pool_gan')(dr_gan)
    output = Dense(n_corpus, activation='softmax', name='softmax_gan')(pool_gan)
    if state==False:
        dr_gan.trainable = False
        pool_gan.trainable = False
        output.trainable = False
    return output


def calculate_loss(y_true, y_pred):
    current_loss = -K.sum(K.categorical_crossentropy(y_true, y_pred))
    return current_loss


def buildModel(embedding_matrix):
    emb_token = Embedding(input_dim=embedding_matrix.shape[0],  # 索引字典大小
                          output_dim=embedding_matrix.shape[1],  # 词向量的维度
                          weights=[embedding_matrix],
                          trainable=True,
                          name='token_emd')

    tokens_input = Input(shape=(max_word,),  # 若为None则代表输入序列是变长序列
                         name='tokens_input', dtype='int32')
    tokens = emb_token(tokens_input)

    pos_input = Input(shape=(max_word,), name='pos_input')
    pos = Embedding(input_dim=len(pos_index),  # 索引字典大小
                    output_dim=50,  # 词向量的维度
                    trainable=True,
                    name='pos_emd')(pos_input)
    gx_input = Input(shape=(max_word,), name='gx_input')
    gx = Embedding(input_dim=len(gx_index),  # 索引字典大小
                    output_dim=10,  # 词向量的维度
                    trainable=True,
                    name='gx_emd')(gx_input)
    mergeLayers = [tokens, pos,gx]
    concat_input = concatenate(mergeLayers, axis=-1)  # (none, none, 230)

    # Dropout on final input
    concat_input = Dropout(0.5)(concat_input)
    output_pub = _shared_layer(concat_input)    # (none, none, 200)

    # CWS Classifier
    models = {}
    for modelName in ['main', 'aux']:
        output = _private_layer(concat_input, output_pub, modelName)
        # output_adv = discriminator(output_pub, n_corpus=2, state=False)
        model = Model(inputs=[tokens_input, pos_input,gx_input], outputs=[output])
        rmsprop = RMSprop(lr=1e-3, clipvalue=5.0)
        # sgd = SGD(lr=0.001, momentum=0.9, decay=0., nesterov=True, clipvalue=5)
        model.compile(loss=['categorical_crossentropy'],
                        metrics=["accuracy"],   # sparse_categorical_accuracy
                        optimizer=rmsprop)
        models[modelName] = model
    models['main'].summary()

    # '''
    # discriminator
    # '''
    # output_adv = discriminator(output_pub, n_corpus=2, state=True)
    # # output, loss_function = _private_layer(lstm_input, output_pub)
    # # concat_output = concatenate(inputs=[output, output_adv], name='concat_output')
    # adv_model = Model(inputs=[tokens_input, chars_input], outputs=output_adv)
    # rmsprop = RMSprop(lr=2e-4, clipnorm=5.)
    # adv_model.compile(loss='categorical_crossentropy',    # custom_loss
    #             metrics=['categorical_accuracy'],
    #             optimizer=rmsprop)
    # # discriminator.summary()
    # models['discriminator'] = adv_model

    '''保存模型为图片
    pip3 install pydot-ng 
    sudo apt-get install graphviz'''
    # plot_model(models['discriminator'], to_file='model.png')
    return models


# 该回调函数将在每个epoch后保存概率文件
from keras.callbacks import Callback
class WritePRF(Callback):
    def __init__(self, max_f, X_test, y_test,path):
        super(WritePRF, self).__init__()
        self.x_test = np.asarray(X_test[0])
        self.pos_test = np.asarray(X_test[1])
        self.gx_test =np.asarray(X_test[2])
        self.y_true = y_test
        self.max_f = max_f
        self.path=path

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict([self.x_test, self.pos_test,self.gx_test])  # 测试
        y_pred = predictions.argmax(axis=-1)  # Predict classes
        pre, rec, f1 = predictLabels2(y_pred, self.y_true)

        if f1 >= self.max_f:
            self.max_f = f1
            self.model.save(self.path+'Model.h5', overwrite=True)
            print('do saving')
            with open(self.path+'prf.txt', 'a') as pf:
                print('write prf...... ')
                pf.write("epoch= " + str(epoch) + '\n')
                pf.write("precision= " + str(pre) + '\n')
                pf.write("recall= " + str(rec) + '\n')
                pf.write("Fscore= " + str(f1) + '\n')


def predictLabels2(y_pred, y_true):
    # y_true = np.squeeze(y_true, -1)
    lable_pred = list(y_pred)
    lable_true = list(y_true)
    # print(lable_pred)
    # print(lable_true)

    print('\n计算PRF...')
    # import BIOF1Validation
    # pre, rec, f1 = BIOF1Validation.compute_f1(lable_pred, lable_true, idx2label, 'O', 'OBI')
    pre, rec, f1 = prf(lable_pred, lable_true, idx2label)
    print('precision: {:.2f}%'.format(100.*pre))
    print('recall: {:.2f}%'.format(100.*rec))
    print('f1: {:.2f}%'.format(100.*f1))

    return round(Decimal(100.*pre), 2), round(Decimal(100.*rec), 2), round(Decimal(100.*f1), 2)


def prf(lable_pred, lable_true, idx2label):
    '''
    数据中1的个数为a，预测1的次数为b，预测1命中的次数为c
    准确率 precision = c / b
    召回率 recall = c / a
    f1_score = 2 * precision * recall / (precision + recall)
    '''
    assert len(lable_pred)==len(lable_true)

    a = 0.
    for i in range(len(lable_true)):
        if lable_true[i]==1:
            a+=1
    b = 0.
    for i in range(len(lable_pred)):
        if lable_pred[i] == 1:
            b += 1

    c=0.
    for i in range(len(lable_true)):
        if lable_pred[i]==1 and lable_true[i]==1:
            c+=1

    precision = c/b
    recall = c/a
    f1 = 2*precision*recall / (precision+recall)
    return precision, recall, f1


if __name__ == '__main__':

    # load data
    root_list =['abstract_by_wiki','discuss_by_wiki','result_by_wiki',
    'wiki_by_abstract','wiki_by_discuss','wiki_by_result']
    fold_list =['/2/','/3/','/4/','/5/']
    for root in root_list:
        for fold in fold_list:

            root1 = 'corpus_extracted/'+root+fold
            save_path =root1+'pkl/'
            print root1
            label2idx = {'0': 0, '1': 1}
            idx2label = {0: '0', 1: '1'}

            with open(root1 + 'pkl/train.pkl', "rb") as f:
                train_x, train_y, train_pos,train_gx = pkl.load(f)
            with open(root1 + 'pkl/test.pkl', "rb") as f:
                test_x, test_y, test_pos,test_gx = pkl.load(f)
            with open(root1 + 'pkl/emb.pkl', "rb") as f:
                embedding_matrix, pos_index, max_word = pkl.load(f)
            gx_index =[0,1,2]
            print(len(train_x), len(train_y), len(train_pos))  # 2890
            print(len(test_x), len(test_y), len(test_pos))  # 2890

            epochs=40
            batch_size = 32
            max_f = 0.

            test_y = np.asarray(test_y).argmax(axis=-1)  # Predict classes
            # print(len(y_true))

            models = buildModel(embedding_matrix)
            # 该回调函数将在每个epoch后保存概率文件
            write_prob = WritePRF(max_f, [test_x, test_pos,test_gx], test_y,save_path)
            models['main'].fit(x=[np.asarray(train_x), np.asarray(train_pos),np.asarray(train_gx)], y=np.asarray(train_y),
                                epochs=epochs, batch_size=batch_size,
                                callbacks=[write_prob])


            # 预测
            model = load_model('model/bestModel.h5', custom_objects=create_custom_objects())
            ConllevalCallback(epoch, max_f, [test_x, test_char], test_y, model)
