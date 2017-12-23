#coding: utf-8
import os
import pickle as pkl
import string
from collections import OrderedDict
import numpy as np
np.random.seed(1337)
from tqdm import tqdm
import codecs

from nltk.corpus import stopwords
stop_word = stopwords.words('english')

nb_word = 0 # 用于计算临时的最长句子的长度
max_word = 0 # 用于保存最长句子的长度（不用）
EMBEDDING_DIM = 100 # 词向量的维
MAX_NB_WORDS = 1000000

label2idx = {'0':0, '1':1}
# idx2label = {0:'O', 1:'B', 2:'I'}
# label2idx = {'O': 0, 'B':1, 'I':2}

embeddingFile = './corpus_extracted/vec-100.txt'

def get_word_index(trainFile, testFile):
    """
    1、建立索引字典- word_index
    :param trainFile:
    :param testFile:
    :return:
    """
    print('\n获取索引字典- word_index \n')
    word_counts = {}

    for f in [trainFile, testFile]:
        for line in f:
            for w in line:
                if w in word_counts:
                    word_counts[w] += 1
                else:
                    word_counts[w] = 1

    # 根据词频来确定每个词的索引
    wcounts = list(word_counts.items())
    wcounts.sort(key=lambda x: x[1], reverse=True)
    sorted_voc = [wc[0] for wc in wcounts]
    # note that index 0 is reserved, never assigned to an existing word
    word_index = dict(list(zip(sorted_voc, list(range(1, len(sorted_voc) + 1)))))

    # 加入未登陆新词和填充词
    word_index['retain-padding'] = 0
    word_index['retain-unknown'] = len(word_index)
    print('word_index 长度：%d' % len(word_index))
    return word_index


def pos2index(pos, pos_index):
    """
    将一行转换为转化为词索引序列
    :param sent:
    :param word_index:
    :return:
    """
    charVec = []
    for p in pos:
        charVec.append(pos_index[p])
    while len(charVec)<max_word:
        charVec.append(0)
    return charVec[:]


def sent2vec2(sent, word_index):
    """
    将一行转换为转化为词索引序列
    :param sent:
    :param word_index:
    :return:
    """
    charVec = []
    for char in sent:
        if char in word_index:
            charVec.append(word_index[char])
        else:
            print(char)
            charVec.append(word_index['retain-unknown'])
    while len(charVec) < max_word:
        charVec.append(0)
    return [i for i in charVec]


def doc2vec(train, label1, pos1, test, label2, pos2,gx1,gx2, word_index):
    """
    2、将全部训练和测试语料 转化为词索引序列
    :param ftrain:
    :param ftrain_label:
    :param ftest:
    :param ftest_label:
    :param word_index:
    :return:
    """
    x_train, y_train, x_test, y_test = [], [], [], []
    pos_train, pos_test = [], []
    # tags = ['-1', '+1']  # 标注统计信息对应 [ 1.  0.] [ 0.  1.]
    gx_train,gx_test =[],[]
    for line in train:
        index_line = sent2vec2(line, word_index)
        print(index_line)
        x_train.append(index_line)
    for line in test:
        index_line = sent2vec2(line, word_index)
        x_test.append(index_line)
    for line in pos1:
        index_line = pos2index(line, pos_index)
        pos_train.append(index_line)
    for line in pos2:
        index_line = pos2index(line, pos_index)
        pos_test.append(index_line)
    for line in gx1:
        index_line = pos2index(line, gx_index)
        gx_train.append(index_line)
    for line in gx2:
        index_line = pos2index(line, gx_index)
        gx_test.append(index_line)
    for line in label1:
        index = label2idx.get(line)
        index_line = [0, 0]
        index_line[index]=1
        y_train.append(index_line)
    for line in label2:
        index = label2idx.get(line)
        index_line = [0, 0]
        index_line[index] = 1
        y_test.append(index_line)

    return x_train, y_train, pos_train, x_test, y_test, pos_test,gx_train,gx_test


def process_data(datasDic, labelsDic, posDic,gxDic, word_index):
    """
    3、将转化后的 词索引序列 转化为神经网络训练所用的张量
    :param data_label:
    :param word_index:
    :param max_len:
    :return:
    """
    train = datasDic['train']
    label1 = labelsDic['train']
    pos1 = posDic['train']
    test = datasDic['test']
    label2 = labelsDic['test']
    pos2 = posDic['test']
    gx1 =gxDic['train']
    gx2 =gxDic['test']
    return doc2vec(train, label1, pos1, test, label2, pos2,gx1,gx2, word_index)



def readEmbedFile(embFile):
    """
    读取预训练的词向量文件，引入外部知识
    """
    print("\nProcessing Embedding File...")
    embeddings = OrderedDict()
    embeddings["PADDING_TOKEN"] = np.zeros(EMBEDDING_DIM)
    embeddings["UNKNOWN_TOKEN"] = np.random.uniform(-0.001, 0.001, EMBEDDING_DIM)
    embeddings["NUMBER"] = np.random.uniform(-0.001, 0.001, EMBEDDING_DIM)

    with codecs.open(embFile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i in tqdm(range(len(lines))):
        line = lines[i]
        if len(line.split())<=2:
            continue
        values = line.strip().split()
        print line
        word =values[0]
        # word = wordNormalize(word)
        vector = np.asarray(values[1:], dtype=np.float32)
        embeddings[word] = vector

    print('Found %s word vectors.' % len(embeddings))  # 693537
    return embeddings


def produce_matrix(word_index):
    miss_num=0
    num=0

    embeddingsDic = readEmbedFile(embFile=embeddingFile)

    num_words = min(MAX_NB_WORDS, len(word_index)+1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddingsDic.get(word)
        if embedding_vector is not None:
            num=num+1
            embedding_matrix[i] = embedding_vector
        else:
            # words not found in embedding index will be all-random.
            vec = np.random.uniform(-0.001, 0.001, size=EMBEDDING_DIM)  # 随机初始化
            embedding_matrix[i] = vec
            miss_num=miss_num+1
    print('missnum',miss_num)    # 10604
    print('num',num)    # 56293
    return embedding_matrix

if __name__ == '__main__':
    root_list =['corpus_extracted/wiki_by_abstract',
                'corpus_extracted/wiki_by_discuss',
                'corpus_extracted/wiki_by_result',
                'corpus_extracted/abstract_by_wiki',
                'corpus_extracted/discuss_by_wiki',
                'corpus_extracted/result_by_result']
    fold_list =['/2/','/3/','/4/','/5/']
    for r in root_list:
        for fold in fold_list:
            root =r+fold
            print root

            datasDic = {'train':[], 'aux':[], 'test':[]}
            labelsDic = {'train':[], 'aux':[], 'test':[]}
            posDic = {'train':[], 'aux':[], 'test':[]}
            gxDic ={'train':[],'aux':[],'test':[]}
            num = 1
            pos_index = OrderedDict()
            gx_index =OrderedDict()
            for t in ['train', 'aux', 'test']:
                with codecs.open(root + t + '.data', encoding='utf-8') as f:
                    for line in f:
                        token = line.strip('\n').split(' ')
                        max_word = max(max_word, len(token))
                        datasDic[t].append(token)

                with codecs.open(root + t + '.label', encoding='utf-8') as f:
                    for line in f:
                        token = line.strip('\n')
                        labelsDic[t].append(token)

                with codecs.open(root + t + '.pos', encoding='utf-8') as f:
                    for line in f:
                        pos = line.strip('\n').split(' ')
                        posDic[t].append(pos)
                        for p in pos:
                            if p in pos_index:
                                continue
                            else:
                                pos_index[p]=num
                                num+=1
                num =1
                with codecs.open(root + t + '.gx', encoding='utf-8') as f:
                    for line in f:
                        pos = line.strip('\n').split(' ')
                        gxDic[t].append(pos)
                        for p in pos:
                            if p in gx_index:
                                continue
                            else:
                                gx_index[p]=num
                                num+=1
                
            print('max_word: ', max_word)   # 13

            # 取其中一份的 200 个实例作为训练数据
            positive=0
            negative=0
            for i in range(len(labelsDic['aux'][300:])):
                label = labelsDic['aux'][300+i]
                if label=='1' and positive<50:
                    positive+=1
                    datasDic['train'].append(datasDic['aux'][300+i])
                    labelsDic['train'].append(labelsDic['aux'][300+i])
                    posDic['train'].append(posDic['aux'][300+i])
                    gxDic['train'].append(gxDic['aux'][300+i])
                elif label=='0' and negative<150:
                    negative+=1
                    datasDic['train'].append(datasDic['aux'][300+i])
                    labelsDic['train'].append(labelsDic['aux'][300+i])
                    posDic['train'].append(posDic['aux'][300+i])
                    gxDic['train'].append(gxDic['aux'][300+i])
                elif positive==50 and negative==150:
                    break

            print('pos 个数：{}'.format(len(pos_index)))   # 32
            print("pos_index['Blank']: ", pos_index['Blank'])

            word_index = get_word_index(datasDic['train'], datasDic['test'])
            print('Found %s unique tokens.\n' % len(word_index))  # 9962

            x_train, y_train, pos_train, x_test, y_test, pos_test,gx_train,gx_test = process_data(datasDic, labelsDic, posDic,gxDic, word_index)
            print(np.asarray(x_train).shape)
            print(np.asarray(y_train).shape)
            print(np.asarray(pos_train).shape)
            if os.path.isdir(root+'pkl'):
                print 'exists'
            else:
                os.makedirs(root+'pkl')
            with open(root+'pkl/train.pkl', "wb") as f:
                pkl.dump((x_train, y_train, pos_train,gx_train), f, -1)
            with open(root+'pkl/test.pkl', "wb") as f:
                pkl.dump((x_test, y_test, pos_test,gx_test), f, -1)

            embedding_matrix = produce_matrix(word_index)
            with open(root+'pkl/emb.pkl', "wb") as f:
                pkl.dump((embedding_matrix, pos_index, max_word), f, -1)
            embedding_matrix = {}

            print('\n保存成功')
    
    
