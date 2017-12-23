#coding: utf-8
# corss domain hedge detection
# train:source domain, test:target domain, aux:target labeled domain
import os

def build_file_path(train_index,test_index,fold_index):
    train_label_file = root1 + types[train_index] + '/' + types[train_index] +'.crf'
    train_file = root2 + types[train_index] + '/' + types[train_index] +'.temp'
    train_gongxian_file =root3+ types[train_index] + '/' + types[train_index] +'.gongxian'
    train_keyword_file =root2+types[train_index] + '/' + types[train_index] +'.keyword'

    aux_label_file = root1 + types[test_index] + '/' + types[test_index] + str(fold_index)+'_train' +'.crf'
    aux_file = root2 + types[test_index] + '/' + types[test_index] + str(fold_index)+'_train' +'.temp'
    aux_gongxian_file =root3+types[test_index] + '/' + types[test_index] + str(fold_index)+ '_train' +'.gongxian'
    aux_keyword_file =root2+types[test_index] + '/' + types[test_index] + str(fold_index)+ '_train' +'.keyword'
    # 另外4份作为测试
    test_label_file = root1 + types[test_index] + '/' + types[test_index] + str(fold_index)+ '_test' +'.crf'
    test_file = root2 + types[test_index] + '/' + types[test_index] + str(fold_index)+ '_test' +'.temp'
    test_gongxian_file = root3 + types[test_index] + '/' + types[test_index] + str(fold_index)+ '_test' +'.gongxian'
    test_keyword_file =root2 + types[test_index] + '/' + types[test_index] + str(fold_index)+ '_test' +'.keyword'
    file_list = [[train_label_file, train_file,train_gongxian_file,train_keyword_file, 'train'],
             [aux_label_file, aux_file, aux_gongxian_file,aux_keyword_file,'aux'],
             [test_label_file, test_file,test_gongxian_file,test_keyword_file, 'test']]
    return file_list

def extract_features(file_list):
    for label_data in file_list:
        '''
        从.crf 文件中获取【词 pos label
        '''
        labels = []
        temp = []
        with open(label_data[0], 'r') as f:
            for line in f:
                if not line == '\t\n':
                    split_line = line.strip('\n').split('\t')
                    temp.append(split_line)
                else:
                    labels.append(temp)
                    temp = []
        gongxian =[]
        temp =[]
        with open(label_data[2], 'r') as f:
            for line in f:
                line =line.strip()
                if not line=='' :

                    temp.append(line)
                else:
                    gongxian.append(temp)
                    temp = []
        keywords =[]
        temp =[]
        with open(label_data[3], 'r') as f:
            for line in f:
                line =line.strip()
                if not line=='' :

                    temp.append(line)
                else:
                    keywords.append(temp)
                    temp = []
        print (len(gongxian),len(labels),len(keywords))

        '''
        从.temp 文件中抽取训练样例、词性文件、标签文件
        '''
        i = 0   # cue 计数器
        cue_num = 0
        tag = 0
        sample = []
        sample_label = []
        sample_pos = []
        sample_gongxian =[]
        sample_keyword =[]
        sentence = []
        sentence_cue=[]
        with open(label_data[1], 'r') as f:
            for line in f:
                if not line=='\n':
                    line = line.strip('\n')
                    if line=='<ccue>':
                        cue_num+=1
                        tag=cue_num
                    elif line=='</ccue>':
                        tag=0
                    else:
                        sentence.append(line)
                        sentence_cue.append(tag)
                else:
                    label = labels[i]
                    gongxian_ =gongxian[i]
                    keyword =keywords[i]
                    k=0
                    while k<len(sentence_cue):
                        if sentence_cue[k]==0:
                            k+=1
                            continue
                        else:
                            # 候选词，加入训练样例中
                            x=[]
                            pos=[]
                            y=[]
                            gx =[]
                            kw =[]
                            if k-2>=0:
                                # 左边两个词存在
                                x.append(sentence[k - 2])
                                pos.append(label[k - 2][1])
                                x.append(sentence[k - 1])
                                pos.append(label[k-1][1])
                            elif k-1>=0:
                                # 左边1个词存在
                                x.append('Blank')
                                pos.append('Blank')
                                x.append(sentence[k - 1])
                                pos.append(label[k-1][1])
                            else:
                                # 左边不存在
                                x.append('Blank')
                                pos.append('Blank')
                                x.append('Blank')
                                pos.append('Blank')

                            x.append(sentence[k])
                            pos.append(label[k][1])
                            y.append('1' if label[k][2]=='B' or label[k][2]=='I' else '0')   # 仅保留当前词的标签
                            gx.append(gongxian_[k])
                            kw.append(keyword[k])
                            # 判断候选样例是否有多个词组成
                            while k+1<len(sentence_cue) and sentence_cue[k+1]==sentence_cue[k]:
                                k+=1
                                x.append(sentence[k])
                                pos.append(label[k][1])

                            if k+2<len(sentence):
                                # 右边两个词存在
                                x.append(sentence[k + 1])
                                pos.append(label[k + 1][1])
                                x.append(sentence[k + 2])
                                pos.append(label[k + 2][1])
                            elif k+1<len(sentence):
                                # 右边1个词存在
                                x.append(sentence[k + 1])
                                pos.append(label[k + 1][1])
                                x.append('Blank')
                                pos.append('Blank')
                            else:
                                x.append('Blank')
                                pos.append('Blank')
                                x.append('Blank')
                                pos.append('Blank')

                            sample.append(x)
                            sample_pos.append(pos)
                            sample_label.append(y)
                            sample_gongxian.append(gx)
                            sample_keyword.append(kw)
                        k+=1
                    sentence = []
                    sentence_cue = []
                    i+=1
        print(sample_pos[-5:])
        print(sample_label[-5:])
        print(sample_gongxian[-5:])
        print (sample_keyword[-5:])

        w = open(root + "/"+label_data[4] + '.data', 'w')
        p = open(root +"/"+ label_data[4] + '.pos', 'w')
        l = open(root +"/"+ label_data[4] + '.label', 'w')
        gx =open(root+"/"+label_data[4]+'.gongxian','w')
        kw =open(root+"/"+label_data[4]+'.keyword','w')
        for i in range(len(sample)):
            word, pos, label = '', '', ''
            gx_line,kw_line ='',''
            for j in range(len(sample[i])):
                word += sample[i][j] + ' '
                pos += sample_pos[i][j] + ' '
            label += sample_label[i][0] + ' '
            gx_line +=sample_gongxian[i][0]+' '
            kw_line+=sample_keyword[i][0]+' '
            w.write(word.strip() + '\n')
            p.write(pos.strip() + '\n')
            l.write(label.strip() + '\n')
            gx.write(gx_line.strip()+'\n')
            kw.write(kw_line.strip()+'\n')
def build_gongxian_features(train_index,test_index,fold_index):
    root ='./corpus_extracted/'+types[test_index]+'_by_'+types[train_index]+'/'+str(fold_index)
    print (root)
    train_file =root +'/train.gongxian'
    train_pos =root+'/train.pos'
    aux_file =root+'/aux.gongxian'
    aux_pos =root+'/aux.pos'
    test_file =root+'/test.gongxian'
    test_pos =root+'/test.pos'
    train_writer =open(root+'/train.gx','w')
    aux_writer =open(root+'/aux.gx','w')
    test_writer =open(root+'/test.gx','w')
    with open(train_file,'r')as f,open(train_pos,'r')as pos:
        for line in f:
            line =line.strip()
            pos_line =pos.readline().strip()
            line =(line+' ')* len(pos_line.split(' '))
            
            train_writer.write(line.strip()+'\n')
    with open(aux_file,'r')as f,open(aux_pos,'r')as pos:
        for line in f:
            line =line.strip()
            pos_line =pos.readline().strip()
            line =(line+' ')* len(pos_line.split(' '))
            aux_writer.write(line.strip()+'\n')
    with open(test_file,'r')as f,open(test_pos,'r')as pos:
        for line in f:
            line =line.strip()
            pos_line =pos.readline().strip()
            line =(line+' ')* len(pos_line.split(' '))
            test_writer.write(line.strip()+'\n')

    train_writer.close()
    aux_writer.close()
    test_writer.close()

if __name__=='__main__':
    ff = './corpus_extracted/'
    root1 = './text_feature/'
    root2 = './keyword_feature/'
    root3 ='./gongxian_feature/'
    types = ['abstract', 'discuss', 'result', 'wiki']
    
    train_index =3
    
    for test_index in (0,1,2):
        for fold_index in range (1,6):
            build_gongxian_features(train_index,test_index,fold_index)
            # root =ff+types[ ]+'_by_'+types[train_index]+'/'+str(fold_index)

            # if os.path.isdir(root):
            #     print  'root exist'
            # else:
            #     os.makedirs(root)

            # file_lists =build_file_path(train_index,test_index,fold_index)
            # extract_features(file_lists)



