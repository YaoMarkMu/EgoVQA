"""Preprocess the data of MSVD-QA."""
import os
import sys
import numpy as np
import random

import pandas as pd
from pandas import Series, DataFrame
import h5py
import pickle as pkl
import glob
import tables
import json
import csv

from util.preprocess import VideoVGGExtractor
from util.preprocess import VideoC3DExtractor
from util.preprocess import prune_embedding


def extract_video_vgg(dataset_path, video_cam, suffix='.jpg'):
    #hf = h5py.File('feats/vgg_feat.hdf5', 'w')
    extract_vgg(dataset_path, video_cam, suffix)

def extract_vgg(dataset_path, video_cam, suffix):
    import tensorflow as tf
    """Extract VGG features."""

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.gpu_options.visible_device_list = '0'

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
        extractor = VideoVGGExtractor(sess)
        for vname in video_cam:
            # 2_M => 2/frames/M
            segs = vname.split('_')
            video_path = dataset_path + '/' + segs[0] + "/frame/" + segs[1]
            frames = sorted(glob.glob(video_path + "/*"+suffix))
            print '[VGG] Process Video %s %d frames' % (vname, len(frames))
            vgg_feat = extractor.extract_by_id(frames)
            vgg_feat = np.concatenate(vgg_feat, axis=0)
            print vgg_feat.shape
            with open('feats/vgg_%s.pkl' % (vname,), 'w') as f:
                pkl.dump({'feat':vgg_feat}, f)


def extract_video_c3d(dataset_path, video_cam, suffix='.jpg'):
    #hf = h5py.File('feats/c3d_feat.hdf5', 'w')
    extract_c3d(dataset_path, video_cam, suffix)


def extract_c3d(dataset_path, video_cam, suffix):
    import tensorflow as tf
    """Extract C3D features."""
    #c3d_features = list()
    # Session config.
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    #sess_config.gpu_options.visible_device_list = '0'

    with tf.Graph().as_default(), tf.Session(config=sess_config) as sess:
        extractor = VideoC3DExtractor(sess)
        for vname in video_cam:
            segs = vname.split('_')
            video_path = dataset_path + '/' + segs[0] + "/frame/" + segs[1]
            frames = sorted(glob.glob(video_path + "/*"+suffix))
            print('[C3D]', video_path)
            c3d_feat = extractor.extract_by_id(frames)
            c3d_feat = np.concatenate(c3d_feat, axis=0)
            print c3d_feat.shape
            #print c3d_feat[:,:10]
            with open('feats/c3d_%s.pkl' % (vname,), 'w') as f:
                pkl.dump({'feat':c3d_feat}, f)



'''
Name: answer, Length: 412, dtype: object
     video cam   start     end                                           question                   answer  index                  answer2
0      1.0   D    93.0   105.0                  how many people am I talking with                      two      0                      two
1      1.0   D   110.0   120.0             what is the man in white clothes doing                      sit      1                      sit
2      1.0   D   142.0   160.0             what is the man in white clothes doing                    drink      2                    drink
3      1.0   D   223.0   235.0                      what are the two people doing                      sit      3                      sit
'''

def create_qa_encode(csv_path, vocab):
    """Encode question/answer for generate batch faster.

    In train split, remove answers not in answer set and convert question and answer
    to one hot encoding. In val and test split, only convert question to one hot encoding.
    """
    li = []
    dtypes = [np.int32, str, np.int32, np.int32, str, str, str]

    for csv_file in sorted(os.listdir(csv_path)):
        #print csv_file

        if not csv_file.endswith('.csv'):
            continue
        train_qa = pd.read_csv(os.path.join(csv_path, csv_file)) #, dtype=dtypes)
        #print train_qa

        li.append(train_qa)

    df = pd.concat(li, axis=0, ignore_index=True)
    #df['index'] = range(len(df))
    #df.set_index('index')
    answer_list = df['answer'].apply(str)
    question_list = df['question'].apply(str)
    #print df[df['question'].isnull()], "=="


    num_cand = 5
    candidate_answer = [[] for _ in range(num_cand)]
    correct_label = []
    numerical_labels = ['zero', 'one', 'two', 'three', 'four']

    # preprocess answer pool
    what_doing_pool = []
    what_other_pool = []
    what_color_pool = []
    who_pool = []
    where_pool = []

    for q,a in zip(question_list, answer_list):
        print q, a
        if q.startswith('what'):
            if 'doing' in q:
                what_doing_pool.append(a)
            elif 'color' in q:
                what_color_pool.append(a)
            else:
                what_other_pool.append(a)
        elif q.startswith('who'):
            who_pool.append(a)
        elif q.startswith('where'):
            where_pool.append(a)

    what_other_pool = list(set(what_other_pool))
    what_doing_pool = list(set(what_doing_pool))
    what_color_pool = list(set(what_color_pool))
    who_pool = list(set(who_pool))
    where_pool = list(set(where_pool))

    print what_color_pool
    print
    print what_other_pool
    print
    print what_doing_pool
    print
    print who_pool
    print
    print where_pool
    print
    all_pool = list(set(answer_list))

    num_count = 0
    num_what_doing = 0
    num_what_color = 0
    num_what_obj = 0
    num_where = 0
    num_who = 0
    num_other = 0

    # gather what .. doing
    for q,a in zip(question_list, answer_list):

        if q.startswith('how many'):
            #print q, '==', a
            assert a in numerical_labels
            lb = numerical_labels.index(a)
            for ca,nl in zip(candidate_answer,numerical_labels):
                ca.append(nl)
            correct_label.append(lb)
            #print q,a,lb
            num_count += 1

        else:

            if q.startswith('what'):
                if 'doing' in q:
                    pool = what_doing_pool
                    num_what_doing += 1
                elif 'color' in q:
                    pool = what_color_pool
                    num_what_color += 1
                else:
                    pool = what_other_pool
                    num_what_obj += 1

            elif q.startswith('who'):
                pool = who_pool
                num_who += 1
            else:
                pool = all_pool
                if q.startswith('where'):
                    num_where += 1
                else:
                    num_other += 1

            temp = [a]
            #words = [set(a.split())]
            while len(temp)<num_cand:
                a2 = random.choice(pool)
                # a2w = set(a2.split())
                # print a2w
                # print [ (a2w | cw) for cw in words]
                # print any([ (a2w | cw) for cw in words])
                # print
                if (a2 not in temp): # and not any([ (a2w | cw) for cw in words]):
                    #words.append(a2w)
                    temp.append(a2)
            random.shuffle(temp)
            lb = temp.index(a)
            for ca, tp in zip(candidate_answer, temp):
                ca.append(tp)
            correct_label.append(lb)
            #print temp, q, a, lb
            #print


    print 'num_count', num_count
    print 'num_what_doing', num_what_doing
    print 'num_what_color', num_what_color
    print 'num_what_obj', num_what_obj
    print 'num_where', num_where
    print 'num_who', num_who
    print 'num_other', num_other

    for j in range(num_cand):
        df['a%d'%(j+1,)] = candidate_answer[j]
    df['label'] = correct_label


    ##### save to CSV #####
    df.to_csv('./qa.csv', index=True)


    def _encode_question(question):
        """Map question to sequence of vocab id. 3999 for word not in vocab."""
        question_id = ''
        words = str(question).rstrip('?').split()
        #print words
        #print 'am' in vocab.values
        #print 'I' in vocab.values

        for word in words:
            word = word.lower()
            if word in vocab.values:
                question_id = question_id + \
                    str(vocab[vocab == word].index[0]) + ','
            elif word[-1] == 's' and word[:-1] in vocab.values:
                question_id = question_id + \
                          str(vocab[vocab == word[:-1]].index[0]) + ','
            elif word[-3:] == 'ing' and word[:-3] in vocab.values:
                question_id = question_id + \
                          str(vocab[vocab == word[:-3]].index[0]) + ','
            else:
                question_id = question_id + '3999' + ','
                print word, question
        return question_id.rstrip(',')

    print df[df['question'].isnull()], "=="

    df['question_encode'] = df['question'].apply(_encode_question)
    df['video_cam'] = df.apply(lambda row: str(row['video'])+'_'+row['cam'], axis=1)

    for j in range(num_cand):
        df['a%d_encoder'%(j+1,)] = df['a%d'%(j+1,)].apply(_encode_question)

    #
    # df_train = df.sample(frac=0.7)
    # df_val_test = df.drop(df_train.index)
    # df_train = df_train.sample(frac=1)
    # df_test = df_val_test.sample(frac=0.5)
    # df_val = df_val_test.drop(df_test.index)
    # df_val = df_val.sample(frac=1)
    # print 'train/val/test'
    # print df_train.shape, df_val.shape, df_test.shape
    # #print df_val
    # #print df_test
    #
    # df_train.to_json('train.json', 'records')
    # df_val.to_json('val.json', 'records')
    # df_test.to_json('test.json', 'records')

    df.to_json('qa_data.json', 'records')


    # make video names
    video_list = df['video'].tolist()
    cam_list = df['cam'].tolist()
    video_cam = set([])
    for vid, cm in zip(video_list, cam_list):
        vidcam = str(vid) + '_' + str(cm)
        video_cam.add(vidcam)
    return video_cam



def prune_embedding(vocab_path, glove_path, embedding_path):
    """Prune word embedding from pre-trained GloVe.

    For words not included in GloVe, set to average of found embeddings.

    Args:
        vocab_path: vocabulary path.
        glove_path: pre-trained GLoVe word embedding.
        embedding_path: .npy for vocabulary embedding.
    """
    # load GloVe embedding.
    glove = pd.read_csv(
        glove_path, sep=' ', quoting=csv.QUOTE_NONE, header=None)
    glove.set_index(0, inplace=True)
    # load vocabulary.
    vocab = pd.read_csv(vocab_path, header=None)[0]

    embedding = np.zeros([len(vocab), len(glove.columns)], np.float64)
    not_found = []
    for i in range(len(vocab)):
        word = vocab[i]
        if word in glove.index:
            embedding[i] = glove.loc[word]
        else:
            not_found.append(i)
    print('Not found:\n', vocab.iloc[not_found])

    embedding_avg = np.mean(embedding, 0)
    embedding[not_found] = embedding_avg

    np.save(embedding_path, embedding.astype(np.float32))


def main():
    random.seed(123)

    ########## read vocabulary files ##################
    vocab_path = 'vocab.txt'
    vocab = pd.read_csv(vocab_path, header=None)[0]

    ########## creat question answers ##################
    csv_path = 'csv/'
    #video_cam = create_qa_encode(csv_path, vocab)
    #print sorted(video_cam)

    ########## create splits #############
    split_1 = [['1_D', '1_M', '2_F', '2_M', '3_F', '3_M', '4_F', '4_M'],
                  ['5_D', '5_M', '6_D', '6_M'], ['7_D', '7_M', '8_M', '8_X']]

    split_2 = [['1_D', '1_M', '3_F', '3_M', '4_F', '4_M', '7_D', '7_M'],
               ['2_F', '2_M', '8_M', '8_X'], ['5_D', '5_M','6_D', '6_M']]

    split_3 = [['1_D', '1_M', '5_D', '5_M', '6_D', '6_M', '8_M', '8_X'],
               ['4_F', '4_M', '7_D', '7_M'], ['2_F', '2_M', '3_F', '3_M']]

    splits = [split_1, split_2, split_3]
    with open('data_split.json', 'w') as f:
        json.dump(splits, f)

    ########## extract features ##################
    # video_cam = ['1_D', '1_M', '2_F', '2_M', '3_F', '3_M', '4_F', '4_M',
    #              '5_D', '5_M', '6_D', '6_M', '7_D', '7_M', '8_M', '8_X']
    # suffix=".jpg"

    #video_cam = ['9_F', '9_X']
    #suffix = ".png"

    # dataset_path = '/home/fan6/Program/deepview'
    # extract_video_vgg(dataset_path, video_cam, suffix)
    # extract_video_c3d(dataset_path, video_cam, suffix)


    ########## create embeddings  ##################
    #prune_embedding('vocab.txt','util/glove.6B.300d.txt','word_embedding.npy')



if __name__ == '__main__':
    main()
