"""Generate data batch."""
import os

import pandas as pd
import numpy as np
import pickle as pkl
import tables

class EgoVQA(object):
    """Use bucketing and padding to generate data batch.

    All questions are divided into 4 buckets and each buckets has its length.
    """

    def __init__(self, train_batch_size, data_dir, feats_dir, video_cam):
        """Load video feature, question answer, vocabulary, answer set.

        Note:
            The `train_batch_size` only impacts train. val and test batch size is 1.
        """
        self.train_batch_size = train_batch_size

        # self.video_feature = tables.open_file(
        #     os.path.join(preprocess_dir, 'video_feature_20.h5'))
        self.video_feature = {}

        vgg_dict = {}
        c3d_dict = {}

        assert len(video_cam) == 3 # train/val/tet

        total_qa = pd.read_json(os.path.join(data_dir, 'qa_data.json'))
        # answer_set = list(set(total_qa['answer'].tolist()))
        # print len(answer_set)
        # pd.DataFrame(answer_set).to_csv('./data/answer_set.txt', header=False, index=False)
        # return


        for ix, data_group in enumerate(video_cam):

            if ix==0:
                self.train_qa = total_qa[total_qa['video_cam'].isin(data_group)]
            elif ix==1:
                self.val_qa = total_qa[total_qa['video_cam'].isin(data_group)]
            else:
                self.test_qa = total_qa[total_qa['video_cam'].isin(data_group)]

            for vc in data_group:
                with open(os.path.join(feats_dir, 'vgg_%s.pkl' % (vc,)), 'r') as f:
                    feat1 = pkl.load(f)
                    feat1 = feat1['feat']
                    vgg_dict[vc] =feat1

                with open(os.path.join(feats_dir, 'c3d_%s.pkl' % (vc,)), 'r') as f:
                    feat1 = pkl.load(f)
                    feat1 = feat1['feat']
                    c3d_dict[vc] = feat1

        self.video_feature['vgg'] = vgg_dict
        self.video_feature['c3d'] = c3d_dict


        # contains encode, for fast generate batch by saving encode time
        # self.train_qa = pd.read_json(
        #     os.path.join(data_dir, 'train.json'))
        # self.val_qa = pd.read_json(
        #     os.path.join(data_dir, 'val.json'))
        # self.test_qa = pd.read_json(
        #     os.path.join(data_dir, 'test.json'))

        print 'train/val/test:',
        print self.train_qa.shape,
        print self.val_qa.shape,
        print self.test_qa.shape
        print len(self.train_qa) + len(self.val_qa) + len(self.test_qa)


        
        # init train batch setting
        self.train_qa['question_length'] = self.train_qa.apply(
            lambda row: len(row['question_encode']), axis=1)
        self.train_buckets = [
            self.train_qa[(self.train_qa['question_length'] >= 0)
                          & (self.train_qa['question_length'] <= 16)],
            self.train_qa[(self.train_qa['question_length'] >= 17)
                          & (self.train_qa['question_length'] <= 21)],
            self.train_qa[(self.train_qa['question_length'] >= 22)
                          & (self.train_qa['question_length'] <= 30)],
            self.train_qa[(self.train_qa['question_length'] >= 31)
                          & (self.train_qa['question_length'] <= 50)]
        ]

        print 'question buckets: '
        for tt in self.train_buckets:
            print len(tt),
        print

        self.current_bucket = 0
        self.train_batch_length = [16, 21, 30, 50]
        self.train_batch_idx = [0, 0, 0, 0]
        self.train_batch_total = [
            len(self.train_buckets[0]) // train_batch_size,
            len(self.train_buckets[1]) // train_batch_size,
            len(self.train_buckets[2]) // train_batch_size,
            len(self.train_buckets[3]) // train_batch_size,
        ]

        self.num_bucket = len(self.train_batch_total)

        # upset the arise of questions of same video
        for i in range(self.num_bucket):
            self.train_buckets[i] = self.train_buckets[i].sample(frac=1)
        self.has_train_batch = True

        # init val example setting
        self.val_example_total = len(self.val_qa)
        self.val_example_idx = 0
        self.has_val_example = True

        # init test example setting
        self.test_example_total = len(self.test_qa)
        self.test_example_idx = 0
        self.has_test_example = True

    def reset_train(self):
        """Reset train batch setting."""
        # random
        for i in range(self.num_bucket):
            self.train_buckets[i] = self.train_buckets[i].sample(frac=1)
        self.current_bucket = 0
        self.train_batch_idx = [0, 0, 0, 0]
        self.has_train_batch = True

    def reset_val(self):
        """Reset val batch setting."""
        self.val_example_idx = 0
        self.has_val_example = True

    def reset_test(self):
        """Reset train batch setting."""
        self.test_example_idx = 0
        self.has_test_example = True

    def get_train_batch(self):
        """Get [train_batch_size] examples as one train batch. Both question and answer
        are converted to int. The data batch is selected from short buckets to long buckets."""
        vgg_batch = []
        c3d_batch = []
        question_batch = []
        answer_batch = []

        bucket = self.train_buckets[self.current_bucket]
        start = self.train_batch_idx[
            self.current_bucket] * self.train_batch_size
        end = start + self.train_batch_size

        question_text = bucket.iloc[start:end]['question'].values
        question_encode = bucket.iloc[start:end]['question_encode'].values
        answer_batch = bucket.iloc[start:end]['label'].values
        video_ids = bucket.iloc[start:end]['video'].values
        cam_ids = bucket.iloc[start:end]['cam'].values
        start_ids = bucket.iloc[start:end]['start'].values
        end_ids = bucket.iloc[start:end]['end'].values
        batch_length = self.train_batch_length[self.current_bucket]
        question_lengths = []

        answer_texts = []
        answer_encoders = []
        for j in range(5):
            as_j = bucket.iloc[start:end]['a%d' % (j+1,)].values
            answer_texts.append(as_j)
            enc_as_j = bucket.iloc[start:end]['a%d_encoder' % (j+1,)].values
            answer_encoders.append(enc_as_j)
        #print answer_encoders
        #print video_ids, cam_ids, start_ids, end_ids


        for i in range(self.train_batch_size):
            question_as = []
            qid = [int(x) for x in question_encode[i].split(',')]
            #print qid, question_text[i]
            # five candidates
            enc_as = []
            max_enc_len = 0
            for j in range(5):
                enc_a = [int(x) for x in answer_encoders[j][i].split(',')]
                max_enc_len = max(max_enc_len, len(enc_a))
                enc_as.append(enc_a)
            max_enc_len += len(qid)

            for j in range(5):
                qid2 = qid + enc_as[j]
                qid2 = np.pad(qid2, (0, max_enc_len - len(qid2)), 'constant')
                question_as.append(qid2)
                #print qid2, answer_texts[j][i]

            question_lengths.append(max_enc_len)
            question_batch.append(question_as)
            vd_cam = '%d_%s' % (video_ids[i], cam_ids[i])

            vgg_feat = self.video_feature['vgg'][vd_cam][start_ids[i]:end_ids[i]+1]
            c3d_feat = self.video_feature['c3d'][vd_cam][start_ids[i]:end_ids[i]+1]
            #print vgg_feat.shape, c3d_feat.shape
            vgg_batch.append(vgg_feat)
            c3d_batch.append(c3d_feat)


        self.train_batch_idx[self.current_bucket] += 1
        # if current bucket is ran out, use next bucket.
        if self.train_batch_idx[self.current_bucket] == self.train_batch_total[self.current_bucket]:
            self.current_bucket += 1

        if self.current_bucket == len(self.train_batch_total):
            self.has_train_batch = False

        return vgg_batch, c3d_batch, question_batch, answer_batch, question_lengths

    def get_val_example(self):
        """Get one val example. Only question is converted to int."""
        question_text = self.val_qa.iloc[self.val_example_idx]['question']
        question_encode = self.val_qa.iloc[self.val_example_idx]['question_encode']
        answer = self.val_qa.iloc[self.val_example_idx]['label']
        video_id = self.val_qa.iloc[self.val_example_idx]['video']
        cam_id = self.val_qa.iloc[self.val_example_idx]['cam']
        start_id = self.val_qa.iloc[self.val_example_idx]['start']
        end_id = self.val_qa.iloc[self.val_example_idx]['end']

        qid = [int(x) for x in question_encode.split(',')]
        answer_texts = []
        answer_encoders = []


        for j in range(5):
            as_j = self.val_qa.iloc[self.val_example_idx]['a%d' % (j+1,)]
            answer_texts.append(as_j)
            enc_as_j = self.val_qa.iloc[self.val_example_idx]['a%d_encoder' % (j+1,)]
            answer_encoders.append(enc_as_j)

        enc_as = []
        max_enc_len = 0
        for j in range(5):
            enc_a = [int(x) for x in answer_encoders[j].split(',')]
            max_enc_len = max(max_enc_len, len(enc_a))
            enc_as.append(enc_a)
        max_enc_len += len(qid)

        question_as = []
        for j in range(5):
            qid2 = qid + enc_as[j]
            qid2 = np.pad(qid2, (0, max_enc_len - len(qid2)), 'constant')
            question_as.append(qid2)

        vd_cam = '%d_%s' % (video_id, cam_id)
        vgg_feat = self.video_feature['vgg'][vd_cam][start_id:end_id + 1]
        c3d_feat = self.video_feature['c3d'][vd_cam][start_id:end_id + 1]

        self.val_example_idx += 1
        if self.val_example_idx == self.val_example_total:
            self.has_val_example = False

        return [vgg_feat], [c3d_feat], [question_as], [answer], [max_enc_len]

    def get_test_example(self):
        """Get one test example. Only question is converted to int."""
        # example_id = self.test_qa.iloc[self.test_example_idx]['id']
        # question_encode = self.test_qa.iloc[
        #     self.test_example_idx]['question_encode']
        # video_id = self.test_qa.iloc[self.test_example_idx]['video_id']
        # answer = self.test_qa.iloc[self.test_example_idx]['answer']
        # question_lengths = []
        #
        # question = [int(x) for x in question_encode.split(',')]
        # question_lengths.append(len(question))
        # vgg = self.video_feature.root.vgg[video_id - 1]
        # c3d = self.video_feature.root.c3d[video_id - 1]
        #
        # self.test_example_idx += 1
        # if self.test_example_idx == self.test_example_total:
        #     self.has_test_example = False

        #return [vgg], [c3d], [question], answer, question_lengths

        question_text = self.test_qa.iloc[self.test_example_idx]['question']
        question_encode = self.test_qa.iloc[self.test_example_idx]['question_encode']
        answer = self.test_qa.iloc[self.test_example_idx]['label']
        video_id = self.test_qa.iloc[self.test_example_idx]['video']
        cam_id = self.test_qa.iloc[self.test_example_idx]['cam']
        start_id = self.test_qa.iloc[self.test_example_idx]['start']
        end_id = self.test_qa.iloc[self.test_example_idx]['end']
        answer_text = self.test_qa.iloc[self.test_example_idx]['answer']

        qid = [int(x) for x in question_encode.split(',')]
        answer_encoders = []

        for j in range(5):
            as_j = self.test_qa.iloc[self.test_example_idx]['a%d' % (j + 1,)]
            enc_as_j = self.test_qa.iloc[self.test_example_idx]['a%d_encoder' % (j + 1,)]
            answer_encoders.append(enc_as_j)

        enc_as = []
        max_enc_len = 0
        for j in range(5):
            enc_a = [int(x) for x in answer_encoders[j].split(',')]
            max_enc_len = max(max_enc_len, len(enc_a))
            enc_as.append(enc_a)
        max_enc_len += len(qid)

        question_as = []
        for j in range(5):
            qid2 = qid + enc_as[j]
            qid2 = np.pad(qid2, (0, max_enc_len - len(qid2)), 'constant')
            question_as.append(qid2)

        vd_cam = '%d_%s' % (video_id, cam_id)
        vgg_feat = self.video_feature['vgg'][vd_cam][start_id:end_id + 1]
        c3d_feat = self.video_feature['c3d'][vd_cam][start_id:end_id + 1]

        self.test_example_idx += 1
        if self.test_example_idx == self.test_example_total:
            self.has_test_example = False



        return [vgg_feat], [c3d_feat], [question_as], [answer], [max_enc_len], question_text, answer_text


class ZHVQA(object):
    """Use bucketing and padding to generate data batch.

    All questions are divided into 4 buckets and each buckets has its length.
    """

    def __init__(self, train_batch_size, preprocess_dir):
        """Load video feature, question answer, vocabulary, answer set.

        Note:
            The `train_batch_size` only impacts train. val and test batch size is 1.
        """
        self.train_batch_size = train_batch_size

        self.video_feature = tables.open_file(
            os.path.join(preprocess_dir, 'video_feature_20.h5'))

        # contains encode, for fast generate batch by saving encode time
        self.train_qa = pd.read_json(
            os.path.join(preprocess_dir, 'train_qa_encode.json'))
        self.val_qa = pd.read_json(
            os.path.join(preprocess_dir, 'val_qa_encode.json'))
        self.test_qa = pd.read_json(
            os.path.join(preprocess_dir, 'test_qa_encode.json'))

        # init train batch setting
        self.train_qa['question_length'] = self.train_qa.apply(
            lambda row: len(row['Question'].split()), axis=1)
        self.train_buckets = [
            self.train_qa[(self.train_qa['question_length'] >= 2)
                          & (self.train_qa['question_length'] <= 6 - 1)],
            self.train_qa[(self.train_qa['question_length'] >= 7 - 1)
                          & (self.train_qa['question_length'] <= 11 - 1)],
            self.train_qa[(self.train_qa['question_length'] >= 12 - 1)
                          & (self.train_qa['question_length'] <= 16 - 1)],
            self.train_qa[(self.train_qa['question_length'] >= 17 - 1)
                          & (self.train_qa['question_length'] <= 21 - 1)]
        ]

        print 'bucket lengths:',
        for t in self.train_buckets:
            print len(t),
        print

        self.current_bucket = 0
        self.train_batch_length = [6, 11, 16, 21]
        self.train_batch_idx = [0, 0, 0, 0]
        self.train_batch_total = [
            len(self.train_buckets[0]) // train_batch_size,
            len(self.train_buckets[1]) // train_batch_size,
            len(self.train_buckets[2]) // train_batch_size,
            len(self.train_buckets[3]) // train_batch_size,
        ]
        # upset the arise of questions of same video
        for i in range(4):
            self.train_buckets[i] = self.train_buckets[i].sample(frac=1)
        self.has_train_batch = True

        # init val example setting
        self.val_example_total = len(self.val_qa)
        self.val_example_idx = 0
        self.has_val_example = True

        # init test example setting
        self.test_example_total = len(self.test_qa)
        self.test_example_idx = 0
        self.has_test_example = True

    def reset_train(self):
        """Reset train batch setting."""
        # random
        for i in range(4):
            self.train_buckets[i] = self.train_buckets[i].sample(frac=1)
        self.current_bucket = 0
        self.train_batch_idx = [0, 0, 0, 0]
        self.has_train_batch = True

    def reset_val(self):
        """Reset val batch setting."""
        self.val_example_idx = 0
        self.has_val_example = True

    def reset_test(self):
        """Reset train batch setting."""
        self.test_example_idx = 0
        self.has_test_example = True

    def get_train_batch(self):
        """Get [train_batch_size] examples as one train batch. Both question and answer
        are converted to int. The data batch is selected from short buckets to long buckets."""
        vgg_batch = []
        c3d_batch = []
        question_batch = []
        answer_batch = []

        bucket = self.train_buckets[self.current_bucket]
        start = self.train_batch_idx[
                    self.current_bucket] * self.train_batch_size
        end = start + self.train_batch_size

        question_encode = bucket.iloc[start:end]['question_encode'].values
        candidate_encode = bucket.iloc[start:end]['candidate_encode'].values
        answer_batch = bucket.iloc[start:end]['answer_encode'].values
        video_ids = bucket.iloc[start:end]['video_id'].values
        batch_length = self.train_batch_length[self.current_bucket]
        question_lengths = []

        for i in range(self.train_batch_size):
            tmp = []
            qid = [int(x) for x in question_encode[i].split(',')]

            for j in range(len(candidate_encode[i])):
                cid = candidate_encode[i][j]
                qid2 = qid + [cid]
                qid2 = np.pad(qid2, (0, batch_length - len(qid2)), 'constant')
                tmp.append(qid2)

            question_lengths.append(len(qid) + 1)
            question_batch.append(tmp)
            vgg_batch.append(self.video_feature.root.vgg[video_ids[i] - 1])
            c3d_batch.append(self.video_feature.root.c3d[video_ids[i] - 1])

        # print question_batch
        self.train_batch_idx[self.current_bucket] += 1
        # if current bucket is ran out, use next bucket.
        if self.train_batch_idx[self.current_bucket] == self.train_batch_total[self.current_bucket]:
            self.current_bucket += 1

        if self.current_bucket == len(self.train_batch_total):
            self.has_train_batch = False

        return vgg_batch, c3d_batch, question_batch, answer_batch, question_lengths

    def get_val_example(self):
        """Get one val example. Only question is converted to int."""
        question_encode = self.val_qa.iloc[self.val_example_idx]['question_encode']
        candidate_encode = self.val_qa.iloc[self.val_example_idx]['candidate_encode']
        video_id = self.val_qa.iloc[self.val_example_idx]['video_id']
        answer = self.val_qa.iloc[self.val_example_idx]['answer_encode']
        question_lengths = []

        question = [int(x) for x in question_encode.split(',')]

        question_batch = []
        for j in range(len(candidate_encode)):
            cid = candidate_encode[j]
            qid2 = question + [cid]
            question_batch.append(qid2)

        question_lengths.append(len(question) + 1)
        vgg = self.video_feature.root.vgg[video_id - 1]
        c3d = self.video_feature.root.c3d[video_id - 1]

        self.val_example_idx += 1
        if self.val_example_idx == self.val_example_total:
            self.has_val_example = False

        return [vgg], [c3d], [question_batch], [answer], question_lengths

    def get_test_example(self):
        """Get one test example. Only question is converted to int."""
        example_type = self.test_qa.iloc[self.test_example_idx]['Type']
        question_encode = self.test_qa.iloc[self.test_example_idx]['question_encode']
        candidate_encode = self.test_qa.iloc[self.test_example_idx]['candidate_encode']
        video_id = self.test_qa.iloc[self.test_example_idx]['video_id']
        answer = self.test_qa.iloc[self.test_example_idx]['answer_encode']
        question_lengths = []

        question = [int(x) for x in question_encode.split(',')]

        question_batch = []
        for j in range(len(candidate_encode)):
            cid = candidate_encode[j]
            qid2 = question + [cid]
            question_batch.append(qid2)

        question_lengths.append(len(question) + 1)
        vgg = self.video_feature.root.vgg[video_id - 1]
        c3d = self.video_feature.root.c3d[video_id - 1]

        self.test_example_idx += 1
        if self.test_example_idx == self.test_example_total:
            self.has_test_example = False

        return [vgg], [c3d], [question_batch], [answer], question_lengths, example_type




if __name__ == "__main__":
    video_cam = ['1_D', '1_M', '2_F', '2_M', '3_F', '3_M', '4_F', '4_M',
                 '5_D', '5_M', '6_D', '6_M', '7_D', '7_M', '8_M', '8_X']
    train_dataset = EgoVQA(4, './data', './data/feats', video_cam)
    #train_dataset.get_train_batch()

    for _ in range(10):
        train_dataset.get_val_example()