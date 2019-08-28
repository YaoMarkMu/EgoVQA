"""Preprocess the data for model."""
import os
import inspect
import csv

import numpy as np
from PIL import Image
import skvideo.io
import scipy
import tensorflow as tf
import pandas as pd

from .vgg16 import Vgg16
from .c3d import c3d


class VideoVGGExtractor(object):
    """Select uniformly distributed frames and extract its VGG feature."""

    def __init__(self, sess):
        """Load VGG model.

        Args:
            frame_num: number of frames per video.
            sess: tf.Session()
        """
        self.inputs = tf.placeholder(tf.float32, [1, 224, 224, 3])
        self.vgg16 = Vgg16()
        self.vgg16.build(self.inputs)
        self.sess = sess

    def extract_by_id(self, frame_ids):
        """Get VGG fc7 activations as representation for video.

        Args:
            path: Path of video.
        Returns:
            feature: [batch_size, 4096]
        """
        features = []
        for ix, fpath in enumerate(frame_ids):
            if ix % 100 == 0:
                print ix, len(frame_ids)
            img = Image.open(fpath).resize((224, 224), Image.BILINEAR)
            frame_data = np.array(img)
            feature = self.sess.run(
                self.vgg16.relu7, feed_dict={self.inputs: [frame_data]})
            features.append(feature)

        return features

class VideoC3DExtractor(object):
    """Select uniformly distributed clips and extract its C3D feature."""

    def __init__(self, sess):
        """Load C3D model."""
        self.inputs = tf.placeholder(
            tf.float32, [1, 16, 112, 112, 3])
        _, self.c3d_features = c3d(self.inputs, 1, 1)
        saver = tf.train.Saver()
        path = inspect.getfile(VideoC3DExtractor)
        path = os.path.abspath(os.path.join(path, os.pardir))
        saver.restore(sess, os.path.join(
            path, 'sports1m_finetuning_ucf101.model'))
        self.mean = np.load(os.path.join(path, 'crop_mean.npy'))
        self.sess = sess

    def extract_by_id(self, frame_ids):
        """Select self.batch_size clips for video. Each clip has 16 frames.

        Args:
            path: Path of video.
        Returns:
            clips: list of clips.
        """
        features = []
        for ix, fpath in enumerate(frame_ids):
            if ix % 100 == 0:
                print ix, len(frame_ids)
            #if ix>100: break
            clip_start = ix - 8
            clip_end = ix + 8
            if clip_start < 0:
                clip_end = clip_end - clip_start
                clip_start = 0
            if clip_end > len(frame_ids):
                clip_start = clip_start - (clip_end - len(frame_ids))
                clip_end = len(frame_ids)

            #print ix, fpath
            new_clip = []
            for j in range(clip_start, clip_end):
                img = Image.open(frame_ids[j]).resize((112, 112), Image.BILINEAR)
                frame_data = np.array(img) * 1.0
                frame_data -= self.mean[j-clip_start]
                new_clip.append(frame_data)
                #print frame_ids[j]
            #print '---'

            feature = self.sess.run(
                self.c3d_features, feed_dict={self.inputs: [new_clip]})
            #print feature.shape
            features.append(feature)
        return features


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
