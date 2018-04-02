from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from tensorflow.contrib import learn
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import ast
import h5py

stopWords = stopwords.words('english')
stemmer = SnowballStemmer('english')

def preprocess(s):
    #lowercase and split to words
    words = word_tokenize(s.strip().lower())
    #remove stop words and do stemming
    words = [stemmer.stem(w) for w in words if not w in stopWords]
    return ' '.join(words)

def loadTrainingData(filename, labels_filename, vocab_filename):
    with h5py.File(filename, "r") as f:
        x = f['trainingFeatures'].value
        y = f['trainingLabels'].value
        f.close()
    labels_dict = json.loads(open(labels_filename).read())
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_filename)
    return x, y, labels_dict, vocab_processor

def loadAndPrepareDataForPredict(model_filename, labels_filename, vocab_filename):
    labels_dict = json.loads(open(labels_filename).read())
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_filename)
    return model, label_dict, vocab_processor

def train_input_fn(features, labels, batch_size, mode=tf.estimator.ModeKeys.EVAL):
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False
    dataset = tf.data.Dataset.from_tensor_slices(({'x': features}, labels))
    if shuffle:
        dataset = dataset.shuffle(features.shape[0])
    dataset = dataset.batch(batch_size)
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def serving_input_fn():
    receiver_tensor = {
      'x': tf.placeholder(tf.int64, [None, 50]),
    }
    features = {
      key: tensor
      for key, tensor in receiver_tensor.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensor)
