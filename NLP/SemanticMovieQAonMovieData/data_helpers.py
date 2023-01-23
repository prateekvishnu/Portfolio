import numpy as np
import re
import itertools
from collections import Counter
import pandas as pd

import sent2vec
sent2vec_model = sent2vec.Sent2vecModel()
sent2vec_model.load_model('torontobooks_unigrams.bin')


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels():
    # Load data from files
    df = pd.read_csv("data/train20.csv")
    selected = ['Category', 'Descript']
    df = df.dropna(axis=0, how='any', subset=selected)
    df = df.reindex(np.random.permutation(df.index))
    labels = sorted(list(set(df[selected[0]].tolist())))
    

    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    x_raw= df[selected[1]].apply(lambda x: clean_str(x).split(' ')).tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
    
    sent_raw = df[selected[1]].apply(lambda x: sent2vec_model.embed_sentence(clean_str(x))).tolist()
    sent_raw = np.array(sent_raw)

    return [x_raw, y_raw, num_labels, labels, sent_raw]


def pad_sentences(sentences, padding_word="<PAD/>", testStringLength = None):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    if testStringLength is not None:
        sequence_length = testStringLength
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    word_counts['UNKNOWN_WORD'] = 1
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data for the MR dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels, num_labels, actual_labels, sent_raw = load_data_and_labels()
    
    labels_temp = range(num_labels)
    labels_dict = zip(actual_labels, labels_temp)
    labels_dict = set(labels_dict)
    labels_dict = {x[1]: x[0] for i, x in enumerate(labels_dict)}

    
    sentences_padded = pad_sentences(sentences) 
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv, num_labels, labels_dict, sent_raw]

def transform_testdata(test_strs):

    test_strs = [clean_str(sent) for sent in test_strs]
    test_strs = [s.split(" ") for s in test_strs]
    
    test_strs_padded = pad_sentences(test_strs)
    x = np.array([[vocabulary[word] for word in sentence] for sentence in test_strs_padded])
    return x


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
