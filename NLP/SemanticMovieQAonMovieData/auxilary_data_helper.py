#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#import jnius_config

#jnius_config.set_options('-Xms128m', '-Xmx512m')
import os
import config
os.environ['CLASSPATH'] = config.k_parser_path +':'+ config.extractor_path + ':'+ config.eventextractor_path 

from jnius import autoclass
from tokenizeData import Lemmatizer
import numpy as np
import re
import itertools
from collections import Counter
import io
import pandas as pd
import pickle
import os.path
EventExtractor = autoclass('Extractors.EventExtraction')
NerExtractor = autoclass('Extractors.NamedEntityRecognition')
Iterator = autoclass('java.util.Iterator')


def extract_ners(plot):
    ner_extractor = NerExtractor()
    extracted_entities = ner_extractor.getNamedEntities(plot)

    Iterator = extracted_entities.iterator()
    
    entities = []
    while Iterator.hasNext():
        entities.append(Iterator.next())
    return entities


def extract_events(sentences):
    event_extractor = EventExtractor()
    extracted_sentences = event_extractor.eventExtractionEngine(sentences)

    length = extracted_sentences.size()  
    events = []
    for i in range(length):
        tags = extracted_sentences.get(i).split(",")
        event = Lemmatizer(tags[1].split('-')[0])
        events.append(event)
    return events
        
   
def load_event_data():
    events_all_summaries= []
    
    summaries, labels, num_labels, actual_labels = load_event_data_and_labels()
    
    labels_temp = range(num_labels)
    labels_dict = zip(actual_labels, labels_temp)
    labels_dict = set(labels_dict)
    labels_dict = {x[1]: x[0] for i, x in enumerate(labels_dict)}
    
    if(os.path.exists('./auxilary_data/events')):
        print("Reading from existing events")
        with open ('./auxilary_data/events', 'rb') as fp:
            events_all_summaries = pickle.load(fp)
    else:
        for i in range(len(summaries)):
            summary = summaries[i]
            sentences = summary.split('.')
            events_in_summary = filter(None, extract_events(sentences))
            events_all_summaries.append(list(set(events_in_summary)))
        with open('./auxilary_data/events', 'wb') as fp:
            pickle.dump(events_all_summaries, fp)
            
    events_all_summaries = pad_sentences(events_all_summaries)
    vocabulary, vocabulary_inv = build_vocab(events_all_summaries, vocab_type = 'event')
    events_onehot = build_input_data(events_all_summaries, vocabulary, vocab_type ='event')
    
    return [events_onehot, vocabulary, vocabulary_inv, labels_dict]

def load_ners_data():
    ners_all_summaries = [] 
    summaries, labels, num_labels, actual_labels = load_event_data_and_labels()
    
    if(os.path.exists('./auxilary_data/ners')):
        print("Reading from existing ners")
        with open ('./auxilary_data/ners', 'rb') as fp:
            ners_all_summaries = pickle.load(fp)
    else:
        for i in range(len(summaries)):
            summary = summaries[i]
            ners_in_summary = extract_ners(summary)
            ners_all_summaries.append(ners_in_summary)
        with open('./auxilary_data/ners', 'wb') as fp:
            pickle.dump(ners_all_summaries, fp)
            
    ners_all_summaries = pad_sentences(ners_all_summaries)
    vocabulary, vocabulary_inv = build_vocab(ners_all_summaries, vocab_type = 'ner')
    ners_onehot = build_input_data(ners_all_summaries, vocabulary, vocab_type='ner')
    
    return [ners_onehot, vocabulary, vocabulary_inv]


def build_input_data(events_all_summaries, vocabulary, synonyms = None, is_test = False, vocab_type = None):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = []
    if is_test:
        for event_summary in events_all_summaries:
            single_summary = []
            for word in event_summary:
                if word in vocabulary:
                    single_summary.append(vocabulary[word])
                else:
                    word_synonyms = synonyms[word]
                    added = False
                    for word_syn in word_synonyms:
                        if word_syn in vocabulary:
                            single_summary.append(vocabulary[word_syn])
                            added = True
                            break
                    if not added:
                        single_summary.append(getUnknownWordForVocab(vocabulary, vocab_type))
            x.append(single_summary)
    else:
        x = np.array([[vocabulary[word] if word in vocabulary else getUnknownWordForVocab(vocabulary, vocab_type) for word in event_summary] for event_summary in events_all_summaries])
   
    num_events = len(vocabulary)
    events_onehot_list = np.zeros((len(events_all_summaries), num_events), int)
    for i in range(len(events_onehot_list)):
        one = events_onehot_list[i]
        x1 = x[i]
        one[x1] = 1
    
    return events_onehot_list

def getUnknownWordForVocab(vocabulary, vocab_type):
    if vocab_type == 'event':
        return vocabulary['UNKNOWN_EVENT']
    elif vocab_type == 'ner':
        return vocabulary['UNKNOWN_NER']
    else:
        return vocabulary['<PAD/>']
    
def build_vocab(events_all_summaries, vocab_type = None):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*events_all_summaries))
    if vocab_type == 'event':
        word_counts['UNKNOWN_EVENT'] = 1
    elif vocab_type == 'ner':
        word_counts['UNKNOWN_NER'] = 1
        
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]
      
    
def pad_sentences(events_all_summaries, padding_word="<PAD/>", testStringLength = None):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in events_all_summaries)
    if testStringLength is not None:
        sequence_length = testStringLength
    padded_events = []
    for i in range(len(events_all_summaries)):
        event_summary = events_all_summaries[i]
        num_padding = sequence_length - len(event_summary)
        new_event_summary = event_summary + [padding_word] * num_padding
        padded_events.append(new_event_summary)
    return padded_events
  
    

def load_event_data_and_labels():
    
    df = pd.read_csv("data/trainMovie.csv")
    selected = ['sentiment', 'review']
    labels = sorted(list(set(df[selected[0]].tolist())))
    
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    
    x_raw = df[selected[1]].apply(lambda x: x).tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()
    
    
    return [x_raw, y_raw, num_labels, labels]

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
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
