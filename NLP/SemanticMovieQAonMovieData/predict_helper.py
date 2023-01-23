#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from nltk.corpus import wordnet as wn
import numpy as np
import data_helpers
import auxilary_data_helper
from tokenizeData import Lemmatizer
import itertools
import pickle
from keras.models import load_model
from neuralcoref import Coref

import sent2vec
sent2vec_model = sent2vec.Sent2vecModel()
sent2vec_model.load_model('torontobooks_unigrams.bin')
coref = Coref()

def evaluate_on_test(x_test, y_test, model):
    evaluate = model.evaluate(x_test, y_test)
    print("Evaluated against test dataset: " + evaluate)


def predict_movie(testStrs, model, labels_dict, vocabulary, event_voc, ners_voc, multiple = False):
    #testStrs = ["A young girl rises to fame in Broadway while many other theatre figures are jealous and disgusted. She starts as a general assistant to another bright star of Broadway and slowly she becomes the new star in the town. Another highschool girl offers to help and becomes a maid to Eve."]
    testStrs = get_coreferenced_str(testStrs)
    testStr_vector = transform_testdata(testStrs, vocabulary)
    events_onehot = extract_events_onehot(testStrs, event_voc)
    ners_onehot = extract_ners_onehot(testStrs, ners_voc)
    sent2vec_vector = get_sent2vec_embeds(testStrs)
    
    preds = model.predict(testStr_vector)
    
    test_predictions = []
    
    for i in range(len(preds)):
        print("\n Predictions for Query : {:d}".format(i+1))
        pred = preds[i]
        test_predictions.append(labels_dict[pred.argmax()])
        if(not multiple):
            print("Predicting top movie")
            print(labels_dict[pred.argmax()])
            continue
        print("Predicting top 5 movies: \n")
        pred_dict = {i: x for i, x in enumerate(pred)}
        sorted_pred_dict = sorted(pred_dict, key=pred_dict.get, reverse=True)
        sliced_pred = sorted_pred_dict[:5]
        for r in sliced_pred:
            print(labels_dict[r] + str(pred_dict[r]))
            
    print("\n")
    return test_predictions

def transform_testdata(test_strs, vocabulary):
    test_strs = [Lemmatizer(data_helpers.clean_str(sent)) for sent in test_strs]
    test_strs = [s.split(" ") for s in test_strs]
    
    test_strs_padded = data_helpers.pad_sentences(test_strs, testStringLength = 90)
    x = np.array([[vocabulary[word] if word in vocabulary else vocabulary["UNKNOWN_WORD"] for word in sentence] for sentence in test_strs_padded])
    return x

 
def extract_events_onehot(testStrs, event_voc):
    events_all_test_str = []
    for testStr in testStrs:
        sentences = testStr.split('.')
        events_in_str = filter(None, auxilary_data_helper.extract_events(sentences))
        events_all_test_str.append(list(set(events_in_str)))
        
    events_all_test_str = auxilary_data_helper.pad_sentences(events_all_test_str, testStringLength=1379)
    synonyms = get_synonyms(events_all_test_str)
    return auxilary_data_helper.build_input_data(events_all_test_str, event_voc, synonyms=synonyms, is_test = True, vocab_type='event')
    
def extract_ners_onehot(testStrs, ners_voc):
     ners_all_test_str = []
     for testStr in testStrs:
         ners_in_str = auxilary_data_helper.extract_ners(testStr)
         ners_all_test_str.append(ners_in_str)
     ners_all_test_str = auxilary_data_helper.pad_sentences(ners_all_test_str, testStringLength=1379)
     return auxilary_data_helper.build_input_data(ners_all_test_str, ners_voc, vocab_type='ner')

def get_synonyms(events_all_summaries):
    synonyms ={event: get_synonyms_for_word(event) for event in list(itertools.chain.from_iterable(events_all_summaries))}
    return synonyms        
    
def get_synonyms_for_word(word):
    synonyms = []
    for syn in wn.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms

def get_sent2vec_embeds(testStrs):
    sent2vec_embeds = []
    for testStr in testStrs:
        sent2vec_embeds.append(sent2vec_embed(testStr))
    return np.array(sent2vec_embeds)
    
def sent2vec_embed(sent):
    return sent2vec_model.embed_sentence(auxilary_data_helper.clean_str(sent))

def get_coreferenced_str(testStrs):
    coreferenced_strs = []
    for testStr in testStrs:
        clusters = coref.one_shot_coref(utterances= testStr)
        testStr = coref.get_resolved_utterances()[0]
        coreferenced_strs.append(testStr)
    return coreferenced_strs

def load_model_and_params():
    with open ('./model_parameters/model_params_conv_1', 'rb') as fp:
        model_params = pickle.load(fp)
    model = load_model('./final_models/model_conv_1.h5')
    return model, model_params
    
