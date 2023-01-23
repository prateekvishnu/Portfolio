#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import predict_helper
import pandas as pd
import time

queries_df = pd.read_csv('./data/test.txt', sep="*", delimiter="*", header = None)
test_queries = queries_df[1].tolist()
test_labels = queries_df[0].tolist()
print (test_queries)

#predict
model, model_params = predict_helper.load_model_and_params()
start = time.time()
test_predictions = predict_helper.predict_movie(test_queries, model, model_params['labels_dict'], model_params['vocabulary'], model_params['event_voc'], model_params['ners_voc'], multiple = True)
end = time.time()

delta = int(end-start)
print('Time took to predict: '+ str(delta) + ' seconds')

pred_counter = 0
for i in range(len(test_labels)):
    label = test_labels[i]
    pred_label = test_predictions[i]
    if label == pred_label:
        pred_counter += 1
    print("Actual: "+ label+", Predicted: "+pred_label+"\n")

print("Prediction Accuracy: "+str((pred_counter/len(test_labels))*100))

"""
import predict_helper

model, model_params = predict_helper.load_model_and_params()
test_queries = ["Ripley is the only survivor from an alien attack on his cargo ship Nostromo. She has a pet cat and the company starts a formal hearing on the ship's destruction. Her daughter is dead while she was floating in space. At the end, Ripley starts the journey back to earth."]
test_predictions = predict_helper.predict_movie(test_queries, model, model_params['labels_dict'], model_params['vocabulary'], model_params['event_voc'], model_params['ners_voc'], multiple = True)
print (test_predictions)
Aliens(1986)-Synopsis
"""
"""
1st prediction
BeautyandtheBeast(1991)
test_queries = ["A prince gets cursed to be transformed into a hideous beast and his servants into household items by a disguised old beggar."]
2nd position
AnnieHall(1977)-Synopsis
test_queries = ["A film about a comedian who falls in love with a singer and photographer."]


"""