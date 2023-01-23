
import numpy as np
import data_helpers
from w2v import train_word2vec
import auxilary_data_helper
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input, MaxPooling1D, Convolution1D, Embedding, Reshape
from keras.layers.merge import Concatenate
from keras.layers.merge import concatenate
from keras.utils import plot_model
import pickle
from sklearn.model_selection import StratifiedKFold

np.random.seed(0)

# Model Hyperparameters
embedding_dim = 50
filter_sizes = (3, 8)
num_filters = 10
dropout_prob = (0.5, 0.8)
hidden_dims = 50

# Training parameters
batch_size = 64
num_epochs = 2
kfold_splits = 2
# Prepossessing parameters
sequence_length = 400
max_words = 5000

# Word2Vec parameters
min_word_count = 1
context = 10
events_seq_length = 300
ners_seq_length = 300
sent2vec_seq_length = 300

def transform_events_input(event_all_summaries, labels_y, labels_dict):
    new_events_all_summaries = np.array([event_all_summaries[label] for label in labels_y])
    return new_events_all_summaries

def load_data():
    #embeding sentences input
    x, y, vocabulary, vocabulary_inv_list, num_labels, labels_dict, sent2vec_raw = data_helpers.load_data()
    vocabulary_inv = {key: value for key, value in enumerate(vocabulary_inv_list)}
    labels_y = y.argmax(axis=1)
    
    #events input
    events_onehot, event_voc, event_voc_inv_list, events_labels_dict = auxilary_data_helper.load_event_data()
    event_voc_inv = {key: value for key, value in enumerate(event_voc_inv_list)}
    event_x = transform_events_input(events_onehot, labels_y, labels_dict)
    
    #ners input
    ners_onehot, ners_voc, ners_voc_inv_list = auxilary_data_helper.load_ners_data()
    ners_voc_inv = {key: value for key, value in enumerate(ners_voc_inv_list)}
    ners_x = transform_events_input(ners_onehot, labels_y, labels_dict)
    
    return x, event_x, ners_x, sent2vec_raw, y, vocabulary, vocabulary_inv, event_voc, event_voc_inv, ners_voc, ners_voc_inv, num_labels, labels_dict

def create_model(x_train, event_x_train, ners_x_train, sent_x_train, y_train, x_test, event_x_test,  ners_x_test, sent_x_test, y_test, vocabulary, vocabulary_inv):
        
    global sequence_length, events_seq_length, ners_seq_length, sent2vec_seq_length
    
    #adjust seq lengths
    if sequence_length != x_test.shape[1]:
        print("Adjusting sequence length for actual size: {:d}".format(x_test.shape[1]))
        sequence_length = x_test.shape[1]
    
    if events_seq_length != event_x_test.shape[1]:
        print("Adjusting event sequence length for actual size: {:d}".format(event_x_test.shape[1]))
        events_seq_length = event_x_test.shape[1]
    
    if ners_seq_length != ners_x_test.shape[1]:
        print("Adjusting ners sequence length for actual size: {:d}".format(ners_x_test.shape[1]))
        ners_seq_length = ners_x_test.shape[1]
        
    if sent2vec_seq_length != sent_x_test.shape[1]:
        print("Adjusting sent2vec sequence length for actual size: {:d}".format(sent_x_test.shape[1]))
        sent2vec_seq_length = sent_x_test.shape[1]
        
    # Prepare embedding layer weights and convert inputs for static model
    embedding_weights = train_word2vec(np.vstack((x_train, x_test)), vocabulary_inv, num_features=embedding_dim,
                                           min_word_count=min_word_count, context=context)
    
    # Build model
    input_shape = (sequence_length,)
    model_input = Input(shape=input_shape, name="Input_Layer")
    embed = Embedding(len(vocabulary_inv), embedding_dim, input_length=sequence_length, name="embedding_layer")(model_input)
    embed = Dropout(dropout_prob[0], name="embedding_dropout_layer")(embed)
    
    # Convolutional block
    conv_blocks = []
    for sz in filter_sizes:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=sz,
                             padding="valid",
                             activation="relu",
                             strides=1, name="conv_layer_"+ str(sz))(embed)
        conv = MaxPooling1D(pool_size=2, name="conv_maxpool_layer_"+ str(sz))(conv)
        conv = Flatten(name="conv_flatten_layer_"+ str(sz))(conv)
        conv_blocks.append(conv)
        
    #concatenate conv layers
    conv_layers = Concatenate(name="conv_concate_layer")(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    flated_conv_layers = Dropout(dropout_prob[1], name="concate_dropout_layer")(conv_layers)
    
    #add events input layer
    events_input_layer = Input(shape=(events_seq_length,), name="event_input_layer")
    events_dense = Dense(int((events_seq_length/2)), activation="relu", name="event_dense_layer")(events_input_layer)
    
    #add ners input layer
    ners_input_layer = Input(shape=(ners_seq_length,), name="ner_input_layer")
    ners_dense = Dense(int((ners_seq_length/2)), activation="relu", name="ners_dense_layer")(ners_input_layer)
    
    #add sent2vec input layer
    sent2vec_input_layer = Input(shape=(700,), name="sent2vec_input_layer")
    sent2vec_dense_layer = Dense(350, activation="relu", name="sent2vec_dense_layer")(sent2vec_input_layer)
    
    
    #merge all input layers
    merged = concatenate([flated_conv_layers, events_dense, ners_dense, sent2vec_dense_layer], name="conv_event_ner_sent2vec_merge_layer")
    
    #convolution layer for the contcatenated features
    merged_reshaped = Reshape((6, 357))(merged)
    merged_conv = Convolution1D(filters= 10,
                             kernel_size= 3,
                             padding="valid",
                             activation="relu",
                             strides=1, name="merged_conv_layer_"+ str(3))(merged_reshaped)
    merged_conv = MaxPooling1D(pool_size=2, name="merged_conv_maxpool_layer_"+ str(3))(merged_conv)
    merged_conv = Flatten(name="merged_conv_flatten_layer_"+ str(3))(merged_conv)

    #dense layer
    dense = Dense(hidden_dims, activation="relu", name="conv_event_merge_dense_layer")(merged_conv)
    model_output = Dense(num_labels, activation="softmax", name="Output_layer")(dense)
    
    #create model
    model = Model([model_input, events_input_layer, ners_input_layer, sent2vec_input_layer], model_output)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    return model, embedding_weights


def train_evaluate_model(model, embedding_weights, x_train, event_x_train, ners_x_train, sent_x_train, y_train, x_test, event_x_test,  ners_x_test, sent_x_test, y_test):
    
    # Initialize weights with word2vec
    weights = np.array([v for v in embedding_weights.values()])
    print("Initializing embedding layer with word2vec weights, shape", weights.shape)
    embedding_layer = model.get_layer("embedding_layer")
    embedding_layer.set_weights([weights])
    
    # Train the model
    history = model.fit([x_train, event_x_train, ners_x_train, sent_x_train], y_train, batch_size=batch_size, epochs=num_epochs,
              validation_data=([x_test, event_x_test,  ners_x_test, sent_x_test], y_test), verbose=2)
    return history


if __name__ == "__main__":
    
    # Data Preparation
    print("Load data...")
    x, event_x, ners_x, sent_x, y, vocabulary, vocabulary_inv, event_voc, event_voc_inv, ners_voc, ners_voc_inv, num_labels, labels_dict = load_data()
    labels_y = y.argmax(axis=1)
    
    k_fold_val = StratifiedKFold(n_splits = kfold_splits, shuffle=True)
    for index, (train_indices, val_indices) in enumerate(k_fold_val.split(x, labels_y)):
        print("Training on fold " + str(index+1) + "/10...")
        
        x_train, x_test = x[train_indices], x[val_indices]
        y_train, y_test = y[train_indices], y[val_indices]
        
        event_x_train, event_x_test = event_x[train_indices], event_x[val_indices]
        ners_x_train, ners_x_test = ners_x[train_indices], ners_x[val_indices]
        sent_x_train, sent_x_test = sent_x[train_indices], sent_x[val_indices]
        
        model = None
        model, embedding_weights = create_model(x_train, event_x_train, ners_x_train, sent_x_train, y_train, x_test, event_x_test,  ners_x_test, sent_x_test, y_test, vocabulary, vocabulary_inv)
        history = train_evaluate_model(model, embedding_weights, x_train, event_x_train, ners_x_train, sent_x_train, y_train, x_test, event_x_test,  ners_x_test, sent_x_test, y_test)
        
        accuracy_history = history.history['acc']
        val_accuracy_history = history.history['val_acc']
        print("\n Last training accuracy: " + str(accuracy_history[-1]) + ", last validation accuracy: " + str(val_accuracy_history[-1])+" \n ")
        
    #Model Summary
    print(model.summary())
    #plot_model(model, to_file='event_summary_classification.png')
    
    model.save("model.h5")
    
    model.save("./final_models/model_conv_all.h5")
    
    #save model params for predict
    model_params = {"sequence_length": sequence_length, 
                    "events_seq_length": events_seq_length, 
                    "ners_seq_length":ners_seq_length,
                    "sent2vec_seq_length":sent2vec_seq_length,
                    "vocabulary": vocabulary,
                    "event_voc": event_voc,
                    "ners_voc": ners_voc,
                    "labels_dict": labels_dict}
    
    with open('./model_parameters/model_params_conv_all', 'wb') as fp:
        pickle.dump(model_params, fp)
    

