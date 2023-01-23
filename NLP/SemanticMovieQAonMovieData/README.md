# Semantic Movie Search

## Problem
This is one of the common type of Question Answering problem over Natural Language. A previously known Movie Summaries is taken as the knowledge base to answer the queries pertaining to the same.
For Ex: Given the summary of Batman Movie, queries could be: <br/>
"Hero in the movie is afraid of bats and through intense training from Ras Al Gul, he conquers his fear and evolves as a super hero." <br/>

## Approach
The approach to this problem is to treat it as text classification while retaining the semantic features from the summaries and queries. A combination of Neural Networks is used to acheive the same:
1. A Convolution Neural Network on sentences of the Summaries to obtain the features with their spatial significance.
2. A Fully connected NN to obtain the features of "Actions" in the query.
3. A Fully connected NN to obtain the featueres of "NERs" in the query.
4. A Fully connected NN to obtain the featueres of sentence embedding.
<br/>
The features obtained are concatenated and passed through another CNN and finally to output layer, which predicts the probability of each movie that a query might belong to.

## Dependencies
To successuflly run the project, the following deps are requried:
1. Keras
2. Numpy
3. pickle
4. [Sent2Vec](https://github.com/epfml/sent2vec)
5. [sent2vec_wiki_bigrams](sent2vec_wiki_bigrams )
6. NLTK
7. [neuralcoref](https://github.com/huggingface/neuralcoref)
8. Pandas
9. [pyjnius](https://github.com/kivy/pyjnius)
10. k-parser
11. The jars (Event Extractor, NER Extractor) provided with the project

## Setup
All the above libraries are required to proceed to launch the project. Once everything is installed, follow:

### Setup EventExtractor, NER Extractor libraries using pyjnius and jar files
This is required for the 2, 3 Neural Networks from the "Approach". The jar files provided extract the events and ners for a given sentence.<br/>
1. Setup k-parser as per its documentation.
2. Place the provided jars in a directory.
3. Open the config.py and change the respective paths (use full path).

**Note:** Events and NERs for the summaries have been generated already and they are used for training. If required to generate events/ners for new data, please be noted that it make take several minutes. 

### Setup Sent2vec:
Once the sent2vec is installed, you are required to download the 16gb wiki_bigrams file and place it in the project root folder.

## Run
1. Run sentiment_cnn.py to train the above discussed network. This would generate model and its params.
2. Run predict.py to predict on the queries in the ./data/test.txt file.





