import os
import codecs
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from tokenizeData import Lemmatizer

def create_train_data_csv():
    files = os.listdir("data/CoreferencedPlots")
    w = open('data/train20.csv', 'w+')
    w.write('Category,Descript\n')
    for file in files[:20]:
        with codecs.open('data/CoreferencedPlots/'+file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            text = Lemmatizer(text)
            tokens = sent_tokenize(text)
            lines = len(tokens)
            for i in range(lines):
                for j in range(i+1, lines):
                    #for k in range(j+1, lines):
                    line1 = tokens[i].replace(",", "")
                    line2 = tokens[j].replace(",", "")
                    #line3 = tokens[k].replace(",", "")
                    combineline = line1 + ' ' + line2
                    w.write(file[:-4] + ', ' + combineline + '\n')
            f.close()
    w.close()

def create_movie_summary_csv():    
    files = os.listdir("data/CoreferencedPlots")
    w = open('data/trainMovie.csv', 'w+')
    w.write('sentiment,review\n')
    for file in files[:20]:
        with codecs.open('data/CoreferencedPlots/'+file, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
            text = text.replace(",", "")
            w.write(file[:-4] + ', ' + text + '\n')
            f.close()
    w.close()
    
    
create_train_data_csv()
create_movie_summary_csv()

