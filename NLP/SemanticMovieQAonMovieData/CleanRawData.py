import os
import re
files = os.listdir("Data/RawPlots")

for file in files:
    f = open('Data/RawPlots/' + file, 'r', encoding='utf8')
    text = f.read()
    text = re.sub('<[^<]+?>', '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\.', '. ', text)
    text = re.sub('\"', '', text)
    text = re.sub(',', '', text)
    text = re.sub('\'', '', text)
    text = re.sub('\(', '', text)
    text = re.sub('\)', '', text)
    text = text.strip()
    file1 = re.sub(',', '_', file)
    w = open('CleanData/'+file1, 'w+')
    w.write(text)
    w.close()
    f.close()

