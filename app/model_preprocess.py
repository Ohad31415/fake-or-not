import pickle

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
 
lemmatizer = WordNetLemmatizer()

def pre_process(sent):
    sent = ' '.join([lemmatizer.lemmatize(word) for word in sent.split()])
    sent = tokenizer.texts_to_sequences([sent])
    
    maxlen = 200
    padding='post'
    sent = pad_sequences(sent, padding=padding, maxlen=maxlen)

    return sent
