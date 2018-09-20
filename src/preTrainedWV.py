from gensim.models import KeyedVectors
from keras.layers import Flatten
from keras.layers import MaxPooling1D
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from nltk.corpus import stopwords
from pathlib import Path
import pickle

MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 30
EMBEDDING_DIM = 300

EMBEDDING_FILE = "../lib/GoogleNews-vectors-negative300.bin"
category_index = {"balance":0, "budgeting":1, "housing":2}#, "unknown":3}
category_reverse_index = dict((y,x) for (x,y) in category_index.items())
STOPWORDS = set(stopwords.words("english"))

trainingDataPickle = Path("financebot/src/data/trainingDataPickle")
if trainingDataPickle.is_file():
    data = pickle.load(open('financebot/src/data/trainingDataPickle','rb'))
    balance = data['balance']
    budgeting = data['budgeting']
    housing = data['housing']
    unknown = data['unknown']

datasets = [balance, budgeting, housing]
for data in datasets:
    print("Has null values: ", data.isnull().values.any())

def preprocess(text):
    text= text.strip().lower().split()
    text = filter(lambda word: word not in STOPWORDS, text)
    return " ".join(text)

for dataset in datasets:
    dataset['title'] = dataset['title'].apply(preprocess)


model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
embedding_layer = model.get_keras_embedding()
