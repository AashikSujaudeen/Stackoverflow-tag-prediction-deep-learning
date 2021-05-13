from datetime import datetime
from bs4 import BeautifulSoup
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import unicodedata
import nltk
import gensim
import re
import pickle
import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG, filename='./logs/logfile_Preprocess.txt')
nltk.download('punkt')

logging.info('')
logging.info('Preprocess - Phase - Started at {}'.format(datetime.now()))
logging.info('--------------------------------------------------------')
logging.info('')


stop_words =set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
VAL_WRAP_WIDTH = 60
timer_initial = datetime.now()
F_PATH = 'datasets/'
TRAIN = 'Train.csv'
TEST = 'Test.csv'
CHUNK_SIZE = 750000 # 0.75 million observations w.r.t. my machine capacity

# 1-DATA EXTRACTION
timer = datetime.now()
train_df = pd.read_csv(F_PATH+TRAIN, sep=',', encoding='latin-1')
train_df = train_df.head(CHUNK_SIZE)
train_df['Questions'] = train_df['Title'] + ' ' + train_df['Body']
train_df['Questions_Raw'] = train_df['Title'] + ' ' + train_df['Body']
logging.info("Input file load completed")
logging.info("")
logging.info("Time taken for input file load: {}".format(datetime.now() - timer))
logging.info("")

#Class to pro-process the data before training the data
class Preprocessor(object):
    def __init__(self, max_seq_length):
        self._max_seq_length = max_seq_length
        self._tokenizer = None
        self._top_n_tags = None
        self._vocab_size = None
        self._dataframe = None


# To remove null based records
    def remove_nulls(self, df):
        logging.info("Nulls then: {}".format(df.isnull().sum()))
        df = df.dropna()
        logging.info("Nulls now: {}".format(df.isnull().sum()))
        return df


    # To remove duplicate records
    def remove_duplicates(self, df):
        duplicates_df = df[df.duplicated(['Questions','Tags'])]
        duplicate_percentage = duplicates_df.shape[0]/df.shape[0]*100
        logging.info('')
        logging.info("Duplicates - then: {:.2f}%".format(duplicate_percentage))
        df = df.drop(duplicates_df.index)
        duplicates_df = df[df.duplicated(['Questions', 'Tags'])]
        duplicate_percentage = duplicates_df.shape[0]/df.shape[0]*100
        logging.info("Duplicates - now: {:.2f}%".format(duplicate_percentage))
        return df


    #To take data based on the top n tags
    def top_n_tags_based_data(self, df, n_tags):
        tag_count = 0
        total_tag_count = sum([tag_count + len(tag_list.split(" ")) for tag_list in df["Tags"]])

        tag_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(" "))
        vectorized_tags = tag_vectorizer.fit_transform(df["Tags"])  # .values.astype('U'))
        individual_tags = tag_vectorizer.get_feature_names()

        tag_frequency = vectorized_tags.sum(axis=0).A1
        tag_frequency_df = pd.DataFrame({'Tags': individual_tags, 'Frequency': tag_frequency})
        tag_frequency_df.sort_values(by=['Frequency'], inplace=True, ascending=False)


        tag_top_frequency = tag_frequency_df['Frequency'].values[0:n_tags]

        logging.info("Percentage of total tags covered by top {} tags: {}%".format(n_tags, tag_top_frequency.sum()/total_tag_count * 100))

        top_n_tags = tag_frequency_df['Tags'].values[0:tag_top_frequency.shape[0]]

        with open('models/unique_tags.pickle', 'wb') as handle:
            pickle.dump(top_n_tags, handle, protocol=pickle.HIGHEST_PROTOCOL)

        id_list = []
        questions_list = []
        questions_raw_list = []
        tags_list = []
        # pattern = '|'.join(top_10_tags)
        timer = datetime.now()
        for index in df.index:
            if bool([tag for tag in top_n_tags if (tag in df['Tags'][index])]):
                id_list.append(df['Id'][index])
                questions_list.append(df['Questions'][index])
                questions_raw_list.append(df['Questions_Raw'][index])
                tags_list.append(df['Tags'][index])
        df = pd.DataFrame({'Id': id_list, 'Questions_Raw': questions_raw_list,'Questions': questions_list, 'Tags': tags_list})
        logging.info("Total Time taken for top {} tags based data : {}".format(n_tags, datetime.now() - timer_initial))

        self._top_n_tags = top_n_tags
        return df


    # To remove the HTML scripts and the code contents
    def html_cleanser(self, text):
        #logging.info("HTML Before: {}".format(text))
        text = re.sub('<code>(.*?)</code>', '', text, flags=re.MULTILINE | re.DOTALL)
        soup = BeautifulSoup(text, 'html.parser')
        soup = soup.text.replace('\n', '')
        return soup


    # To remove emojies
    def deEmojify(self, text):
        regrex_pattern = re.compile(pattern="["
                                            u"\U0001F600-\U0001F64F"  # emoticons
                                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                            "]+", flags=re.UNICODE)
        return regrex_pattern.sub(r'', text)


    # To expand the short forms used in the question/answers
    def text_expansion(self, text):
        text = text.lower()
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "can not ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"\'scuse", " excuse ", text)
        text = re.sub(r"\'\n", " ", text)
        text = re.sub(r"\'\xa0", " ", text)
        text = re.sub('\s+', ' ', text)
        text = text.strip(' ')
        return text


    # To remove the stopwords in the data
    def remove_stopwords(self, text):
        text_tokens = word_tokenize(text)
        tokens_without_sw = ' '.join([word for word in text_tokens if not word in stop_words])
        return tokens_without_sw


    # To remove the single letters which dont contribute much to the machine learning except the language based single word like 'C' or 'c'
    def remove_single_letters(self, text):
        text_tokens = word_tokenize(text.lower())
        text_without_sl = ' '.join(str(stemmer.stem(word)) for word in text_tokens if (len(word) != 1 or word == 'c' or word == 'C'))
        # text_without_sl = ' '.join(str(stemmer.stem(word)) for word in text_tokens)
        # ' '.join(i for i in s.split() if not (i.isalpha() and len(i)==1))
        return text_without_sl


    # To remove the accented letters
    def clean_accented_chars(self, text):
        return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')


    # To remove the punctuations
    def remove_punctuations(self, text):
        punc = '''!()``-`[]{};:'"\,<>./?@$%^&*_~'''
        for element in text:
            if element in punc:
                text = text.replace(element, "")
        return text

    # To encode the tags
    def tag_encoder(self, df):
        df.Tags = df.Tags.apply(lambda x: x.split(" "))
        final_tag_data = []
        for tags in df['Tags']:
            temp = []
            for tag in tags:
                if tag in self._top_n_tags:
                    temp.append(tag)
            final_tag_data.append(temp)
        df['Tags_Smoothened'] = final_tag_data
        df = df[df['Tags_Smoothened'].str.len() > 0]
        tag_encoder = MultiLabelBinarizer()
        tags_encoded = tag_encoder.fit_transform(df.Tags_Smoothened)

        with open('models/tag_encoder.pickle', 'wb') as handle:
            pickle.dump(tag_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logging.info("Encoded Tags - Shape: {}".format(tags_encoded.shape))
        tags_encoded = tags_encoded.astype(float)

        return df, tags_encoded

    # To normalize the input data by calling the data wrangling methods
    def normalize(self, df, n_tags):
        logging.info("Preprocessing - started...")
        df = self.top_n_tags_based_data(df, n_tags)
        logging.info("Preprocessing activity - 1. Top {} tags based data extraction - Completed".format(n_tags))
        logging.info("")
        df = self.remove_duplicates(df)
        logging.info("Preprocessing activity - 2. Remove Duplicates: Completed at {}".format(datetime.now()))
        logging.info("")

        df = self.remove_nulls(df)
        logging.info("Preprocessing activity - 3. Remove Nulls: Completed at {}".format(datetime.now()))
        logging.info("")

        df['Questions'] = df['Questions'].apply(self.html_cleanser)
        logging.info("Preprocessing activity - 4. HTML Cleanser: Completed at {}".format(datetime.now()))
        logging.info("")

        df['Questions'] = df['Questions'].apply(self.deEmojify)
        logging.info("Preprocessing activity - 5. Remove Emojis: Completed at {}".format(datetime.now()))
        logging.info("")

        df['Questions'] = df['Questions'].apply(self.text_expansion)
        logging.info("Preprocessing activity - 6. Text Expansions: Completed at {}".format(datetime.now()))
        logging.info("")

        df['Questions'] = df['Questions'].apply(self.remove_stopwords)
        logging.info("Preprocessing activity - 7. Stopwords removal: Completed at {}".format(datetime.now()))
        logging.info("")

        df['Questions'] = df['Questions'].apply(self.remove_single_letters)
        logging.info("Preprocessing activity - 8. Remove single letters: Completed at {}".format(datetime.now()))
        logging.info("")

        df['Questions'] = df['Questions'].apply(self.clean_accented_chars)
        logging.info("Preprocessing activity - 9. Remove accented characters: Completed at {}".format(datetime.now()))
        logging.info("")

        df['Questions'] = df['Questions'].apply(self.remove_punctuations)
        logging.info("Preprocessing activity - 10. Remove punctuations: Completed at {}".format(datetime.now()))
        logging.info("")
        logging.info("Preprocessing - completed")
        return df

    # Create vocabulary for the training data
    def fit(self, X):
        # Create vocabulary from input corpus.
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X)
        unique_tokens = tokenizer.word_index
        #vocab_size = len(tokenizer.word_index)
        vocab_size = len(unique_tokens)
        with open('models/vocab_size.pickle', 'wb') as handle:
            pickle.dump(vocab_size, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self._tokenizer = tokenizer
        # CHECK ON FIXED VOCAB SIZE AS HYPERPARAMETER
        logging.info("Found {} unique tokens".format(vocab_size))

        with open('models/tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return vocab_size


    # Create sequence for the train data (preparation for deep learning model)
    def transform(self, X):
        # Transform text to sequence of integers
        text_sequence = self._tokenizer.texts_to_sequences(X)

        # Fix sequence length to max value. Sequences shorter than the length are
        # padded in the beginning and sequences longer are truncated at the beginning.
        padded_text_sequence = pad_sequences(text_sequence, maxlen=self._max_seq_length)
        padded_text_sequence = padded_text_sequence.astype(float)
        return padded_text_sequence


# Custom word to vector embedding builder using Gensim
class CustomWord2VectorEmbedding(Preprocessor):
    #def __init__(self,  W2V_SIZE, W2V_WINDOW, W2V_MIN_COUNT, W2V_EPOCH):
     #   #self._data = data
      #  self._W2V_SIZE = W2V_SIZE
       ## self._W2V_WINDOW = W2V_WINDOW
        #self._W2V_MIN_COUNT = W2V_MIN_COUNT
        #self._W2V_EPOCH = W2V_EPOCH
        #self._w2v_model = None

    def gensim_model(self, data, W2V_SIZE, W2V_WINDOW, W2V_MIN_COUNT, W2V_EPOCH):
        w2v_model = gensim.models.word2vec.Word2Vec(size=W2V_SIZE,
                                                    window=W2V_WINDOW,
                                                    min_count=W2V_MIN_COUNT,
                                                    workers=8)
        documents = [_text.split() for _text in np.array(data)]
        w2v_model.build_vocab(documents)
        words = w2v_model.wv.vocab.keys()
        vocab_size = len(words)
        logging.info("Vocab size: {}".format(vocab_size))

        # Train Word Embeddings
        w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)
        w2v_model.save('models/SO_word2vec_embeddings.bin')
        #self._w2v_model = w2v_model
        return w2v_model

    def build_embedding(self, model, vocab_size):
        word2vector_size = self._max_seq_length
        embedding_matrix = np.zeros((vocab_size+1, word2vector_size))

        with open('models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        logging.info("Token size: ".format(len(tokenizer.word_index)))
        for word, i in tokenizer.word_index.items():
            if word in model.wv:
                embedding_matrix[i] = model.wv[word]

        logging.info("embedding_matrix.shape: {}".format(embedding_matrix.shape))
        with open('models/embedding_matrix.pickle', 'wb') as handle:
            pickle.dump(embedding_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return embedding_matrix


# Parameters Needed for word2vec model
MAX_SEQ_LENGTH = 500
TOP_N_TAGS = 500
W2V_SIZE = MAX_SEQ_LENGTH
W2V_WINDOW = 7
W2V_MIN_COUNT = 10
W2V_EPOCH = 32

# Create Preprocess class instance for text preprocessing
preprocessor = Preprocessor(MAX_SEQ_LENGTH)

# Normalize the input data (features data)
df_new = preprocessor.normalize(train_df, TOP_N_TAGS)

# Encode the tags (label data)
df_new, tags_encoded = preprocessor.tag_encoder(df_new)

# Create vocabulary for the training data
vocab_size = preprocessor.fit(df_new.Questions)

# Split the data for training and testing
X_train, X_test, X_train_raw, X_test_raw, y_train, y_test = train_test_split(np.array(df_new.Questions), np.array(df_new.Questions_Raw), tags_encoded, test_size=0.2, random_state=42)
logging.info("Train size: {}, Test size: {}".format(len(X_train), len(X_test)))

# Dump the processed data for re-use
with open('models/preprocessor.pkl', 'wb') as handle:
    pickle.dump(preprocessor, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('models/X_train.pickle', 'wb') as handle:
    pickle.dump(X_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('models/X_test.pickle', 'wb') as handle:
    pickle.dump(X_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('models/X_train_raw.pickle', 'wb') as handle:
    pickle.dump(X_train_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('models/X_test_raw.pickle', 'wb') as handle:
    pickle.dump(X_test_raw, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('models/y_train.pickle', 'wb') as handle:
    pickle.dump(y_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('models/y_test.pickle', 'wb') as handle:
    pickle.dump(y_test, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('models/X_train.pickle', 'rb') as handle:
    X_train = pickle.load(handle)

# Create sequence for the train data (preparation for deep learning model)
X_train_padded = preprocessor.transform(X_train)
X_test_padded = preprocessor.transform(X_test)

with open('models/X_train_padded.pickle', 'wb') as handle:
    pickle.dump(X_train_padded, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('models/X_test_padded.pickle', 'wb') as handle:
    pickle.dump(X_test_padded, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Prepare the train data for embedding layer
embedding = CustomWord2VectorEmbedding(MAX_SEQ_LENGTH)
word2vec_model = embedding.gensim_model(df_new.Questions, W2V_SIZE, W2V_WINDOW, W2V_MIN_COUNT, W2V_EPOCH)
embedding_matrix = embedding.build_embedding(word2vec_model, vocab_size)


logging.info('')
logging.info('Preprocess Phase - Completed at {}'.format(datetime.now()))
logging.info('--------------------------------------------------------')
logging.info('')





