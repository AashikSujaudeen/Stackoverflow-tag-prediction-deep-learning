from datetime import datetime
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import unicodedata
import nltk
import re
nltk.download('punkt')

stop_words =set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
VAL_WRAP_WIDTH = 60
timer_initial = datetime.now()

timer = datetime.now()

class Normalize(object):
    def __init__(self):
        pass

    def to_lowercase(self, text):
        text_tokens = word_tokenize(text.lower())
        text_tokens = ' '.join([word for word in text_tokens])
        return text_tokens

    def html_cleanser(self, text):
        text = re.sub('<code>(.*?)</code>', '', text, flags=re.MULTILINE | re.DOTALL)
        soup = BeautifulSoup(text, 'html.parser')
        soup = soup.text.replace('\n', '')
        # soup = soup.text.replace('<p>', '')
        return soup

    def deEmojify(self, text):
        regrex_pattern = re.compile(pattern="["
                                            u"\U0001F600-\U0001F64F"  # emoticons
                                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                            "]+", flags=re.UNICODE)
        return regrex_pattern.sub(r'', text)

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

    def remove_stopwords(self, text):
        text_tokens = word_tokenize(text)
        tokens_without_sw = ' '.join([word for word in text_tokens if not word in stop_words])
        return tokens_without_sw

    def remove_single_letters(self, text):
        text_tokens = word_tokenize(text)
        text_without_sl = ' '.join(str(stemmer.stem(word)) for word in text_tokens if (len(word) != 1 or word == 'c' or word == 'C'))
        return text_without_sl

    def clean_accented_chars(self, text):
        return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

    def remove_punctuations(self, text):
        punc = '''!()``-`[]{};:'"\,<>./?@$%^&*_~'''
        for element in text:
            if element in punc:
                text = text.replace(element, "")
        return text

    def normalize(self, text):
        print('')

        text = self.to_lowercase(text)
        text = self.remove_single_letters(text)
        text = self.clean_accented_chars(text)
        text = self.text_expansion(text)
        text = self.remove_punctuations(text)
        text = self.deEmojify(text)
        text = self.remove_stopwords(text)
        return text

    def normalize_(self, text):
        print('')
        text = self.html_cleanser(text)
        text = self.to_lowercase(text)
        text = self.remove_single_letters(text)
        text = self.clean_accented_chars(text)
        text = self.text_expansion(text)
        text = self.remove_punctuations(text)
        text = self.deEmojify(text)
        text = self.remove_stopwords(text)
        return text