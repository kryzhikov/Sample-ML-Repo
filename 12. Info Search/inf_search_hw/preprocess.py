
import re
import nltk
from nltk.corpus import stopwords
from functools import lru_cache
from nltk.stem.snowball import SnowballStemmer



def tokenize(text: str, regex="[А-Яа-яёA-z]+") -> list: # будем токенизировать с помощью регулярки
    regex = re.compile(regex)
    tokens = regex.findall(text.lower())

    return tokens

def remove_stopwords(
    lemmas, stopwords=stopwords.words("russian")):
    """Returns list of stemmas without stopwords"""
    return [w for w in lemmas if not w in stopwords and len(w) > 3]

@lru_cache(maxsize=128) # ускорим лемматизацию кэшированием 
def stemmatize_word(token):
    """Returns stemma"""
    snowball = SnowballStemmer('russian')
    # pymorphy = MorphAnalyzer()
    # return pymorphy.parse(token)[0].normal_form
    return snowball.stem(token)

def stemmatize_text(text):
    """Returns list of stemmas"""
    return [stemmatize_word(w) for w in text]

def clean_text(text):
    """Returns list of stemmas without stopwords"""
    tokens = tokenize(text)
    stemmas = stemmatize_text(tokens)

    return remove_stopwords(stemmas)
