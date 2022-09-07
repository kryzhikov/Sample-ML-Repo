from multiprocessing import Pool, cpu_count
import re
from itertools import repeat

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

from math import ceil


# очистка текста
def clear_data_process(data, ind):
    desc = data.iloc[ind]['text'].lower()

    # удаляем пунктуацию
    desc = re.sub('[^a-zA-Z]', ' ', desc)

    # удаляем теги
    desc = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", desc)

    # удаление цифр и специальных символов
    desc = re.sub("(\\d|\\W)+", " ", desc)

    return desc


def clear_data(data, pool):
    size = len(data.seq)

    data['text'] = pool.starmap(
        clear_data_process,
        zip(repeat(data), range(size)),
        ceil(size / cpu_count())
    )


# уменьшение словаря
# стемминг
def delete_stopwords_then_stem_process(data, sw_eng, stem, ind):
    sent = data.iloc[ind]['text']
    clean_sent = ' '.join([stem(word) for word in sent.split() if word not in sw_eng])

    return clean_sent


def delete_stopwords_then_stem(data, pool):
    nltk.download('stopwords', quiet=True)
    sw_eng = set(stopwords.words('english'))
    stemmer = SnowballStemmer(language='english')
    size = len(data.seq)

    data['text'] = pool.starmap(
        delete_stopwords_then_stem_process,
        zip(repeat(data), repeat(sw_eng), repeat(stemmer.stem), range(len(data.seq))),
        ceil(size / cpu_count())
    )


def prepare_data():
    x = pd.read_csv('labeled_lyrics_cleaned.csv', delimiter=',')
    x = x.assign(text=x.artist.astype(str) + ' ' + x.song.astype(str) + ' ' + x.seq.astype(str))
    print('csv read')
    with Pool() as pool:
        clear_data(x, pool)
        print('clear_data')

        delete_stopwords_then_stem(x, pool)
        print('delete_stopwords_then_stem')

    return x
