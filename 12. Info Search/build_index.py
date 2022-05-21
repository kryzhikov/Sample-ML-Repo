import concurrent.futures as cf
import multiprocessing as mp
import pandas as pd
import re
import json


class Document:
    def __init__(self, title, text, url, tags):
        # можете здесь какие-нибудь свои поля подобавлять
        self.title = title
        self.text = text
        self.url = url
        self.tags = self.lower_array(tags)
    
    @staticmethod
    def lower_array(arr):
        arr = list(map(str.lower, arr))
        return arr

    def get_text(self):
        return self.title + self.text + ' '.join(self.tags)

    def format(self, query):
        # возвращает пару тайтл-текст-url, отформатированную под запрос
        return [self.title, self.text[:150] + ' ...', self.url]

    def __dict__(self):
        return {'title': self.title, 'text': self.text, 'url': self.url, 'tags': self.tags}

    def __str__(self):
        return f'{self.title[:10]}... {self.text[:10]}... {self.url}, {self.tags[:3]}'


LENGTH = 10
# index = mp.Array(mp.Manager().dict(), range(LENGTH))

def str_to_list(str):
    elements = str.split(',')
    arr = []

    for elem in elements:
        arr.append( re.sub(r'[\[\'\]]+', '', elem) )

    return arr


def add_new_document(info):
    folder.append(Document(
        info['title'], info['text'], info['url'], str_to_list(info['tags'])
    ))


def save_index():
    with open('12. Info Search\index.json', 'w') as f:
        pages = []
        for elem in index:
            pages.append(elem.__dict__())

        data = json.dumps(pages)
        f.write(data)


def build():
    df = pd.read_csv('12. Info Search\medium_articles.csv')

    # for i in range(LENGTH):
    #     add_new_document(df.iloc[i])
    print(folder.len())

    with cf.ProcessPoolExecutor() as executor:
        executor.map(add_new_document, [df.iloc[i] for i in range(LENGTH)])

    # print(folder.len())
    # save_index()


if __name__ == '__main__':
    
    build()