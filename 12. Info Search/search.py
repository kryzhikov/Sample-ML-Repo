import random
from build_index import Document

# вот ссылка на датасет - https://www.kaggle.com/datasets/fabiochiusano/medium-articles
# там много-много разных текстиков с сайти Medium


# LENGTH = 192368
LENGTH = 10


class Document:
    def __init__(self, title, text, url, tags):
        # можете здесь какие-нибудь свои поля подобавлять
        self.title = title
        self.text = text
        self.url = url
        self.tags = tags
    
    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.text + ' ...', self.url]

index = []


def build_index():
    # считывает сырые данные и строит индекс
    # данные я читаю из index.json, который надо создать, запустив файл build_index.py

    with open('12. Info Search\index.json') as f:
        

    pass



def score(query, document):
    # возвращает какой-то скор для пары запрос-документ
    # больше -- релевантнее
    return random.random()

def retrieve(query):
    # возвращает начальный список релевантных документов
    # (желательно, не бесконечный)
    candidates = []
    for doc in index:
        if query.lower() in doc.title.lower() or query in doc.text.lower():
            candidates.append(doc)
    return candidates[:10]