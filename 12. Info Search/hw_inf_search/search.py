import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
import json
import string
import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
import pickle
from sklearn.metrics.pairwise import cosine_similarity


class Article:
    def __init__(self, article_id, title, abstract):
        self.id = article_id
        self.title = title
        self.abstract = abstract
    
    def format(self, query):
        # возвращает тройку тайтл-текст-ссылка, отформатированную под запрос
        return [
            self.title,
            self.abstract[:500] + ' ...',
            'https://www.arxiv.org/abs/{}'.format(self.id)
        ]


class SearchEngine:
    def __init__(self):
        self.inv_index = {} # inv_index[word] = [doc_id_0, doc_id_1, ..., doc_id_n]
        self.articles_dict = {} # article_dict[id] = article with needed id
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('omw-1.4')
        nltk.download('averaged_perceptron_tagger')
        with open('Data/tf-idf-vectorizer', 'rb') as f:
            self.tf_idf = pickle.load(f)
        
    def get_wordnet_pos(self, treebank_tag):
        my_switch = {
            'J': wordnet.ADJ,
            'V': wordnet.VERB,
            'N': wordnet.NOUN,
            'R': wordnet.ADV,
        }
        for key, item in my_switch.items():
            if treebank_tag.startswith(key):
                return item
        return wordnet.NOUN

    def lemmatize_text(self, text):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        tokenized_text = nltk.word_tokenize(text)
        pos_tagged = [(word, self.get_wordnet_pos(tag))
                    for word, tag in pos_tag(tokenized_text)]
        return ' '.join([lemmatizer.lemmatize(word, tag)
                        for word, tag in pos_tagged])

    def tokenize(self, text):
        text = ' '.join(text.split()) # Remove '\n' from text
        text = self.lemmatize_text(text)
        remove_punctuation=str.maketrans('', '', string.punctuation) # Remove punctuation
        text = text.translate(remove_punctuation)
        return text.lower().split()

    def build_index(self):
        # считывает сырые данные и строит индекс
        # Считаем данные, обработаем тексты, сделаем инвертированный индекс

        # Get articles
        articles = []
        with open('Data/articles.json') as data:
            for line in data:
                article = json.loads(line)
                article = Article(article['id'], article['title'], article['abstract'])
                articles.append(article)
                self.articles_dict[article.id] = article

        # Process each article into (id, tokens)    
        articles = map(lambda article: (article.id, set(self.tokenize(article.title)).union(self.tokenize(article.abstract))), articles)
        articles = list(articles)

        # Inverted index
        for article in articles:
            article_id = article[0]
            text = article[1]
            for word in text:
                if not (word in self.inv_index):
                    self.inv_index[word] = set()
                self.inv_index[word].add(article_id)

    def score(self, query, article):
        # возвращает какой-то скор для пары запрос-документ
        # больше -- релевантнее
        query = self.tokenize(query)
        article = self.tokenize(article.abstract)
        query = self.tf_idf.transform([' '.join(query)])
        article = self.tf_idf.transform([' '.join(article)])
        return cosine_similarity(query, article).sum()

    def retrieve(self, query):
        # возвращает начальный список релевантных документов
        words = self.tokenize(query)
        if len(words) == 0:
            return []
        words = list(set(words)) # delete duplicates
        articles = []
        candidate_lists = []
        for word in words:
            if not (word in self.inv_index):
                return []
            candidate_lists.append(list(self.inv_index[word]))
        num_words = len(words)
        cursor = [0]*num_words
        while True: # TODO remove while-true
            # Check if some cursor out of range
            exit_flag = False
            for i in range(num_words):
                if cursor[i] >= len(candidate_lists[i]):
                    exit_flag = True
                    break
            if exit_flag:
                break
            # Check if all cursors point at the same object
            obj = candidate_lists[0][cursor[0]]
            same_object = True
            for i in range(num_words):
                if candidate_lists[i][cursor[i]] != obj:
                    same_object = False
                    break
            # If all cursors point at the same object, add it to articles
            if same_object:
                articles.append(obj)
                for i in range(num_words):
                    cursor[i] += 1
            # Otherwise find minimal object and update it
            else:
                minimal_object = candidate_lists[0][cursor[0]]
                for i in range(num_words):
                    minimal_object = min(minimal_object, candidate_lists[i][cursor[i]])
                for i in range(num_words):
                    if candidate_lists[i][cursor[i]] == minimal_object:
                        cursor[i] += 1

        articles = list(map(lambda article_id: self.articles_dict[article_id], articles))
        return articles