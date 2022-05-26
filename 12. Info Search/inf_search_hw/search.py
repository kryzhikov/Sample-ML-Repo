import pandas as pd
from preprocess import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer



class Document:
    def __init__(self, title, text, date):
        # можете здесь какие-нибудь свои поля подобавлять
        self.title = title
        self.text = text
        self.date = date
    
    def format(self, query):
        # возвращает пару тайтл-текст, отформатированную под запрос
        return [self.title, self.text + ' ...']

index = {
    
}

# тут у меня вижак не хотел искать файл, пришлось полный путь писать
# датасет не влезет на гит, вот линк -- https://drive.google.com/file/d/1MXk4uyO9pajEv_prRvticXaVOPGFYiKI/view?usp=sharing
df = pd.read_csv('/Users/apotekhin/repositories/Sample-ML-Repo/12. Info Search/inf_search_hw/df_cleaned.csv')
# наш оффлайн признак -- дата новости
df = df.sort_values(by='date', ascending=False)
df.reset_index(inplace=True)

def build_index():
    # проходит по стеммам и добавляем уникальные в словарь,
    # где ключ -- стемма, значение -- индексы новостей в датасете
    for i, text in enumerate(df.stemmas):
        for word in set(text.split()):
            if word.lower() not in index:
                index[word.lower()] = []
            index[word.lower()].append(i)
    

def score(query, document):
    # возвращает косинусное расстояние между текстами на tfidf векторах
    query = ' '.join(clean_text(query))
    document = ' '.join(clean_text(document.title))
    return cosine_similarity(query, document)

def cosine_similarity(a, b, vectorizer=TfidfVectorizer()):
    tfidf = vectorizer.fit_transform([a, b])
    return ((tfidf * tfidf.T).toarray())[0, 1]

def two_lists_intersections(lst_1, lst_2):
    # зануляем указатели
    i = j = 0
    # итоговый список
    res_list = []
    # пока один из указателей не вышел за пределы
    while i < len(lst_1) and j < len(lst_2):
        # если один из указателей меньше другого -- двигаем его
        if lst_1[i] < lst_2[j]:
            i += 1
        elif lst_2[j] < lst_1[i]:
            j += 1
        # если равны -- добавляем элемент
        else:
            res_list.append(lst_1[i])
            i += 1
            j += 1
            
    if not res_list:
        return []
    
    # если один из указателей достиг конца списка -- двигаем другой до предела
    while i < len(lst_1) and res_list[-1] == lst_1[i]:
        res_list.append(lst_1[i])
        
    while j < len(lst_2) and res_list[-1] == lst_2[j]:
        res_list.append(lst_2[j])
            
            
    return res_list

def multiple_lists_intersections(lst):
    # повторяем операцию с пересечением на каждую итерацию
    if len(lst) < 3:
        print("Use two_lists_intersections instead")
        return 
    res_lst = two_lists_intersections(lst[0], lst[1])
    for i in lst[2:]:
        res_lst = two_lists_intersections(res_lst, i)
        
    return res_lst

def calculate_intersections(concordances):
    if len(concordances) == 2:
        return two_lists_intersections(concordances[0], concordances[1])
    elif len(concordances) > 2:
        return multiple_lists_intersections(concordances)
    elif len(concordances) == 1:
        return concordances[0]
    else:
        return []


def retrieve(query):
    # возвращает начальный список релевантных документов отсортированных по дате
    query = clean_text(query) 
    concordances = []
    for token in query:
        if token in index:
            if index[token] in concordances:
                continue
            concordances.append(index[token])
            
    if len(concordances) == 0:
        return [Document(title="Sorry, no news for that", text="Maybe tomorrow...", date='Not today')]
    
    index_intersections = calculate_intersections(concordances)
    print(index_intersections)
    if len(index_intersections) == 0:
        return [Document(title="Sorry, no news for that", text="Maybe tomorrow...", date='Not today')]
    response_docs = []
    if len(index_intersections) == 1:
        response_docs.append([df.loc[index_intersections[0], 'title'], df.loc[index_intersections[0], 'text'],  
                                df.loc[index_intersections[0], 'date']])
    else:
        for i in index_intersections:
            response_docs.append([df.loc[i, 'title'], df.loc[i, 'text'], df.loc[i, 'date']])
            if len(response_docs) > 50:
                return [Document(title, text, date) for title, text, date in response_docs]
    return [Document(title, text, date) for title, text, date in response_docs]
