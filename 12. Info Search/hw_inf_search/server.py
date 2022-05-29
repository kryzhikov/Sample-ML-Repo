from flask import Flask, render_template, request
from search import SearchEngine
from time import time

app = Flask(__name__, template_folder='.')
search_engine = SearchEngine()
search_engine.build_index()


@app.route('/', methods=['GET'])
def index():
    start_time = time()
    query = request.args.get('query')
    if query is None:
        query = ''
    documents = search_engine.retrieve(query)
    documents = sorted(documents, key=lambda doc: -search_engine.score(query, doc))[:10]
    results = [doc.format(query)+['%.2f' % search_engine.score(query, doc)] for doc in documents] 
    return render_template(
        'index.html',
        time="%.2f" % (time()-start_time),
        query=query,
        results=results
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
