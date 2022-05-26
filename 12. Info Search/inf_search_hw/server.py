from flask import Flask, render_template, request
from search import score, retrieve, build_index
from time import time

app = Flask(__name__, template_folder='.')
build_index()


@app.route('/', methods=['GET'])
def index():
    start_time = time()
    query = request.args.get('query')
    if query is None:
        query = ''
    documents = retrieve(query)
    scored = [(doc, score(query, doc)) for doc in documents]
    scored = sorted(scored, key=lambda doc: -doc[1])
    results = [doc.format(query)+['%.2f' % scr] + [doc.date] for doc, scr in scored] 
    return render_template(
        'index.html',
        time="%.2f" % (time()-start_time),
        query=query,
        search_engine_name='Potekhindex',
        results=results
    )


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)





