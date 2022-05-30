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
    documents, documents_ = retrieve(query)
    scored = [(documents[num], score(query, doc)) for num, doc in enumerate(documents_)]
    scored = sorted(scored, key=lambda doc: -doc[1])
    results = [doc.format(query)+['%.2f' % scr] for doc, scr in scored] 
    return render_template(
        'index.html',
        time="%.2f" % (time()-start_time),
        query=query,
        search_engine_name='Google 2.0',
        results=results
    )


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)
