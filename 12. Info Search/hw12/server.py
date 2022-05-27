from time import time

from flask import Flask, render_template, request

from initialization import prepare_data
from search import score, retrieve, build_index

data = None
app = Flask(__name__, template_folder='.')


@app.route('/', methods=['GET'])
def index():
    start_time = time()

    query = request.args.get('query')
    if query is None:
        query = ''

    documents = retrieve(query, data)
    scored = [(doc, score(query, doc)) for doc in documents]
    scored = sorted(scored, key=lambda doc: doc[1], reverse=True)

    results = [doc.format() + ['%.2f' % scr] for doc, scr in scored]
    return render_template(
        'index.html',
        time="%.2f" % (time() - start_time),
        query=query,
        search_engine_name='Google',
        results=results
    )


if __name__ == '__main__':
    data = prepare_data()
    build_index(data)
    print('index built')

    app.run(debug=True, host='127.0.0.1', port=8080, use_reloader=False)
