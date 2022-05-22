from flask import Flask, render_template, request
from search import score, retrieve, get_index, get_doc, pretify
from time import time
import logging

app = Flask(__name__, template_folder='.')
get_index()

logging.basicConfig(filename='12. Info Search\Logs\server.log', filemode='a', format='%(message)s', level='DEBUG')

@app.route('/', methods=['GET'])
def index():
    start_time = time()
    query = request.args.get('query')

    if query is None:
        query = ''

    start = time()
    documents, ids = retrieve(query)
    logging.info(f'Retrieve: {((time() - start) * 1000):.2f}ms')

    start = time()
    scored = score(query, documents)
    scored = list(zip(scored, ids))
    logging.info(f'Score: {((time() - start) * 1000):.2f}ms')

    start = time()
    scored = sorted(scored, key=lambda doc: -doc[0][1])
    logging.info(f'Sorting: {((time() - start) * 1000):.2f}ms')

    results = [doc.format(query) + ['%.2f' % scr] + [id] for (doc, scr), id in scored] 
    return render_template(
        'index.html',
        time="%.2f" % (time()-start_time),
        query=query,
        search_engine_name='Articoogle',
        results=results
    )


@app.route('/page/<id>')
def page_view(id):
    doc = get_doc(int(id))

    return render_template(
        'page.html',
        title = doc.title,
        url=doc.url,
        text=pretify(doc.text)
    )


if __name__ == '__main__':
    app.run(debug=False, host='127.0.0.1', port=8080)
