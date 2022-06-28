from xml.dom.minidom import Document
from flask import Flask, render_template, request
from search import score, retrieve, build_index
from time import time
import numpy as np

app = Flask(__name__, template_folder='.')
build_index()


@app.route('/', methods=['GET'])
def index():
    start_time = time()
    query = request.args.get('query')
    if query is None:
        query = ''
    documents, q_n = retrieve(query)
    scored = np.array([score(query, doc, q_n) for doc in documents])
    results = [documents[scr].format(query)+['%.2f' % scored[scr]] for scr in (-scored).argsort()[:20]]
    return render_template(
        'index.html',
        time="%.2f" % (time()-start_time),
        query=query,
        search_engine_name='Movies',
        results=results
    )


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)
