#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from search import score, retrieve, build_index, build_tf_idf



if __name__ == "__main__":
    build_index()
    build_tf_idf()

    query = 'tampa bay rays'    

    docs = retrieve(query)
    scored = [(doc, score(query, doc)) for doc in docs]
    scored = sorted(scored, key=lambda doc: -doc[1])
    scored_ids = [item[0].id for item in scored]

    print(scored_ids)