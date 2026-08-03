import argparse
import os

import pandas as pd
import json
import gzip
import collections
import random
from tqdm import tqdm

def build(args):
    with open(args.topics_path, 'r') as f:
        topics = [json.loads(line) for line in f.readlines()]

    docs = {}
    doc_topics = collections.defaultdict(set)
    topic_docs = collections.defaultdict(set)
    narratives = {}

    for topic in topics:
        tid = topic['tid']
        narrative = topic['narrative']

        for question in topic['questions']:
            narrative += '\n' + question['question']
            for docid in question['rel_docs']:
                docs[docid] = None
                topic_docs[tid].add(docid)
                doc_topics[docid].add(tid)
        narratives[tid] = narrative

    with gzip.open(args.coll_path, 'rt') as f:
        for line in tqdm(f, desc='Loading documents'):
            doc = json.loads(line)
            if doc['id'] in docs:
                docs[doc['id']] = doc['text']

    data = []
    for topic in topic_docs:
        already_have = set()
        for docid in topic_docs[topic]:
            data.append({
                'query_id': topic,
                'doc_id': docid,
                'query': narratives[topic],
                'doc': docs[docid],
                'relevance': 1
            })
            already_have.add(docid)
            have_neg = False
            while not have_neg:
                neg_docid = random.choice(list(docs.keys()))
                if neg_docid not in already_have and topic not in doc_topics[neg_docid]:
                    data.append({
                        'query_id': topic,
                        'doc_id': neg_docid,
                        'query': narratives[topic],
                        'doc': docs[neg_docid],
                        'relevance': 0
                    })
                    already_have.add(neg_docid)
                    have_neg = True

    pd.DataFrame(data).to_parquet(args.output_path)

if __name__ == '__main__':
    home_dir = os.environ.get('HOME', '/home/ubuntu')
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--output_path', default='../data/external/train-cdet.parquet')
    arg_parser.add_argument('--topics_path', default=f'{home_dir}/trec/trec35/change/topics/test-topics-all-docs.jsonl')
    arg_parser.add_argument('--coll_path', default=f'{home_dir}/ragtime1/eng-docs.jsonl.gz')
    args = arg_parser.parse_args()

    build(args)
