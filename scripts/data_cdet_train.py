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
    all_rel = set()
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
                all_rel.add(docid)
        narratives[tid] = narrative

    with gzip.open(args.coll_path, 'rt') as f:
        for line in tqdm(f, desc='Loading documents'):
            doc = json.loads(line)
            docs[doc['id']] = doc['text']

    all_docs = list(docs.keys())

    data = []
    for topic in tqdm(topic_docs, desc='Building train data'):
        already_have = all_rel.copy()
        for docid in topic_docs[topic]:
            data.append({
                'query_id': topic,
                'doc_id': docid,
                'query': narratives[topic],
                'doc': docs[docid],
                'relevance': 1
            })
            already_have.add(docid)

        # Add random negatives until we have a ratio of 0.05 rel to 0.95 nonrel
        for _ in tqdm(range(len(topic_docs[topic]) * 19), desc='Adding negatives', leave=False):
            got_one = False
            while not got_one:
                docid = random.choice(all_docs)
                if docid not in already_have and topic not in doc_topics[docid]:
                    data.append({
                        'query_id': topic,
                        'doc_id': docid,
                        'query': narratives[topic],
                        'doc': docs[docid],
                        'relevance': 0
                    })
                    already_have.add(docid)
                    got_one = True

    pd.DataFrame(data).to_parquet(args.output_path)

if __name__ == '__main__':
    home_dir = os.environ.get('HOME', '/home/ubuntu')
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--output_path', default='../data/external/train-cdet.parquet')
    arg_parser.add_argument('--topics_path', default=f'{home_dir}/trec/trec35/change/topics/test-topics-all-docs.jsonl')
    arg_parser.add_argument('--coll_path', default=f'{home_dir}/ragtime1/eng-docs.jsonl.gz')
    args = arg_parser.parse_args()

    build(args)
