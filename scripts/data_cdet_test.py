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
    narratives = {}

    for topic in topics:
        tid = topic['tid']
        narrative = topic['narrative']

        for question in topic['questions']:
            narrative += '\n' + question['question']
        narratives[tid] = narrative

    with open(args.pool_path, 'r') as f:
        for line in tqdm(f, desc='Loading pool docids'):
            topic, date, docid, qid, rank, sim, runtag = line.strip().split()
            docs[docid] = None

    with gzip.open(args.coll_path, 'rt') as f:
        for line in tqdm(f, desc='Loading documents'):
            doc = json.loads(line)
            if doc['id'] in docs:
                docs[doc['id']] = doc['text']

    data = []
    with open(args.pool_path, 'r') as f:
        for line in tqdm(f, desc='Building predict data'):
            topic, date, docid, qid, rank, sim, runtag = line.strip().split()
            data.append({
                'query_id': topic,
                'doc_id': docid,
                'query': narratives[topic],
                'doc': docs[docid],
                'relevance': -1
            })

    pd.DataFrame(data).to_parquet(args.output_path)

if __name__ == '__main__':
    home_dir = os.environ.get('HOME', '/home/ubuntu')
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--output_path', default='../data/external/test-cdet.parquet')
    arg_parser.add_argument('--topics_path', default=f'{home_dir}/trec/trec35/change/topics/test-topics-all-docs.jsonl')
    arg_parser.add_argument('--pool_path', default=f'{home_dir}/trec/trec35/change/pooling/pool.d3')
    arg_parser.add_argument('--coll_path', default=f'{home_dir}/ragtime1/eng-docs.jsonl.gz')
    args = arg_parser.parse_args()

    build(args)
