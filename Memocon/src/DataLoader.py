import os
import json
import collections
import pandas as pd

def load_concepts(concepts_file="/Memocon/datasets/concepts/meta-llama/Llama-3.1-8B-Instruct/concepts_D1_hfq_10000_agg_fth0_pth0_cl10.json"):
    with open(concepts_file, 'r') as f:
        lines = f.readlines()
    data = {}
    for line in lines:
        json_data = json.loads(line)
        title = json_data["title"]
        metadata = {}
        for key in json_data.keys():
            if key == "title":
                continue
            metadata[key] = json_data[key]
        data[title] = metadata
    return data

def load_popularities(path_pageviews="/Memocon/datasets/pageviews/pageviews-20250303-user.txt"):
    with open(path_pageviews, 'r') as f:
        lines = f.readlines()
        
    wiki_popularity = collections.defaultdict(int)
    for line in lines:
        line = line.strip().rsplit(',', 1)
        wiki_popularity[line[0]] = int(line[1])
    return wiki_popularity

def load_cb(CB_file="/Memocon/datasets/QA_dataset.json"):
    with open(CB_file, 'r') as f:
        lines = f.readlines()
    data = {}
    for line in lines:
        json_data = json.loads(line)
        subject = json_data["subject"]
        metadata = {}
        for key in json_data.keys():
            if key == "subject":
                continue
            metadata[key] = json_data[key]
        data[subject] = metadata
    return data

def load_wiki(wiki_file="datasets/wiki_en_popularity.csv"):
    wiki_df = pd.read_csv(wiki_file)
    data = {}
    for i in range(len(wiki_df)):
        line = wiki_df.loc[i]
        title = line["title"]
        metadata = {
            "id": line["id"],
            "url": line["url"],
            "text": line["text"],
            "popularity": line["popularity"],
            "creation_date": line["creation_date"],
        }
        data[title] = metadata
    del wiki_df
    return data

if __name__ == "__main__":

    concepts_file = "/Memocon/datasets/concepts/meta-llama/Llama-3.1-8B-Instruct/concepts_D1_hfq_10000_agg_fth0_pth0_cl10.json"
    CB_file = "/Memocon/datasets/QA_dataset.json"
    wiki_file = "datasets/wiki_en_popularity.csv"
    popularity_file = "/Memocon/datasets/pageviews/pageviews-20250303-user.txt"

    data = load_concepts(concepts_file)
    cb = load_cb(CB_file)                      # https://huggingface.co/datasets/Warrieryes/CB_qa/viewer/default/train
    wiki = load_wiki(wiki_file)
    popularities = load_popularities(popularity_file)
    popularities = {title.replace('_', ' '): pop for title, pop in popularities.items()}

    print(len(cb), len(wiki), len(popularities))
