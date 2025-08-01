import re
import os
import sys
import json
import random
import string
import argparse

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from prettytable import PrettyTable
from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

sys.path.append("/Memocon/src")
from Utils import print_
from LLM import LLM
from Prompt import Prompt, extract_concepts_with_percentages

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', dest='file', type=str)   # File name in selected_dataset_dir
parser.add_argument('-c', '--cuda', dest='cuda', type=str)   # The cuda id to run this script
args = parser.parse_args()

def aggregate_results(all_concepts, frequency_th=0.0, percentage_th=0.0, concepts_limit=10):
    """
    Aggregate high-frequency concepts with multiple sampling：
        1. select concepts that of high frequency >= "frequency_th".
        2. select concepts that of high importance percentage >= "percentage_th".
        3. select only "concepts_limit" concepts according to "frequency".
        
    INPUT: Concepts (List) [concepts1, concepts2, ...], concepts={c1: p1, c2: p2, ...}
    OUTPUT: Aggregated concept (Dict), concepts_aggregated={c1_new: p1_new, c2_new: p2_new, ...}
    """
    concept_stats = defaultdict(list)
    
    for concepts in all_concepts:
        for cp, pct in concepts.items():
            concept_stats[cp].append(pct)
    
    aggregated = []
    for cp, pct in concept_stats.items():
        avg_percent = np.mean(pct)
        frequency = len(pct) / len(all_concepts)
        if avg_percent >= percentage_th and frequency >= frequency_th:
            aggregated.append({"concept": cp, "avg_percent": avg_percent, "frequency": frequency})
            # {'concept': 'Robotics', 'avg_percent': 24.4, 'frequency': 1.0, 'avg_percent_normalized': 0.199}

    aggregated = sorted(aggregated, key=lambda x: x["frequency"], reverse=True)[:concepts_limit]
    
    # Normalized "avg_percent" of the aggregated results
    sum_percent = np.sum([cp["avg_percent"] for cp in aggregated])
    for concept in aggregated:
        concept["avg_percent_normalized"] = round(concept["avg_percent"]/sum_percent, 4)
    
    concepts_aggregated = {item["concept"]: item["avg_percent_normalized"] for item in aggregated}
    return concepts_aggregated

def select_wikitext(text):
    """
    Select the general description of a given wikitext.
    
    Parameters:
        text: long wikitext (string)
    Returns:
        the general description in wikitext before sections (string)
    """
    # split the text into initial paragraphs by \n
    paragraphs = text.split('\n')
    # Filter out empty lines and too-short lines
    paragraphs = [para for para in paragraphs if para.strip()]
    
    selected_paragraphs = ""
    for para in paragraphs:
        words_count = len(para.split())
        if words_count >= 5:
            selected_paragraphs += " " + para
        else:
            break
    return selected_paragraphs[1:]

def extract_concepts(text, prompt, llm, repeats=10, agg_by=None):
    """
    The main function to use LLM and prompt for extracting concepts from wiki text.
    
    Parameters:
        text (str): The long wiki text 
        prompt (PROMPT): The object of PROMPT class.
        llm (LLM): The object of LLM class.
        repeats (int): The number of times the LLM generate concepts for a text.
        agg_by (dict): 
        
    Return:
        The aggregated concepts dict
        
    """
    # Put the text into the prompt
    input = prompt.synthesis(text)      

    # Generate respnse for "repeats" times
    output_gen = llm.gen([input for _ in range(repeats)])

    all_concepts = []
    for output in output_gen:
        concepts = extract_concepts_with_percentages(output)
        if prompt.check_validity(concepts):
            # print(f"Append concepts: {concepts}")
            all_concepts.append(concepts)
        else:
            # print(f"Filter out concepts: {concepts}")
            continue

    if len(all_concepts) == 0:
        # print("No concepts appended.")
        return None
    else:
        return all_concepts
    
    """
    else:
        concepts_agg = aggregate_results(all_concepts, **(agg_by if agg_by is not None else {}))
    return concepts_agg
    """


def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def ascii_letter_normalize(x):
  return ' '.join(re.findall(r'[A-Za-z0-9]+', x))

if __name__ == "__main__":
    cuda_id = args.cuda
    parse_file = args.file
    
    WORD_LIMIT=5             # the maximum number of words per concept
    CONCEPT_LIMIT=10         # the maximum number of concepts per text

    REPEATS=10               # the times a 'concepts' is repeatedly generated for a text
    FREQUENCY_TH=0.0         # the minimum fraction of a concept shown in 'REPEATS' times of 'concepts'
    PERCENTAGE_TH=0.0        # the minimum importance percentage of a concept

    MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    DEVICE = torch.device(f"cuda:{cuda_id}")
    RELOAD_MODEL = True
    
    print_(f"Running on {DEVICE}.")

    dataset_dir = "/Memocon/datasets/"
    split_dataset_dir = "/Memocon/datasets/split_by_year/"
    selected_dataset_dir = "/Memocon/datasets/selected/"
    
    split_data_files = [os.path.join(split_dataset_dir, "wiki_en_"+str(year)+".csv") for year in range(2001, 2026)]
    concepts_save_dir = os.path.join(dataset_dir, "concepts", MODEL)

    if RELOAD_MODEL:
        model = AutoModelForCausalLM.from_pretrained(MODEL)
        model.to(DEVICE)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        tokenizer.pad_token_id = tokenizer.eos_token_id

        generation_config = GenerationConfig( 
                                            max_new_tokens=128,
                                            pad_token_id=tokenizer.eos_token_id,
                                            do_sample=True
                                            )

        llm = LLM(MODEL, DEVICE)
        llm.model = model
        llm.tokenizer = tokenizer
        llm.generation_config = generation_config

    prompt = Prompt(word_limit=WORD_LIMIT, concept_limit=CONCEPT_LIMIT)
    agg_by={
            "frequency_th": FREQUENCY_TH, 
            "percentage_th": PERCENTAGE_TH, 
            "concepts_limit": CONCEPT_LIMIT
            }

    parse_files = [
        #'D1_hfq_10000_r1.csv',
        parse_file
        ]
    for file in parse_files:
        wiki_en_selected = pd.read_csv(os.path.join(selected_dataset_dir, file))
        print_(f"Start extracting concepts for {file}, inclding {len(wiki_en_selected)} rows.")
        
        with open(os.path.join(concepts_save_dir, "concepts_"+file.split(".csv")[0]+".json"), 'a') as f:
            for i in range(len(wiki_en_selected)):
                if i % 50 == 0:
                    print_(f"Line {i}, proceeded {round(i/len(wiki_en_selected), 2)*100}%")
                    
                title = wiki_en_selected.loc[i, 'title']
                popularity = wiki_en_selected.loc[i, 'popularity']
                creation_date = wiki_en_selected.loc[i, 'creation_date']
                text = select_wikitext(wiki_en_selected.loc[i, 'text'])

                all_concepts = extract_concepts(text, prompt, llm, 
                                                repeats=REPEATS, 
                                                # agg_by=agg_by
                                                )
                if all_concepts == None:
                    continue
                else:
                    print_(f"Row {i}, {title}, {popularity}, {all_concepts[0]}")
                    json_data = {"title": title,
                                "popularity": popularity,
                                "creation_date": creation_date,
                                "text": text,
                                "all_concepts": all_concepts}
                    
                    json.dump(json_data, f)
                    f.write("\n")