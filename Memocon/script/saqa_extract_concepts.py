import os
import sys
import json
import argparse
from datasets import load_dataset

sys.path.append("/Memocon/src")
from Utils import print_, process_items
from Prompt import Prompt, extract_concepts_with_percentages
from LLM import LLM

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', dest='model', type=str, default=None)#"meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument('-o', '--direct', dest='direct', type=str, default="forward")   # Iterate forward or backward
parser.add_argument('-s', '--start', dest='start', type=int, default=None)          # The starting position
parser.add_argument('-e', '--end', dest='end', type=int, default=None)              # The ending position
parser.add_argument('-c', '--cuda', dest='cuda', type=str, default="7")             # The cuda id to run this script
args = parser.parse_args()

if __name__ == "__main__":
    model = args.model
    direct = args.direct
    start = args.start
    end = args.end
    cuda_id = args.cuda
    
    REPEATS = 10
    
    MODEL = model
    DEVICE = cuda_id
    llm = LLM(MODEL, DEVICE)
    print_("Model running on cuda:{DEVICE}.")
    
    WORD_LIMIT = 2 
    CONCEPT_LIMIT = 10 
    prompt = Prompt(word_limit=WORD_LIMIT, concept_limit=CONCEPT_LIMIT)
    prompt.template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional text analysis tool specialized in extracting Wikipedia-title-style concepts from text. Your task is to identify the most specific, notable entities (people, events, works, etc.) that could have their own Wikipedia pages, and evaluate their importance percentages according to these rules:
1. Extract only specific, notable concepts that would qualify as Wikipedia article titles (e.g., "Bill Clinton" not just "President")
2. Prioritize proper nouns and specific named entities over general terms
3. Each concept should be exactly as it would appear as a Wikipedia title (proper capitalization, full names when available)
4. Assign an importance percentage to each concept (sum must be 100%)
5. Concepts should typically be 1-{WORD_LIMIT} words unless the full proper name requires more
6. Maximum {KEYWORD_LIMIT} concepts

Response format MUST be:
1. Exact_Wikipedia_Title_1 (importance_1%)
2. Exact_Wikipedia_Title_2 (importance_2%)
...
NO additional text or explanations.

Example 1:
INPUT: "President Bill Clinton awarded what former president a posthumous Medal of Honor, the only president to have received one?"
OUTPUT:
1. Bill Clinton (60%)
2. Medal of Honor (40%)

Example 2:
INPUT: "The Mona Lisa, painted by Leonardo da Vinci during the Italian Renaissance, is displayed at the Louvre Museum in Paris."
OUTPUT:
1. Mona Lisa (40%)
2. Leonardo da Vinci (30%)
3. Italian Renaissance (15%)
4. Louvre (15%)

<|start_header_id|>user<|end_header_id|>
Extract Wikipedia-style concepts from:
INPUT_TEXT:
{INPUT_TEXT} <|eot_id|>

<|start_header_id|>assistant<|end_header_id|>"""
    
    print_("Loading 'trivia_qa' dataset and 'popularity' dataset.")
    
    tqa_dir = "/Memocon/datasets/tqa/"
    with open(os.path.join(tqa_dir, "tqa_popularity_sum_sorted.json"), 'r') as f:
        line = f.readline()
    tqa_popularity = eval(line)
    tqa_popularity = {item[0]:item[1] for item in tqa_popularity}
    
    ds = load_dataset("mandarjoshi/trivia_qa", "rc")["train"]
    print_(f"len(ds)={len(ds)}, len(tqa_popularity)={len(tqa_popularity)}.")
    
    print_(f"Start running concept extraction for SAQA with {MODEL}.")
    print_(f"Using prompt: {prompt.template}\n")
    
    print_(f"Running start from {start}.")
    
    indices = process_items(ds, direct=direct, start=start, end=end)
    
    with open(os.path.join(tqa_dir, f"tqa_concepts_{MODEL.replace('/', '_')}.json"), 'a') as f:
        for i in indices:
            data = ds[i]
            title = data['entity_pages']['title']
            question = data['question']
            
            if len(title) == 0:
                continue
            popularity = {t:tqa_popularity[t] for t in title if t in tqa_popularity.keys()}
            
            print_(f"Row {i}, {title}, {popularity}")
            print(question)
            
            text = question
            input = prompt.synthesis(text)
            concepts_list = []
            for i in range(REPEATS):
                output = llm.gen(input, max_new_tokens=100)
                concepts = extract_concepts_with_percentages(output)
                if prompt.check_validity(concepts):
                    concepts_list.append(concepts)

            if len(concepts_list) != 0:
                print(f"Valid concepts: {len(concepts_list)}, {concepts_list[0]}")
            else:
                print(f"No valid concepts.")
            
            data = {
                "title": title,
                "popularity": popularity,
                "question": question,
                "model": MODEL,
                "num_concepts": len(concepts_list),
                "concepts_list": concepts_list,
            }
            f.write('"""' + str(data) + '"""')
            f.write("\n")