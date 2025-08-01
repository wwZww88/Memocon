import os
import re
import sys
import json
import copy
import random
import argparse
from importlib import reload
from datasets import load_dataset
from typing import List, Callable, Union

sys.path.append("/Memocon/src")
from Utils import print_, process_items
from LLM import LLM
from Perturbation import ShortAnswerQA, PerturbShortAnswerQA

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', dest='model', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument('-o', '--direct', dest='direct', type=str, default="forward")   # Iterate forward or backward
parser.add_argument('-s', '--start', dest='start', type=int, default=None)          # The starting position
parser.add_argument('-e', '--end', dest='end', type=int, default=None)              # The ending position
parser.add_argument('-c', '--cuda', dest='cuda', type=str, default="7")             # The cuda id to run this script
parser.add_argument('-w', '--words', dest='words', type=int, default="70")          # Length of irrelavance context
args = parser.parse_args()    

def parse_output(output: str) -> dict:
    output = re.sub(r'\n', ' ', output)
    output = re.findall(r'[{]\s*"[^{]*[}]', output)[0]
    output = eval(output)
    return output

def validate_format(output: str) -> bool:
    """Validate if the LLM response followed the instructed format."""
    try:
        _ = parse_output(output)
        return True
    except:
        return False
    
def check_correctness(saqa: ShortAnswerQA, output: dict) -> bool:
    """Check if the answer of the LLM is correct."""
    try:
        llm_answer = output["answer"].lower()
    except:
        try:
            llm_answer = output["answer"]
        except:
            return False
    if llm_answer in saqa.answer:
        return True
    else:
        return False
    
def pertubation_result(saqa: ShortAnswerQA, llm):
    """The correctness of saqa answered by llm."""
    input = saqa.get_prompt()
    
    correctness = False
    response_ok = False
    max_retry = 5
    n_retry = 1
    while response_ok is False and n_retry <= max_retry:
        output = llm.gen(input)
        if validate_format(output):
            output = parse_output(output)
            correctness = check_correctness(saqa, output)
            response_ok = True
        else:
            n_retry += 1
            continue 
    return correctness, output

if __name__ == "__main__": 
    model = args.model
    direct = args.direct
    start = args.start
    end = args.end
    cuda_id = args.cuda
    words = args.words
    
    MODEL = model
    DEVICE = cuda_id
    llm = LLM(MODEL, DEVICE)
    print_("Model running on cuda:{DEVICE}.")

    print_("Loading 'trivia_qa' dataset and 'popularity' dataset.")
    
    tqa_dir = "/Memocon/datasets/tqa/"
    with open(os.path.join(tqa_dir, "tqa_popularity_sum_sorted.json"), 'r') as f:
        line = f.readline()
    tqa_popularity = eval(line)
    tqa_popularity = {item[0]:item[1] for item in tqa_popularity}
    
    ds = load_dataset("mandarjoshi/trivia_qa", "rc")["train"]
    print_(f"len(ds)={len(ds)}, len(tqa_popularity)={len(tqa_popularity)}.")
    
    saqa = ShortAnswerQA()
    psaqa = PerturbShortAnswerQA()
    psaqa.generator = llm
    psaqa.datalib = ds
    
    print_(f"Start running SAQA pertubation test for {MODEL}.")
    print_(f"Length of irrelavent context set to {words}.\n")
    print_(f"Running start from {start}.")
    
    indices = process_items(ds, direct=direct, start=start, end=end)
    
    with open(os.path.join(tqa_dir, f"tqa_saqa_w{words}_{MODEL.replace('/', '_')}.json"), 'a') as f:
        for i in indices:
            data = ds[i]
            title = data['entity_pages']['title']
            if len(title) == 0:
                continue
            popularity = {t:tqa_popularity[t] for t in title if t in tqa_popularity.keys()}
            print_(f"Row {i}, {title}, {popularity}")
            print(data['question'])
            
            saqa.load_dict(data)
            saqa_pert = psaqa.mixperturb(saqa, target_words=words)
            
            output = llm.gen(saqa.get_prompt())
            output_pert = llm.gen(saqa_pert.get_prompt())

            correctness, output = pertubation_result(saqa, llm)
            correctness_pert, output_pert = pertubation_result(saqa_pert, llm)
            
            print(f"saqa: {output}, {correctness}.")
            print(f"output_pert: {output_pert}, {correctness_pert}")

            data = {
                "row": i,
                "title": title,
                "popularity": popularity,
                "question": saqa.question,
                "answer": saqa.answer,
                "model": MODEL,
                "saqa":{
                    "question": saqa.question,
                    "prompt": saqa.get_prompt(),
                    "correctness": correctness,
                    "original_response": output,
                },
                "saqa_pert":{
                    "question": saqa_pert.question,
                    "prompt": saqa_pert.get_prompt(),
                    "correctness": correctness_pert,
                    "original_response": output_pert,
                }, 
            }

            f.write('"""' + str(data) + '"""')
            f.write("\n")




