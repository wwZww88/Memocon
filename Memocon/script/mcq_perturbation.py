import os
import re
import sys
import json
import torch
import argparse
from tqdm import tqdm
from random import randint
from importlib import reload

sys.path.append("/Memocon/src")
from Utils import print_, process_items
from LLM import LLM
from DataLoader import load_cb
from Perturbation import MultipleChoiceQA, MixedPerturbMultiChoiceQA

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', dest='model', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
parser.add_argument('-o', '--direct', dest='direct', type=str, default="forward")   # File name in selected_dataset_dir
parser.add_argument('-s', '--start', dest='start', type=int, default=None)          # File name in selected_dataset_dir
parser.add_argument('-e', '--end', dest='end', type=int, default=None)              # File name in selected_dataset_dir
parser.add_argument('-c', '--cuda', dest='cuda', type=str, default="7")             # The cuda id to run this script
parser.add_argument('-l', '--level', dest='level', type=str, default="level1")           # Perturbation Strength level
args = parser.parse_args()

def load_mcq(metadata):
    mcq = MultipleChoiceQA(
        question=metadata["question"],
        options=metadata["options"],
        answer=metadata["correct_option"],
        relation_id=metadata["relation"],
    )
    return mcq

def save_progress():
    # save progress of iteration
    with open(checkpoint_file, 'w') as f:
        f.write('%d' % checkpoint)
    print_("Save checkpoint.txt")
    
def parse_output(output):
    output = re.sub(r'\n', ' ', output)
    output = re.findall(r'[{]\s*"[^{]*[}]', output)[0]
    output = eval(output)
    return output

def check_correctness(mcq, output):
    """Check if the answer of the LLM is correct."""
    llm_answer = output["answer"]
    
    result = [False] * len(mcq.answer)
    for i in range(len(mcq.option_ids)):
        if mcq.option_ids[i] in llm_answer:
            result[i] = True
    
    if result == mcq.answer:
        return True
    else:
        return False
    
def validate_format(output):
    """Validate if the LLM response followed the instructed format."""
    try:
        _ = parse_output(output)
        return True
    except:
        return False
    
def pertubation_result(mcq, llm):
    """The correctness of mcq answered by llm."""
    input = mcq.get_prompt()
    
    correctness = False
    response_ok = False
    max_retry = 10
    n_retry = 1
    while response_ok is False and n_retry <= max_retry:
        output = llm.gen(input)
        if validate_format(output):
            output = parse_output(output)
            correctness = check_correctness(mcq, output)
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
    level = args.level
    
    # MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    MODEL = model
    DEVICE = cuda_id
    llm = LLM(MODEL, DEVICE)
    print_("Model running on cuda:{DEVICE}.")

    print_("Loading 'cb' dataset and 'popularity' dataset.")
    cb_dir = "/Memocon/datasets/cb/"
    with open(os.path.join(cb_dir, "cb_popularity_sum_sorted.json"), 'r') as f:
        line = f.readline()
    cb_popularity = eval(line)
    cb = load_cb()
    print_(f"len(cb_popularity)={len(cb_popularity)}")

    level1 = [
         "OptionIndice",  
        ("OptionFormat", {"method": "random"}) ,
        ]

    level2 = [
        ("OptionAdd", {"num_new_options": 1}), 
        ("OptionFormat", {"method": "random"}),
         "OptionPermutation", 
        ]

    level3 = [
        ("OptionAdd", {"num_new_options": 2}), 
         "OptionIndice",
         "OptionPermutation",
        ("OptionFormat", {"method": "random"}),
        ]
    
    level4 = [
        ("OptionAdd", {"num_new_options": 3}),
        ("Caesar", {"delta": randint(1, 20)}),
         "OptionPermutation",
        ("OptionFormat", {"method": "random"}),          
    ]

    level5 = [
        ("OptionAdd", {"num_new_options": 3}),
        ("Caesar", {"delta": randint(1, 20)}),
         "OptionIndice",    
         "OptionPermutation",
        ("OptionFormat", {"method": "random"}),          
    ]
    
    settings = {
            "level1": level1,
            "level2": level2,
            "level3": level3,
            "level4": level4,
            "level5": level5,
        }
    
    mpmcq = MixedPerturbMultiChoiceQA(perturbations=settings[level])

    print_(f"Start running MCQ pertubation test for {MODEL}.")
    print_(f"Pertubation: {mpmcq.__str__()}.")
    print_(f"Running start from {start}.")

    indices = process_items(cb_popularity, direct=direct, start=start, end=end)
    
    with open(os.path.join(cb_dir, f"cb_mcq_{level}_{MODEL.replace('/', '_')}.json"), 'a') as f:
        for i in indices:
            title = cb_popularity[i][0]
            popularity = cb_popularity[i][1]
            metadata = cb[title]
            
            print_(f"Row {i}, {title}, {popularity}, {metadata['question']}")

            mcq = load_mcq(metadata)
            mcq_pert = mpmcq.mixperturb(mcq)
            
            correctness, output = pertubation_result(mcq, llm)
            correctness_pert, output_pert = pertubation_result(mcq_pert, llm)
            
            print(f"mcq: {output}, {correctness}.")
            print(f"mcq_pert: {output_pert}, {correctness_pert}")

            data = {
                "row": i,
                "title": title,
                "popularity": popularity,
                "question": mcq.question,
                "model": MODEL,
                "pertubation": mpmcq.__str__(),
                "mcq":{
                    "options": mcq.options,
                    "option_ids": mcq.option_ids,
                    "answer": mcq.answer,
                    "prompt": mcq.get_prompt(),
                    "correctness": correctness,
                    "original_response": output,
                },
                "mcq_pert":{
                    "options": mcq_pert.options,
                    "option_ids": mcq_pert.option_ids,
                    "answer": mcq_pert.answer,
                    "prompt": mcq_pert.get_prompt(),
                    "correctness": correctness_pert,
                    "original_response": output_pert,
                },   
            }

            f.write('"""' + str(data) + '"""')
            f.write("\n")
            
            # print(mcq.options, mcq.option_ids, mcq.answer)
            # print(mcq_pert.options, mcq_pert.option_ids, mcq_pert.answer)
            # print('\n')



