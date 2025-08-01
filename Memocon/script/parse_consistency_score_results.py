import os
import sys
from tqdm import tqdm
from functools import partial

import numpy as np
import pandas as pd

sys.path.append("/Memocon/src")
from Utils import cosine_sim_tfidf, bert_similarity, bleu_similarity, sentence_similarity
from Utils import print_

import warnings
warnings.filterwarnings('ignore')

def assemble_mcq(question, options, options_ids):
    result = question + '\n'
    for i in range(len(options)):
        result += options_ids[i] + ' ' + options[i] + '\n'
    return result

"""
dir = "/Memocon/datasets/tqa/"
files = [
    "/Memocon/datasets/tqa/tqa_saqa_w30_meta-llama_Llama-3.1-8B-Instruct.json",
    "/Memocon/datasets/tqa/tqa_saqa_w50_meta-llama_Llama-3.1-8B-Instruct.json",
    "/Memocon/datasets/tqa/tqa_saqa_w70_meta-llama_Llama-3.1-8B-Instruct.json",
    "/Memocon/datasets/tqa/tqa_saqa_w90_meta-llama_Llama-3.1-8B-Instruct.json",
    "/Memocon/datasets/tqa/tqa_saqa_w120_meta-llama_Llama-3.1-8B-Instruct.json",
]
"""

if __name__ == "__main__":

    dir = "/Memocon/datasets/cb/"
    files = [
        "/Memocon/datasets/cb/cb_mcq_level1_meta-llama_Llama-3.1-8B-Instruct.json",
        "/Memocon/datasets/cb/cb_mcq_level2_meta-llama_Llama-3.1-8B-Instruct.json",
        "/Memocon/datasets/cb/cb_mcq_level3_meta-llama_Llama-3.1-8B-Instruct.json",
        "/Memocon/datasets/cb/cb_mcq_level4_meta-llama_Llama-3.1-8B-Instruct.json",
        "/Memocon/datasets/cb/cb_mcq_level5_meta-llama_Llama-3.1-8B-Instruct.json",
    ]

    consistency_score = pd.DataFrame(columns=['row', 'title', 'popularity', 'model', 'pert_level',
                                            'question', 'question_pert', 'prompt', 'prompt_pert',
                                            'qsim_tfidf', 'qsim_bert', 'qsim_bleu', 'qsim_gn',
                                            'psim_tfidf', 'psim_bert', 'psim_bleu', 'psim_gn', 
                                            'correctness_origin', 'correctness_perturb',
                                            ])

    method_list = [cosine_sim_tfidf, bert_similarity, bleu_similarity, sentence_similarity]

    for file in files:
        print_(f"Procsssing {file}")
        with open(file, 'r') as f:
            lines = f.readlines()
        items = [eval(line[3:-4]) for line in lines]
        print_(f"Length: {len(items)}.")
        
        level = file.split('_')[2]

        i = 0
        for item in items:
            row = item['row']
            title = item['title']
            popularity = item['popularity']
            model = item['model']
            
            if i % 500 == 0:
                print_(f"row {i}, {round(i/len(items), 2)*100}%")
            
            # question = item['saqa']['question']
            # question_pert = item['saqa_pert']['question']
            
            question = assemble_mcq(item['question'], item['mcq']['options'], item['mcq']['option_ids'])
            question_pert = assemble_mcq(item['question'], item['mcq_pert']['options'], item['mcq_pert']['option_ids'])
            prompt = item['mcq']['prompt']
            prompt_pert = item['mcq_pert']['prompt']
            
            correctness_origin = item['mcq']['correctness']
            correctness_perturb = item['mcq_pert']['correctness']
            
            qsim = {}
            psim = {}
            for method in method_list:
                try:
                    qsim[method.__name__] = method(question, question_pert)
                except:
                    qsim[method.__name__] = None
                try:
                    psim[method.__name__] = method(prompt, prompt_pert)
                except:
                    psim[method.__name__] = None
            
            consistency_score.loc[len(consistency_score)] = [row, title, popularity, model, level,
                                                question, question_pert, prompt, prompt_pert,
                                                qsim['cosine_sim_tfidf'], qsim['bert_similarity'], qsim['bleu_similarity'], qsim['sentence_similarity'],
                                                psim['cosine_sim_tfidf'], psim['bert_similarity'], psim['bleu_similarity'], psim['sentence_similarity'], 
                                                correctness_origin, correctness_perturb,
                                                ]
            i += 1
            
        print_(f"Items in consistency_score: {len(consistency_score)}.\n")
        
    # print_("Saving result to tqa_consistency_score_result.csv")
    # consistency_score.to_csv(os.path.join(dir, "tqa_consistency_score_result.csv"), index = False)
    
    print_("Saving result to cb_consistency_score_result.csv")
    consistency_score.to_csv(os.path.join(dir, "cb_consistency_score_result.csv"), index = False)
    
    print_("Finished!")