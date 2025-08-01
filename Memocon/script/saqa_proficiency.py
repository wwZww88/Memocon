import os
import sys
from tqdm import tqdm
import pandas as pd

sys.path.append("/Memocon/src")

from KnowledgeExtracter import aggregate_results
from KnowledgeChecker import KnowledgeChecker
from KnowledgeUpdater import KnowledgeUpdater
from Utils import print_

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    
    save_dir = "/Memocon/datasets/tqa/proficiency_result"
    
    knowledgemap_data = pd.read_csv("/Memocon/datasets/tqa/tqa_knowledgemap_data.csv")
    consistency_score = pd.read_csv("/Memocon/datasets/tqa/tqa_consistency_score_result.csv")

    # Init KnowledgeChecker
    sim_index = 'psim_bert'
    case_score = {
        "T-T": 1,
        "T-F": 0.5,
        "F-T": 0,
        "F-F": 0,
    }
    
    intensity_mode = 'highest'
    
    if intensity_mode == 'highest':
        _avg = 0.31
    elif intensity_mode == 'average':
        _avg = 0.19
        
    parameter = {
        "n_concept": 3.5*(10**4),
        "theta": _avg,                
        "p_init": 0.5,
        "decay_rate": 1,
        "window_size": 10,
        "lr_min": 0.1,
        "lr_max": 1,
        "lr_default": 0.8,
        "v_default": 0.8,
    } 
    
    kc = KnowledgeChecker(dataframe=consistency_score, case_score=case_score)

    # Initialize KnowledgeUpdater 
    km = KnowledgeUpdater()
    km.load_param(parameter)

    print_("Running proficiency result for TriviaQA")
    print(f"case_score={case_score}\n")
    print(f"score_by_{intensity_mode}_intensity")
    print(f"parameter={parameter}")
    
    for i in tqdm(range(len(knowledgemap_data))):

        #print(f"{knowledgemap_data.loc[i, 'title']}: {knowledgemap_data.loc[i, 'popularity']}")
        #print(f"question: {knowledgemap_data.loc[i, 'question']}")
        
        pert_data = eval(knowledgemap_data.loc[i, 'pert_data'])
        if intensity_mode == 'highest':
            s = kc.score_by_highest_intensity(pert_data)
        elif intensity_mode == 'average':
            s = kc.score_by_average_intensity(pert_data)
        
        concepts_list = eval(knowledgemap_data.loc[i, 'concepts_list'])
        e = aggregate_results(concepts_list)
        
        p_ = km.update(e, s)

    p_sorted = dict(sorted(km.p.items(), key=lambda item: item[1], reverse=True))
    #print(f"p={p_sorted}")
    
    with open(os.path.join(save_dir, f"proficiency_vector_{intensity_mode}_theta{parameter['theta']}_init{parameter['p_init']}_decay_rate{parameter['decay_rate']}_window_size{parameter['window_size']}_lr_min{parameter['lr_min']}_lr_max{parameter['lr_max']}_lr_default{parameter['lr_default']}_v_default{parameter['v_default']}.txt"), 'w') as f:
        f.write("Proficiency - TriviaQA\n")
        f.write(f"case_score={case_score}\n")
        f.write(f"score_by_highest_intensity\n")
        f.write(f"parameter={km.param_dict()}\n")
        f.write(f"proficiency={p_sorted}\n")