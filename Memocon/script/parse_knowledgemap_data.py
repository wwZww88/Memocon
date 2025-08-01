import sys
from tqdm import tqdm
import pandas as pd

sys.path.append("/Memocon/src")
from KnowledgeChecker import KnowledgeChecker
from Utils import print_

print()
print_('Loading consistency_score df and concept_result df for tqa')

consistency_score =  pd.read_csv("/Memocon/datasets/tqa/tqa_consistency_score_result.csv")
concept_result = pd.read_csv('/Memocon/datasets/tqa/tqa_concept_result.csv')
print_(f'len(consistency_score)={len(consistency_score)}, len(concept_result)={len(concept_result)}')

row_indexs = list(set(consistency_score['row']))
print_(f"len(row_indexs)={len(row_indexs)}")

model = 'meta-llama/Llama-3.1-8B-Instruct'
sim_index = 'psim_bert'

case_score = {
    "T-T": 1,
    "T-F": 0.5,
    "F-T": 0,
    "F-F": 0,
}
print_(f"case_score={case_score}")

kc = KnowledgeChecker(dataframe=consistency_score, case_score=case_score)

"""
with open("/Memocon/datasets/tqa/not_find_row_index.txt", 'r') as f:
    line = f.readline()
not_find = eval(line)
"""

not_find = []
knowledgemap_data = pd.DataFrame(columns=["row", "title", "popularity", "question", "model", 
                                          "pert_data", "score_avg", "score_hst", "concepts_list",  # perturbation result and concepts result
                                          "rows_consistency_score", "rows_concept_result"          # the list of row indexs
                                        ])

# ii = 0
for ri in tqdm(row_indexs):
    """
    ii = ii + 1
    # if (ii-1) % 100 == 0:
        # print(f"Proceeded {ii-1 } items, {round(ii/len(row_indexs),4)*100}%")
    """
          
    # Query sample in consistency_score using its unique row index
    info = consistency_score.query(f"row == {ri}")
    rows_consistency_score = info.index.tolist()
    info = info.iloc[0]
    
    # Find the corresponding sample in concept_result
    find = concept_result[(concept_result['title'] ==  info['title']) & 
                          (concept_result['popularity'] == info['popularity']) & 
                          (concept_result['question'] == info['question']) & 
                          (concept_result['model'] == model)]
    if len(find) == 0:
        not_find.append(ri)
        continue
    
    rows_concept_result = find.index.tolist()
    
    pert_data = kc.get_pert_data(ri, sim_index)
    pert_data = kc.preprocessing(pert_data, sim_index)
    concepts_list = eval(find.iloc[0]['concepts_list'])
    
    score_avg = kc.score_by_average_intensity(pert_data)
    score_hst = kc.score_by_highest_intensity(pert_data)
    
    knowledgemap_data.loc[len(knowledgemap_data)] = [ri, info['title'], info['popularity'], info['question'], model,
                                                     pert_data, score_avg, score_hst, concepts_list, 
                                                     rows_consistency_score, rows_concept_result,
                                                     ]
    
print_(f"len(knowledgemap_data)={len(knowledgemap_data)}, len(not_find)={len(not_find)}")

print_("Saving not_find_row_index to /Memocon/datasets/tqa/not_find_row_index.txt")
with open("/Memocon/datasets/tqa/not_find_row_index.txt", 'w') as f:
    f.write(str(not_find))
    
print_("Saving knowledgemap_data to /Memocon/datasets/tqa/tqa_knowledgemap_data.csv")
knowledgemap_data.to_csv("/Memocon/datasets/tqa/tqa_knowledgemap_data.csv", index=False)

print_("Finished")

"""
Loading consistency_score df and concept_result df for tqa
2025-07-01 15:33:44 len(consistency_score)=205865, len(concept_result)=201379
2025-07-01 15:33:44 len(row_indexs)=41173
2025-07-01 15:33:44 case_score={'T-T': 1, 'T-F': 0.5, 'F-T': 0, 'F-F': 0}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41173/41173 [48:57<00:00, 14.02it/s]
2025-07-01 16:22:42 len(knowledgemap_data)=32339, len(not_find)=8834
2025-07-01 16:22:42 Saving not_find_row_index to /Memocon/datasets/tqa/not_find_row_index.txt
2025-07-01 16:22:42 Saving knowledgemap_data to /Memocon/datasets/tqa/tqa_knowledgemap_data.csv
2025-07-01 16:22:43 Finished
"""