import pandas as pd
from typing import List, Dict, Tuple, Callable, Union

levels = ['w30', 'w50', 'w70', 'w90', 'w120']

def get_pert_data(consistency_score:pd.DataFrame, row_index:int, sim_index:str) -> Dict[str, dict]:
    """Load pert_data from consistency_score_result.csv dataframe."""
    
    pert_data = {level: {'intensity': None, 'consistency' : None} for level in levels}
    for level in levels:
        find = consistency_score.query(f"row == {row_index} and pert_level == '{level}'").squeeze()
        if find.empty:
            print(f"row {row_index}, pert_level={level} not found")
            continue
        else:
            pert_data[level]['intensity'] = find[sim_index]
            pert_data[level]['consistency'] = ('T' if find['correctness_origin'] else 'F') + '-' + ('T' if find['correctness_perturb'] else 'F')
            
    return pert_data

def row_to_dict(dfrow:pd.Series) -> Dict[str, dict]:
    """Load pert_data from score_{sim_index}.csv dataframe row."""
    
    pert_data = {level: {'intensity': None, 'consistency' : None} for level in levels}
    for level in levels:
        pert_data[level]['intensity'] = dfrow[f"{level}_intensity"]
        pert_data[level]['consistency'] = dfrow[f"{level}_consistency"]
        
    return pert_data

def max_min(dataframe:pd.DataFrame) -> Tuple[float, ...]:
    """Retrived the maximun and minimun similarity values among all levels of perturabtions."""
    extremes = []
    for level in levels:
        describe = dataframe[f"{level}_intensity"].describe()
        extremes += [describe['max'], describe['min']]
    return max(extremes), min(extremes)

def preprocessing(pert_data:Dict[str, dict], vmax:float, vmin:float) -> Dict[str, dict]:
    """
    Preprocess the similarity values by
        1. Intensity = 1 - Similarity, since more similar -> less difficult -> weaker intensity
        2. Normalize intensity to [0,1], x'=(x-min)/(max-min)
    """
    for level in pert_data:
        pert_data[level]['intensity'] = (pert_data[level]['intensity'] - vmin) / (vmax - vmin)
        pert_data[level]['intensity'] = 1- pert_data[level]['intensity']
    return pert_data

# Score Strategy 1：sigma(diff_i*score_i)/n_correct
def score_by_average_intensity(pert_data:Dict[str, dict], case_score:Dict[str, float]) -> float:
    """Averaged resisted intensity."""
    n = 0
    sum = 0
    for level, data in pert_data.items():
        if case_score[data['consistency']] == 0:
            continue
        else:
            n += 1
            sum += data['intensity'] * case_score[data['consistency']]

    if n == 0:
        return 0
    else:
        return sum/n

# Score Strategy 2：Select the best performance one
def score_by_highest_intensity(pert_data:Dict[str, dict], case_score:Dict[str, float]) -> float:
    """"Highest resisted intensity."""
    
    def sort_key(item):
        # Sort pert_data by: 1. case_score, 2. intensity 
        key, value = item
        consistency = value['consistency']
        intensity = value['intensity']
        score = case_score[consistency]
        return (-score, -intensity) 

    pert_data_sorted = tuple(sorted(pert_data.items(), key=sort_key))
    
    _, best = pert_data_sorted[0]
    if case_score[best['consistency']] == 0:
        return 0
    else:
        return best['intensity'] * case_score[best['consistency']]
    
def scoring(row_index:int, sim_index:str, case_score:dict, method:Callable, dataframe:pd.DataFrame) -> float:
    """
    Compute the consistency score s for a sample.
    
    Parameters:
        dataframe:DataFrame, consistency_score
        row_index:int, the unique indicator of sample
        sim_index:str, similarity measurement of 'question' or 'prompt'
        method:func, the functions to compute the consistency score
    """
    pert_data = get_pert_data(dataframe, row_index, sim_index)
    pert_data = preprocessing(pert_data, vmax, vmin)
    score = method(pert_data, case_score)
    return score

class KnowledgeChecker:
    def __init__(self, dataframe:pd.DataFrame, case_score:Dict[str, float]):
        self.dataframe = dataframe
        self.case_score = case_score
        
        self.levels = ['w30', 'w50', 'w70', 'w90', 'w120']
        self.sim_index_list = ['qsim_tfidf', 'qsim_bert', 'qsim_bleu', 'qsim_gn',
                          'psim_tfidf', 'psim_bert', 'psim_bleu', 'psim_gn']
        
        self.vmax, self.vmin = self.max_min()
        # print(self.vmax, self.vmin)
        
        
    def max_min(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Retrived the maximun and minimun similarity values among all types of similarity measurement."""
        extremes = {sim_index:[] for sim_index in self.sim_index_list}
        for sim_index in self.sim_index_list:
            describe = self.dataframe[sim_index].describe()
            extremes[sim_index] += [describe['max'], describe['min']]
            
        vmax = {}
        vmin = {}
        for sim_index, values in extremes.items():
            vmax[sim_index] = max(values)
            vmin[sim_index] = min(values)
            
        """
        extremes = []
        for level in levels:
            describe = self.dataframe[f"{level}_intensity"].describe()
            extremes += [describe['max'], describe['min']]
        """
        return vmax, vmin
    
    def get_pert_data(self, row_index:int, sim_index:str) -> Dict[str, dict]:
        """Load pert_data from consistency_score_result.csv dataframe."""
        
        pert_data = {level: {'intensity': None, 'consistency' : None} for level in self.levels}
        for level in self.levels:
            find = self.dataframe.query(f"row == {row_index} and pert_level == '{level}'").squeeze()
            if find.empty:
                print(f"row {row_index}, pert_level={level} not found")
                continue
            else:
                pert_data[level]['intensity'] = find[sim_index]
                pert_data[level]['consistency'] = ('T' if find['correctness_origin'] else 'F') + '-' + ('T' if find['correctness_perturb'] else 'F')
                
        return pert_data
        
    def preprocessing(self, pert_data:Dict[str, dict], sim_index:str) -> Dict[str, dict]:
        """
        Preprocess the similarity values by
            1. Intensity = 1 - Similarity, since more similar -> less difficult -> weaker intensity
            2. Normalize intensity to [0,1], x'=(x-min)/(max-min)
        """
        # Initialize result as pert_data
        result = {level: {'intensity': pert_data[level]['intensity'], 'consistency' : pert_data[level]['consistency']} for level in pert_data}
        
        for level in pert_data:
            # Normalization
            result[level]['intensity'] = (result[level]['intensity'] - self.vmin[sim_index]) / (self.vmax[sim_index] - self.vmin[sim_index])
            
            # Similarity to Intensity
            result[level]['intensity'] = 1- result[level]['intensity']
        return result
        
    def score_by_average_intensity(self, pert_data:Dict[str, dict]) -> float:
        """Averaged resisted intensity."""
        n = 0
        sum = 0
        for level, data in pert_data.items():
            if self.case_score[data['consistency']] == 0:
                continue
            else:
                n += 1
                sum += data['intensity'] * self.case_score[data['consistency']]

        if n == 0:
            return 0
        else:
            return sum/n
        
    def score_by_highest_intensity(self, pert_data:Dict[str, dict]) -> float:
        """"Highest resisted intensity."""
        
        def sort_key(item):
            # Sort pert_data by: 1. case_score, 2. intensity 
            key, value = item
            consistency = value['consistency']
            intensity = value['intensity']
            score = self.case_score[consistency]
            return (-score, -intensity) 

        pert_data_sorted = tuple(sorted(pert_data.items(), key=sort_key))
        
        _, best = pert_data_sorted[0]
        if self.case_score[best['consistency']] == 0:
            return 0
        else:
            return best['intensity'] * self.case_score[best['consistency']]
        
    def scoring(self, row_index:int, sim_index:str, method:str='average') -> float:
        """
        Compute the consistency score s for a sample.
        
        Parameters:
            dataframe:DataFrame, consistency_score
            row_index:int, the unique indicator of sample
            sim_index:str, similarity measurement of 'question' or 'prompt'
            method:func, the functions to compute the consistency score
        """
        pert_data = self.get_pert_data(row_index, sim_index)
        pert_data = self.preprocessing(pert_data, sim_index)
        if method == 'average':
            score = self.score_by_average_intensity(pert_data)
        elif method == 'highest':
            score = self.score_by_highest_intensity(pert_data)

        return score

if __name__ == "__main__":
    
    consistency_score =  pd.read_csv("/Memocon/datasets/tqa/tqa_consistency_score_result.csv")
    case_score = {
        "T-T": 1,
        "T-F": 0.5,
        "F-T": 0,
        "F-F": 0,
    }
    
    kc = KnowledgeChecker(dataframe=consistency_score, case_score=case_score)
    
    row_index = 3
    sim_index = 'psim_bert'
    
    pert_data = kc.get_pert_data(row_index, sim_index)
    print('\n', pert_data, '\n')
    
    pert_data_p = kc.preprocessing(pert_data, sim_index)
    print(pert_data_p, '\n')
    
    score_average = kc.score_by_average_intensity(pert_data_p)
    score_highest = kc.score_by_highest_intensity(pert_data_p)
    print(score_average, score_highest, '\n')
    
    score = kc.scoring(row_index, sim_index, 'average')
    print(score)
    
    """
    sim_index = 'psim_bert'
    consistency_score =  pd.read_csv("/Memocon/datasets/tqa/tqa_consistency_score_result.csv")
    score = pd.read_csv(f"/Memocon/datasets/tqa/tqa_score_{sim_index}.csv")

    row_index = list(set(consistency_score['row']))
    print(f'Length of row_index: {len(row_index)}')

    levels = ['w30', 'w50', 'w70', 'w90', 'w120']

    # Set inconsistency case score.
    case_score = {
        "T-T": 1,
        "T-F": 0.5,
        "F-T": 0,
        "F-F": 0,
    }

    vmax, vmin = max_min(score)

    print("Result under 'average_intensity' and 'highest_intensity':")
    for row_index in row_index[:10]:

        #pert_data_1 = row_to_dict(score.query(f"row == {row_index}").squeeze())
        #pert_data_2 = get_pert_data(consistency_score, row_index, 'psim_tfidf')
        #print(pert_data_1['w30']['intensity'], pert_data_1['w30']['consistency'])
        #print(pert_data_2['w30']['intensity'], pert_data_1['w30']['consistency'])

        
        pert_data = get_pert_data(consistency_score, row_index, sim_index)
        print(f"row-{row_index}", "%.4f, %.4f" % (score_by_average_intensity(pert_data, case_score), score_by_highest_intensity(pert_data, case_score)))
        print(score(row_index=row_index, sim_index=sim_index, case_score=case_score, method=score_by_average_intensity, dataframe=consistency_score))
    """