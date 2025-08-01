import os
import sys
import time
import numpy as np
import pandas as pd
from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

"""Similarity Measurement"""

# TF-IDF
def cosine_sim_tfidf(text1, text2):
    corpus = [text1, text2]
    vectorizer = TfidfVectorizer().fit_transform(corpus)
    return cosine_similarity(vectorizer[0], vectorizer[1])[0][0]

# BERT
# model_bert = SentenceTransformer('all-MiniLM-L6-v2')
def bert_similarity(text1, text2, model):
    embeddings = model.encode([text1, text2])
    return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

# BLEU
def bleu_similarity(text1, text2):
    ref = [text1.split()]
    hyp = text2.split()
    return sentence_bleu(ref, hyp)

# GoogleNews
# model_w2v = KeyedVectors.load_word2vec_format('/GoogleNews-vectors-negative300.bin', binary=True)
def sentence_similarity(text1, text2, model):
    vec1 = np.mean([model[word] for word in text1.split() if word in model], axis=0)
    vec2 = np.mean([model[word] for word in text2.split() if word in model], axis=0)
    return cosine_similarity([vec1], [vec2])[0][0]

def similarity(text_list, method=cosine_sim_tfidf):
    print(method.__name__)
    for comb in combinations(text_list, 2):
        text1 = comb[0]
        text2 = comb[1]
        sim = method(text1, text2)
        print(f"{text1[:5]}-{text2[:5]}: {sim}")
        
def pair_wise_similarity(text_list):
    method_list = [cosine_sim_tfidf, bert_similarity, bleu_similarity, sentence_similarity]
    for method in method_list:
        similarity(text_list, method)
        
"""Print Format"""

def print_(text=''):
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), text)
    
def format(variable, precision=2):
    """Convert variable to a predifined precision for output aesthetics."""
    if isinstance(variable, (int, float)):
        return round(variable, precision)
    elif isinstance(variable, (list, (np.ndarray))):
        return [round(v, precision) for v in variable]
    else:
        return variable
    
def process_items(items, direct="forward", start=None, end=None):
    n = len(items)
    if direct.lower() == "forward":
        if start is None:
            start = 0
        if end is None:
            end = n
        indices = range(start, end)
    elif direct.lower() == "backward":
        if start is None:
            start = n - 1
        if end is None:
            end = -1
        indices = range(start, end, -1)
    else:
        raise ValueError("Invalid value, direct must be 'forward' or 'backward'")
    return indices

"""Statistics"""

def distribution(column, range_min=0, range_max=2000, bin_size=100):
    """
    Analyze popularity ststistics
    
    Parameters:
        column: pandas Series, the column to analyze
        range_min: int, minimum value of the analysis range
        range_max: int, maximum value of the analysis range
        bin_size: int, size of each interval/bin
    """

    print("="*40)
    print(f"Basic statistics for column '{column.name}':")
    print("="*40)
    
    stats = {
        'Total count': len(column),
        'Non-null count': column.count(),
        'Null count': column.isnull().sum(),
        'Mean': column.mean(),
        'Median': column.median(),
        'Std dev': column.std(),
        'Variance': column.var(),
        'Min': column.min(),
        '25% quantile': column.quantile(0.25),
        '50% quantile': column.quantile(0.5),
        '75% quantile': column.quantile(0.75),
        'Max': column.max()
    }
    
    for key, value in stats.items():
        print(f"{key:<15}: {value:>10.2f}" if isinstance(value, (int, float)) else f"{key:<15}: {value:>10}")

    print("\n" + "="*60)
    print(f"Interval distribution for '{column.name}' ({range_min}-{range_max}, bin size={bin_size}):")
    print("="*60)

    bins = np.arange(range_min, range_max + bin_size, bin_size)
    labels = [f"{i}-{i+bin_size-1}" for i in bins[:-1]]
    
    grouped = pd.cut(column, bins=bins, labels=labels, right=False, include_lowest=True)

    distribution = grouped.value_counts().sort_index()
    total = distribution.sum()
    
    print(f"{'Interval':<15} {'Count':>10} {'Percentage':>10}")
    print("-"*40)
    for interval, count in distribution.items():
        percentage = count / total * 100
        print(f"{interval:<15} {count:>10} {percentage:>9.1f}%")

    below_range = (column < range_min).sum()
    above_range = (column > range_max).sum()
    
    print("-"*60)
    print(f"{f'<{range_min}':<15} {below_range:>10} {below_range/len(column)*100:>9.1f}%")
    print(f"{f'>{range_max}':<15} {above_range:>10} {above_range/len(column)*100:>9.1f}%")
    print("="*60)
    
def save_stat(column, filename, **kwargs):
    from io import StringIO
    buffer = StringIO()

    original_stdout = sys.stdout
    sys.stdout = buffer
    distribution(column, **kwargs)
    sys.stdout = original_stdout

    content = buffer.getvalue()
    with open(filename, 'w') as f:
        f.write(content)

    print(content)

if __name__ == "__main__": 
    text1 = "text1 In 1999 Anna Kournikova signed a lucrative contract to model what?"
    text2 = "text2 In 1994 and 1995, gangs of armed youth destroyed the homes of foreign nationals living in Johannesburg, demanding that the police work to repatriate them to their home countries. What did Anna Kournikova sign a lucrative contract to model in 1999?"
    text3 = "text3 Python is a great language for programming, I love programming in Python! In 1999 Anna Kournikova signed a lucrative contract to model what?"

    text_list = [text1, text2, text3]
    pair_wise_similarity(text_list)
    
    # SAQA
    text1 = "text1 In 1999 Anna Kournikova signed a lucrative contract to model what?"
    text2 = "text2 In 1994 and 1995, gangs of armed youth destroyed the homes of foreign nationals living in Johannesburg, demanding that the police work to repatriate them to their home countries. What did Anna Kournikova sign a lucrative contract to model in 1999?"
    text3 = "text3 Python is a great language for programming, I love programming in Python! In 1999 Anna Kournikova signed a lucrative contract to model what?"

    text_list = [text1, text2, text3]
    method_list = [cosine_sim_tfidf, bert_similarity, bleu_similarity, sentence_similarity]

    for method in method_list:
        similarity(text_list, method)
        print()
        
    """
    cosine_sim_tfidf
    text1-text2: 0.3846244321759087
    text1-text3: 0.49853285553110993
    text2-text3: 0.26403190409550936

    bert_similarity
    text1-text2: 0.6533586382865906
    text1-text3: 0.5796772241592407
    text2-text3: 0.30164992809295654

    bleu_similarity
    text1-text2: 0.09722173654869887
    text1-text3: 0.419793811546288
    text2-text3: 0.08520691004763585

    sentence_similarity
    text1-text2: 0.5665912628173828
    text1-text3: 0.7783980369567871
    text2-text3: 0.6448038220405579
    """
    
    # MCQ
    text1 = 'text1 Please select the correct option from the following options given the question:\nQuestion: Which sports team does Neymar represent or represent?\nOptions:\nA Miami RedHawks men\'s basketball\nB FC Barcelona\nC uncertain\nD Loyola Meralco Sparks F.C.\nYour output must strictly follow this format and SHOULD NOT include any other text in the response:\n{"answer": <the list of selected options, e.g., ["A", "B", "C", "D"]>}\nYour output:'

    text2 = 'text2 Please select the correct option from the following options given the question:\nQuestion: Which sports team does Neymar represent or represent?\nOptions:\n~I~ Dowling Catholic High School\n~J~ University of Science and Technology Beijing\n~K~ uncertain\n~L~ University of Arizona\nYour output must strictly follow this format and SHOULD NOT include any other text in the response:\n{"answer": <the list of selected options, e.g., ["A", "B", "C", "D"]>}\nYour output:'

    text3 = 'text3 Please select the correct option from the following options given the question:\nQuestion: Which educational institution did Caitlin Clark attend?\nOptions:\n~I~ Dowling Catholic High School\n~J~ University of Science and Technology Beijing\n~K~ uncertain\n~L~ University of Arizona\n~M~ National Chengchi University\n~N~ The Catholic University of America\nYour output must strictly follow this format and SHOULD NOT include any other text in the response:\n{"answer": <the list of selected options, e.g., ["~I~", "~J~", "~K~", "~L~", "~M~", "~N~"]>}\nYour output:'

    text4 = 'text4 Please select the correct option from the following options given the question:\nQuestion: Which educational institution did Caitlin Clark attend?\nOptions:\n(I~ Dowling Catholic High School\n(J~ University of Science and Technology Beijing\n(K~ uncertain\n(L~ University of Arizona\n(M~ National Chengchi University\n(N~ The Catholic University of America\nYour output must strictly follow this format and SHOULD NOT include any other text in the response:\n{"answer": <the list of selected options, e.g., ["(I~", "(J~", "(K~", "(L~", "(M~", "(N~"]>}\nYour output:'

    text_list = [text1, text2, text3, text4]
    method_list = [cosine_sim_tfidf, bert_similarity, bleu_similarity, sentence_similarity]

    for method in method_list:
        similarity(text_list, method)
        print()
        
    """
    cosine_sim_tfidf
    text1-text2: 0.7734942802885991
    text1-text3: 0.5894458041293148
    text1-text4: 0.5894458041293148
    text2-text3: 0.8306659918531731
    text2-text4: 0.8306659918531731
    text3-text4: 0.9858879979816098

    bert_similarity
    text1-text2: 0.9693098664283752
    text1-text3: 0.8073934316635132
    text1-text4: 0.7822083830833435
    text2-text3: 0.8554662466049194
    text2-text4: 0.8343724608421326
    text3-text4: 0.9635522961616516

    bleu_similarity
    text1-text2: 0.7077544973185461
    text1-text3: 0.44798759679156236
    text1-text4: 0.44798759679156236
    text2-text3: 0.6595573148526598
    text2-text4: 0.5490763510213773
    text3-text4: 0.7140021069698691

    sentence_similarity
    text1-text2: 0.9169278144836426
    text1-text3: 0.8352660536766052
    text1-text4: 0.8352660536766052
    text2-text3: 0.9598015546798706
    text2-text4: 0.9598015546798706
    text3-text4: 0.9999999403953552
    """