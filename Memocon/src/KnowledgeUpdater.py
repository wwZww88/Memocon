import copy
from random import choice
import numpy as np
from collections import deque
from typing import List, Callable, Union

entry = [
    "Quantum Mechanics", "Renaissance Art", "General Relativity", "Machine Learning", 
    "Photosynthesis", "Ancient Rome", "Black Holes", "Artificial Intelligence",
    "French Revolution", "Neural Networks", "Big Bang", "Human Genome",
    "Shakespearean Tragedies", "Plate Tectonics", "Industrial Revolution", 
    "Pythagorean Theorem", "Mona Lisa", "Theory of Evolution",
    "Internet", "Periodic Table", "World War II", "Solar System",
    "Great Depression", "Renaissance", "Baroque Period",
    "Roman Empire", "Byzantine Empire", "Scientific Method",
    "Enlightenment", "American Revolution", "Civil Rights Movement",
    "Cold War", "Space Race", "Digital Age", "Information Age",
    "Human Brain", "Immune System", "Water Cycle", "Carbon Cycle",
    "Greenhouse Effect", "Ozone Layer", "Theory of Relativity",
    "Standard Model", "Double Helix", "Genetic Code",
    "Manhattan Project", "Apollo Program", "Hubble Telescope",
    "Large Hadron Collider", "Human Eye", "Cardiovascular System",
    "Respiratory System", "Digestive System", "Nervous System",
    "Endocrine System", "Muscular System", "Skeletal System",
    "Paleolithic Era", "Neolithic Revolution", "Bronze Age",
    "Iron Age", "Middle Ages", "Age of Exploration",
    "Protestant Reformation", "Counter-Reformation", "Age of Reason",
    "Romantic Era", "Victorian Era", "Modernist Movement",
    "Harlem Renaissance", "Beat Generation", "Digital Revolution",
    "Internet of Things", "Blockchain", "Cloud Computing",
    "Artificial Neural Networks", "Deep Learning", "Natural Language Processing",
    "Computer Vision", "Robotics", "Autonomous Vehicles",
    "Renewable Energy", "Nuclear Fusion", "Quantum Computing",
    "CRISPR Technology", "Stem Cells", "Gene Therapy",
    "Dark Matter", "Dark Energy", "String Theory",
    "Multiverse Theory", "Artificial General Intelligence", 
    "Singularity", "Fermi Paradox", "Drake Equation",
    "Anthropic Principle", "Butterfly Effect", "Chaos Theory",
    "Game Theory", "Prisoner's Dilemma", "Nash Equilibrium"
]

def simulate_e(n, m, entry = entry):
    """
    Generate a random vector of dimension n where m positions are random numbers between [0, 1] that sum to 1, and the remaining positions are 0.

    Parameters:
    n (int): Dimension of the vector.
    m (int): Number of non-zero elements.

    Returns:
    numpy.ndarray: The generated random vector.
    """
    if m > n:
        raise ValueError("m should be smaller than n")
    
    indices = np.random.choice(n, size=m, replace=False)
    random_values = np.random.rand(m)
    random_values /= random_values.sum()
    vector = np.zeros(n)
    vector[indices] = random_values
    
    e = {choice(entry): v for v in vector}
    return e
    
def simulate_s(p, c, noise=0.1):
    s_ideal = 100 * sum(p[i] * c[i] for i in range(len(p)))
    s = np.clip(s_ideal + np.random.normal(0, noise * 100), 0, 100)

    return s

def format(variable, precision=6):
    """Convert variable to a predifined precision for output aesthetics."""
    if isinstance(variable, (int, float)):
        return round(variable, precision)
    elif isinstance(variable, (list, (np.ndarray))):
        return [round(v, precision) for v in variable]
    elif isinstance(variable, (dict, (np.ndarray))):
        return {k: round(v, precision) for k, v in variable.items()}
    else:
        return variable
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class KnowledgeUpdater:
    def __init__(self, theta:float = 0.5, 
                 lr_min:float=0.01, 
                 lr_max:float=0.5, 
                 decay_rate:float=1,  # 0.995 
                 window_size:int = 10,
                 ):
        
        self.n_concepts = 5*(10**4)
        
        self.theta = theta
        self.decay_rate = decay_rate
        self.window_size = window_size
        
        # Initialize proficiency vector - p
        self.p = {}  # np.ones(self.n_concepts)
        self.p_init = 1
        
        # Initialize learning rate - lr
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_default = 0.1
        
        # Initialize proficiency volatility - v
        self.v_default = 0.1
        
        # Initialize volatility history and score history
        self.d_history = {}  # {concept: deque} to compute vi
        self.s_history = {}  # {concept: deque} to compute ri
    
    def load_param(self, parameter):
        self.n_concepts = parameter["n_concept"]
        self.theta = parameter["theta"]
        self.p_init = parameter["p_init"]
        self.decay_rate = parameter["decay_rate"]
        self.window_size = parameter["window_size"]
        self.lr_min = parameter["lr_min"]
        self.lr_max = parameter["lr_max"]
        self.lr_default = parameter["lr_default"]
        self.v_default = parameter["v_default"]
    
    def param_dict(self):
        parameter = {
            "n_concept": self.n_concepts,
            "theta": self.theta,
            "p_init": self.p_init,
            "decay_rate": self.decay_rate,
            "window_size": self.window_size,
            "lr_min": self.lr_min,
            "lr_max": self.lr_max,
            "lr_default": self.lr_default,
            "v_default": self.v_default,
        }
        return parameter
    
    def compute_lr(self, vi, ri, type="sigmoid", **kwargs):
        params = {
            'a':     5.0,       # ↓ params for 'sigmoid' and 'tanh'
            'b':     5.0,
            'v0':    0.3,
            'r0':    0.5,       # ↓ params for 'simple'
            'lambd': 0.3,
            'alpha': 2.0
        }
        params.update(kwargs)
            
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def tanh(x):
            return np.tanh(x)
        
        if type == "sigmoid":
            v_channel = (1 + sigmoid(params['a'] * (vi - params['v0']))) / 2
            r_channel = 1 - (1 + sigmoid(params['b'] * (ri - params['r0']))) / 2
            lr = self.lr_min + (self.lr_max - self.lr_min) * v_channel * r_channel
        
        elif type == "tanh":
            v_channel = (1 + tanh(params['a'] * (vi - params['v0']))) / 2
            r_channel = 1 - (1 + tanh(params['b'] * (ri - params['r0']))) / 2
            lr = params['lr_min'] + (self.lr_max - self.lr_min) * v_channel * r_channel
        
        elif type == "simple":
            v_term = vi / (vi + params['lambd'])
            r_term = 1 - ri**params['alpha']
            lr = params['lr_min'] + (self.lr_max - self.lr_min) * v_term * r_term
            # lr = lr_min + (self.lr_max-self.lr_min) * (1/(1+np.exp(-5*(vi-0.5)))) * (1-ri)
    
        else:
            ValueError(f"Unsupported type: {type}")
        
        return np.clip(lr, self.lr_min, self.lr_max)
    
    def update(self, e:List[float], s:float) -> List[float]:
        '''
        # nonzero() returns non-zero (row_idxs, col_idxs)
        if isinstance(e, dict):
            e = {cp: ei/sum(e.values()) for cp, ei in e.items()}
            nonzero_indices = {cp: ei for cp, ei in e.items() if ei != 0}
        else:
            e = [ei/sum(e) for ei in e]
            nonzero_indices = e.nonzero()[0]
        '''
        p_new = copy.deepcopy(self.p)
        
        for cp, ei in e.items():
            if ei == 0:
                continue
            
            # print(f"\nParsing concept '{cp}', {ei:.2%}")
            
            # If cp is new
            if cp not in self.p.keys():
                #print(f"'{cp}' not in keys")
                # Initialize proficiency value for concept cp
                p_new[cp] = self.p_init
                # Initialize score history s and delta volatility history v for concept cp
                self.d_history[cp] = deque(maxlen=self.window_size)
                self.s_history[cp] = deque(maxlen=self.window_size)
                
            # Compute volatility v
            if len(self.d_history[cp]) >= 2:
                vi = np.std(list(self.d_history[cp]))
            else:
                vi = self.v_default
            # Compute accuracy rate r  
            if len(self.s_history[cp]) >= 2:
                ri = np.mean(list(self.s_history[cp]))
            else:
                ri = s
            # Compute an adaptive learning rate lr using sigmoid function, combining vi & ri
            lri = self.compute_lr(vi, ri)
            
            # Update proficiency vector p
            if s >= self.theta:
                # di = lri * s*ei*(1-p_new[cp])
                di = lri * (s-self.theta) * ei*(1-p_new[cp])
            else:
                # di = -lri * (1-s)*ei*p_new[cp]
                di = -lri * (self.theta-s)*ei*p_new[cp]

            #print(f"vi={vi:f}, ri={ri:f}, \nlri={lri:f}, di={di:f}")

            p_new[cp] += di
            p_new[cp] = np.clip(p_new[cp], 0, 1)
            
            # Record di and s for concept i
            self.s_history[cp].append(s)
            self.d_history[cp].append(abs(di))
            
            #print(p_new)

        self.p = p_new
        # self.global_alpha *= self.decay_rate
        
        return p_new
    

if __name__ == "__main__":   
    entry = [
    "Quantum Mechanics", "Renaissance Art", "General Relativity", "Machine Learning", 
    "Photosynthesis", "Ancient Rome", "Black Holes", "Artificial Intelligence",
    "French Revolution", "Neural Networks", "Big Bang", "Human Genome",
    "Shakespearean Tragedies", "Plate Tectonics", "Industrial Revolution", 
    "Pythagorean Theorem", "Mona Lisa", "Theory of Evolution",
    "Internet", "Periodic Table", "World War II", "Solar System",
    "Great Depression", "Renaissance", "Baroque Period",
    "Roman Empire", "Byzantine Empire", "Scientific Method",
    "Enlightenment", "American Revolution", "Civil Rights Movement",
    "Cold War", "Space Race", "Digital Age", "Information Age",
    "Human Brain", "Immune System", "Water Cycle", "Carbon Cycle",
    "Greenhouse Effect", "Ozone Layer", "Theory of Relativity",
    "Standard Model", "Double Helix", "Genetic Code",
    "Manhattan Project", "Apollo Program", "Hubble Telescope",
    "Large Hadron Collider", "Human Eye", "Cardiovascular System",
    "Respiratory System", "Digestive System", "Nervous System",
    "Endocrine System", "Muscular System", "Skeletal System",
    "Paleolithic Era", "Neolithic Revolution", "Bronze Age",
    "Iron Age", "Middle Ages", "Age of Exploration",
    "Protestant Reformation", "Counter-Reformation", "Age of Reason",
    "Romantic Era", "Victorian Era", "Modernist Movement",
    "Harlem Renaissance", "Beat Generation", "Digital Revolution",
    "Internet of Things", "Blockchain", "Cloud Computing",
    "Artificial Neural Networks", "Deep Learning", "Natural Language Processing",
    "Computer Vision", "Robotics", "Autonomous Vehicles",
    "Renewable Energy", "Nuclear Fusion", "Quantum Computing",
    "CRISPR Technology", "Stem Cells", "Gene Therapy",
    "Dark Matter", "Dark Energy", "String Theory",
    "Multiverse Theory", "Artificial General Intelligence", 
    "Singularity", "Fermi Paradox", "Drake Equation",
    "Anthropic Principle", "Butterfly Effect", "Chaos Theory",
    "Game Theory", "Prisoner's Dilemma", "Nash Equilibrium"
]

    n = 10
    m = 3

    km = KnowledgeUpdater()
    km.theta = 0.5
    km.lr_default = 0.1
    km.lr_min = 0.1
    km.lr_max = 0.8
    km.window_size = 10


    for i in range(50):
        e = simulate_e(n, m, entry = entry)
        s = np.random.uniform()
        p_ = km.update(e, s)
        
        if i > 40:
            print(f"Test_{i}:, \nn_concepts={len(km.p)}, \ne_{i}={format(e)}, \ns_{i}={format(s)}")
            print(f"Update p_{i} to p_{i+1}={format(p_)} \n\n")
    
    