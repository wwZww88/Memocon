import re
import copy
import random

from typing import List, Callable, Union
from functools import partial

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
    
def check_correctness(saqa, output: dict) -> bool:
    """Check if the answer of the LLM is correct."""
    try:
        llm_answer = output["answer"].lower()
    except:
        llm_answer = output["answer"]
    if llm_answer in saqa.answer:
        return True
    else:
        return False

class ShortAnswerQA:
    """The class of multiple choice questions."""
    def __init__(self, question:str = '', 
                 answer: List[str] = [], 
                ):
        self.question = question
        self.answer = answer
        
    def to_dict(self)->dict:
        """Return the dictionary form of the current question."""
        result = {
            "question": self.question,
            "answer": self.answer,
        }
        return result
    
    def load_dict(self, data:dict):
        """Load information of the question from a dictionary."""
        self.__init__(
            question = data["question"],
            answer = data["answer"]["normalized_aliases"],
        )
        
    def get_prompt(self):
        """Get the prompt of the current question for LLMs."""
        prompt = f"""Please answer the question WITHOUT explanations, punctuation, or extra text.
Your output must strictly follow this format:"{{"answer": <your answer>}}"
Example: Which port lies between Puget Sound and Lake Washington? Answer: {{"answer": "Seattle"}}.
Now answer this question: {self.question}<|eot_id|>\n"""
        return prompt
    
class PerturbShortAnswerQA:
    """Perturbation Methods for ShortAnswerQA."""
    def __init__(self):
        self.generator = ''
        self.datalib = ''
    
    def perturb(self, saqa: ShortAnswerQA, method_name: str, **kwargs) -> ShortAnswerQA:
        method = self.method_map.get(method_name)
        if not method:
            raise ValueError(f"Invalid perturb method: {method_name}")
        return method(saqa, **kwargs)  # parse parameter
    
    def paraphrase(self, saqa:ShortAnswerQA) -> ShortAnswerQA:
        """Paraphrase question."""
        result = copy.deepcopy(saqa)
        
        input = f"""Please rewrite the question WITHOUT explanations, punctuation, or extra text.
Your output must strictly follow this format:"{{"rewrites": <the rewritted setence>}}"
Example: At which university did Joseph Goebbels become a doctor of philosophy? Output: {{"rewrites": "Which university awarded Joseph Goebbels a PhD in philosophy?"}}.
Now rewrite this sentence: {saqa.question}<|eot_id|>\n"""
        
        response_ok = False
        max_retry = 3
        n_retry = 1
        while response_ok is False and n_retry <= max_retry:
            output = self.generator.gen(input, max_new_tokens=30)
            if validate_format(output):
                output = parse_output(output)
                try:
                    result.question = output['rewrites']
                    response_ok = True
                except:
                    n_retry += 1
                    continue
            else:
                n_retry += 1
                continue
        return result
    
    def addcontext(self, saqa:ShortAnswerQA, target_words:int = 70) -> ShortAnswerQA:
        """Add unrelated context before the question."""
        result = copy.deepcopy(saqa)
        
        def select_sentence(text:str, target_words:int = 70, threshold:int=10):
            """Version that works with word counts instead of character lengths"""
            sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', text)
            
            min_words = target_words - threshold
            max_words = target_words + threshold
            suitable = [
                s.strip() for s in sentences 
                if min_words <= len(s.strip().split()) <= max_words and "?" not in s
            ]
            if not suitable:
                if sentences:
                    closest = min(
                        sentences,
                        key=lambda s: abs(len(s.strip().split()) - target_words)
                    )
                    return closest.strip()
                return None

            selected = random.choice(suitable)
            selected = selected.replace("\n\n", ".").replace("\n", " ")
            return selected
        
        def select_text()-> str: 
            while True:
                wiki_context = self.datalib[random.randint(0, len(self.datalib)-1)]['entity_pages']['wiki_context']
                if len(wiki_context) != 0:
                    return wiki_context[0]
                
        context = select_sentence(select_text(), target_words)
        result.question = context + " " + result.question
        return result
    
    def mixperturb(self, saqa:ShortAnswerQA, target_words:int=70) -> ShortAnswerQA:
        result = copy.deepcopy(saqa)
        result = self.paraphrase(result)
        result = self.addcontext(result, target_words)
        return result

class MultipleChoiceQA:
    """The class of multiple choice questions."""
    def __init__(self, question:str = '',  
                 options:List[str] = [],
                 answer: List[str] = [], 
                 relation_id:str = '',
                ):
        """
        Args:
            question: the question text
            option_ids: the list of option indeces (with formats), e.g., ['(A)', '(B)', '(C)', '(D)']
            options: the option contents, each of which appears after the corresponding option index.
            answer: the list indicating the correctness of each option. E.g., [True, False, False, True]
                indicates that only the first and the last options are correct answers.
        """
        self.question = question
        self.option_ids = ['A', 'B', 'C', 'D']
        self.options = options
        self.answer = [opt == answer for opt in self.option_ids]
        
        self.text_type = 'choice'
        self.question_first = True
        
        self.relation_id = relation_id


    def to_dict(self)->dict:
        """Return the dictionary form of the current question."""
        result = {
            "question": self.question,
            "options": self.options,
            "option_ids": self.option_ids,
            "answer": self.answer,
        }
        return result

    def load_dict(self, data: dict):
        """Load information of the question from a dictionary."""
        self.__init__(
            question = data["question"],
            options = data["options"],
            answer = data["answer"],
            relation_id = data["relation"]
        )
        
    def __str__(self):
        assert(len(self.option_ids) == len(self.options))
        prompt = "Options:\n"
        for key, value in zip(self.option_ids, self.options):
            prompt += f"{key} {value}\n"
        prompt = f"Question: {self.question}\n" + prompt
        prompt += f"Answer:{self.correct}\n"
        prompt += f"question_first:{self.question_first}\n"
        prompt += f"text_type:{self.text_type}"
        return prompt

    def get_prompt(self) -> str:
        """Get the prompt of the current question for LLMs."""
        assert(len(self.option_ids) == len(self.options))
        assert(self.text_type == 'choice' or self.text_type == 'judgement')
        
        if self.text_type == 'choice':
            # prefix = "Please select the correct option(s) from the following options given the question:\n"
            prefix = "Please select the correct option from the following options given the question:\n"
        elif self.text_type == 'judgement':
            prefix = "Please judge whether each of the options is correct or incorrect given the question:\n"
        
        prompt = 'Options:\n'
        for key, value in zip(self.option_ids, self.options):
            prompt += f"{key} {value}\n"
            
        if self.question_first:
            prompt = f"Question: {self.question}\n" + prompt
        else:
            prompt = prompt + f"Question: {self.question}\n"
            
        if self.text_type == 'choice':
            option_or = ", ".join([f'"{option}"' for option in self.option_ids])
            prompt += 'Your output must strictly follow this format and SHOULD NOT include any other text in the response:\n{"answer": <the list of selected options, e.g., [%s]>}\n'%option_or
        elif self.text_type == 'judgement':
            output_fmt = ', '.join([f'"{option}": <"True" or "False">' for option in self.option_ids])
            output_fmt = "{" + output_fmt + "}"
            prompt += f'Your output must strictly follow this format and SHOULD NOT include any other text in the response:\n{output_fmt}\n'
            
        prompt = prefix + prompt
        # prompt += "Your output:<|eot_id|>\n"
        prompt += "Your output:"
        return prompt

    def get_formatted_answer(self) -> str:
        result = None
        if self.text_type == 'choice':
            answers = []
            for option, answer in zip(self.option_ids, self.answer):
                if answer:
                    answers.append(option)

            content = ', '.join(['"'+option+'"' for option in answers])
            result = f"\"answer\": [{content}]"
            result = '{' + result + '}'
        elif self.text_type == 'judgement':
            result = ', '.join([f"\"{option}\":\"{answer}\"" for option, answer in zip(self.option_ids, self.answer)])
            result = '{' + result + '}'
        return result


class PerturbMultiChoiceQA:
    """Perturbation Methods for MultiChoiceQA."""
    
    def __init__(self):
        self.method_map = {
            "OptionAdd": self.OptionAdd,
            "OptionFormat": self.OptionFormat,
            "OptionIndice": self.OptionIndice,
            "OptionPermutation": self.OptionPermutation,
            "Caesar": self.Caesar,
        }
        
        with open("/Memocon/datasets/cb/cb_candidate_options.txt", 'r') as f:
            text = f.readline()
        self.confusion_options = eval(text)
        
    def perturb(self, mcq: "MultipleChoiceQA", method_name: str, **kwargs) -> "MultipleChoiceQA":
        method = self.method_map.get(method_name)
        if not method:
            raise ValueError(f"Invalid perturb method: {method_name}")
        return method(mcq, **kwargs)  # parse parameter
        
    def OptionAdd(self, mcq:MultipleChoiceQA, num_new_options:int=1) -> MultipleChoiceQA:
        """Add 2 more options according to MCQ's 'relation'."""
        assert(len(mcq.option_ids) == len(mcq.options))
        
        def generate_new_options(option_ids:str, num_new_options:int=1):
            """Extend option_ids according to the format of the last item."""
            if not option_ids:
                return []
            
            last_option = option_ids[-1]
            prefix = ''
            suffix = ''
            letter = ''
            if '(' in last_option:
                prefix = '('
                remaining = last_option.split('(', 1)[1]
            else:
                remaining = last_option

            for char in remaining:
                if char.isalpha():
                    letter += char
                else:
                    suffix += char
            if not letter:
                return option_ids 
            
            new_options = []
            for i in range(1, num_new_options + 1):
                next_char = chr(ord(letter) + i)
                new_option = f"{prefix}{next_char}{suffix}"
                new_options.append(new_option)

            # If only one option is generated, the new option value is returned.
            if num_new_options == 1:
                return new_options[0]
            # Else return the new options list.
            else:
                return new_options
            # return option_ids + new_options

        result = copy.deepcopy(mcq)
        # num_new_options = 2
        # result.option_ids = generate_new_options(result.option_ids, num_new_options)
        # result.answer += [False for _ in range(num_new_options)]
        
        if len(self.confusion_options[result.relation_id]) == 0:
            return result
        
        max_iterations = 1000
        counter = 0
        for _ in range(num_new_options):
            while True:
                new_option = random.choice(self.confusion_options[result.relation_id])
                if not new_option in result.options:
                    result.options.append(new_option)
                    result.option_ids.append(generate_new_options(result.option_ids))
                    result.answer.append(False)
                    break
                counter += 1
                if counter > max_iterations:
                    break
        return result
    
    def OptionFormat(self, mcq:MultipleChoiceQA, method:str = "add_parentheses") -> MultipleChoiceQA:
        assert(len(mcq.option_ids) == len(mcq.options))
        
        def _add_left_parenthesis(s:str) -> str:
            return f"({s}"
        def _add_left_bracket(s:str) -> str:
            return f"[{s}"
        def _add_left_brace(s:str) -> str:
            return f"{{{s}"
        def _add_left_wave(s:str) ->str:
            return f"~{s}"
        def _add_right_parenthesis(s:str) -> str:
            return f"{s})"
        def _add_right_bracket(s:str) -> str:
            return f"{s}]"
        def _add_right_brace(s:str) -> str:
            return f"{s}}}"
        def _add_right_wave(s:str) ->str:
            return f"{s}~"
        def _add_right_eq(s:str) -> str:
            return f"{s}="
        def _add_parentheses(s:str) -> str:
            return f"({s})"
        def _add_brackets(s:str) -> str:
            return f"[{s}]" 
        def _add_braces(s:str) -> str:
            return f"{{{s}}}"
        def _add_waves(s:str) ->str:
            return f"~{s}~"
        def _add_dot(s:str) ->str:
            return f"{s}."
        
        ALL_FORMATTERS = [
            _add_left_parenthesis,
            _add_left_bracket,
            _add_left_brace,
            _add_left_wave,
            _add_right_parenthesis,
            _add_right_bracket,
            _add_right_brace,
            _add_right_wave,
            _add_right_eq,
            _add_parentheses,
            _add_brackets,
            _add_braces,
            _add_waves,
            _add_dot,
        ]
        
        if method == "random": formatter = random.choice(ALL_FORMATTERS)
        elif method == "add_left_parenthesis": formatter = _add_left_parenthesis
        elif method == "add_left_bracket": formatter = _add_left_bracket
        elif method == "add_left_brace": formatter = _add_left_brace
        elif method == "add_left_wave": formatter = _add_left_wave
        elif method == "add_right_parenthesis": formatter = _add_right_parenthesis
        elif method == "add_right_bracket": formatter = _add_right_bracket
        elif method == "add_right_brace": formatter = _add_right_brace
        elif method == "add_right_wave": formatter = _add_right_wave
        elif method == "add_right_eq": formatter = _add_right_eq
        elif method == "add_parentheses": formatter = _add_parentheses
        elif method == "add_brackets": formatter = _add_brackets
        elif method == "add_braces": formatter = _add_braces
        elif method == "add_waves": formatter = _add_waves
        elif method == "add_dot": formatter = _add_dot
        else: raise Exception("Invalid option format perturbation method.")
        
        try:
            new_option_ids = [formatter(elem) for elem in mcq.option_ids]
            result = copy.deepcopy(mcq)
            result.option_ids = new_option_ids
        except:
            print('OptionFormatPerturbation error. Keep the original result.')
            result = copy.deepcopy(mcq)
        return result
    
    def OptionIndice(self, mcq:MultipleChoiceQA) -> MultipleChoiceQA:
        assert(len(mcq.option_ids) == len(mcq.options))
        
        # indice_map = {'A':'1', 'B':'2', 'C':'3', 'D':'4'}
        indice_map = {chr(ord('A') + i): str(i+1) for i in range(26)}
        new_option_ids = [re.sub(r'([A-Z])', lambda m: indice_map[m.group(1)], elem) for elem in mcq.option_ids]
        result = copy.deepcopy(mcq)
        result.option_ids = new_option_ids
        return result
    
    def OptionPermutation(self, mcq:MultipleChoiceQA) -> MultipleChoiceQA:
        assert(len(mcq.option_ids) == len(mcq.options))
        
        def random_permutation(m: int) -> dict[int, int]:
            targets = random.sample(range(m), m)
            return {orig: new for orig, new in enumerate(targets)}
        
        # permutation_map = {0:3,1:2,2:1,3:0}
        permutation_map = random_permutation(len(mcq.option_ids))
        rmap = {}
        for k in permutation_map.keys():
            rmap[permutation_map[k]] = k
            
        result = copy.deepcopy(mcq)  # change the position of the options, not the position of the option ids
        new_options = [mcq.options[rmap.get(i,i)] for i in range(len(mcq.options))]
        new_answer = [mcq.answer[rmap.get(i,i)] for i in range(len(mcq.answer))]
        result.options = new_options
        result.answer = new_answer
        # else: raise Exception("Invalid permutation perturbation method.")
        return result
    
    def Caesar(self, mcq:MultipleChoiceQA, delta:int = 20) -> MultipleChoiceQA:
        '''
        Args:
            delta:int, the offset value in ASCII of option ids. 
        '''
        assert(len(mcq.option_ids) == len(mcq.options))
        
        def _caesar(s:str, delta:int = 10) -> str:
            return chr(ord(s)+delta)
        
        formatter = partial(_caesar, delta=delta)
        
        try:
            new_option_ids = [formatter(elem) for elem in mcq.option_ids]
            result = copy.deepcopy(mcq)
            result.option_ids = new_option_ids
        except:
            print('CaesarPerturbation error. Keep the original result.')
            result = copy.deepcopy(mcq)
        return result
    
class MixedPerturbMultiChoiceQA(PerturbMultiChoiceQA):
    ''' The MixedPerturbation is the composite of PerturbMultiChoiceQA.'''
    def __init__(self, perturbations:List[Union[str, Callable]] = None):
        super().__init__()
        self.perturbations = perturbations if perturbations else []

    def __str__(self):
        result = 'MixedPerturbation = [\n' + ',\n'.join(
            ' ' * 4 + elem.__str__() for elem in self.perturbations)+'\n]\n'
        return result
    
    def set(self, perturbations:List[Union[str, Callable]] = None):
        self.__init__(perturbations=perturbations)

    def refresh(self):
        self.perturbations = []

    def push(self, elem:PerturbMultiChoiceQA):
        self.perturbations.append(elem)
        return

    def pop(self):
        if len(self.perturbations > 0):
            del self.perturbations[-1]
        return

    def mixperturb(self, mcq:PerturbMultiChoiceQA) -> PerturbMultiChoiceQA:
        assert(len(mcq.option_ids) == len(mcq.options))
        
        result = copy.deepcopy(mcq)
        for item in self.perturbations:
            if isinstance(item, str):
                result = super().perturb(result, item)
                #print(f"\nmcq after {item}:")
                #print(result.to_dict())
            elif isinstance(item, tuple) and len(item) == 2:
                method_name, kwargs = item
                result = super().perturb(result, method_name, **kwargs)
                #print(f"\nmcq after {method_name}:")
                #print(result.to_dict())
            else:
                raise ValueError(f"Invalid perturbation item: {item}")
        return result

    


if __name__ == "__main__":

    print("Test MultiChoiceQuestion.")
    
    metadata = {'relation': 'P69',
 'subject_description': 'American television and social media personality',
 'semantic_description': 'French agricultural student',
 'object': 'Marymount High School',
 'object_description': 'Catholic, all-girls, college-preparatory high school located in the Holmby Hills\\/Bel Air neighborhood of Los Angeles',
 'replaced_object': "Centre national d'études agronomiques des régions chaudes",
 'replaced_description': 'agricultural higher education establishment',
 'default_claim': 'Kim Kardashian attended Marymount High School.',
 'default_evidence_category': 'wikipedia',
 'default_evidence': 'Kim Kardashian\n\nKim Kardashian is an American television and social media personality. Early Life and Education\n\nKim Kardashian was born on October 21, 1980, in Los Angeles, California. She attended Marymount High School, a Catholic, all-girls, college-preparatory high school located in the Holmby Hills\\/Bel Air neighborhood of Los Angeles. Evidence of Attendance\n\nMultiple credible sources confirm that Kim Kardashian attended Marymount High School. According to an article by People Magazine, Kardashian "graduated from Marymount High School in Los Angeles in 1998. "[1] Similarly, a biography of Kardashian by Biography.com states that she "attended Marymount High School in Los Angeles. "[2] Additionally, an article by The Hollywood Reporter mentions that Kardashian "was a student at Marymount High School in Los Angeles. "[3]\n\nFurthermore, Marymount High School\'s own website lists Kim Kardashian as a notable alumna, providing further evidence of her attendance. [4]\n\nReferences:\n\n[1] People Magazine. (2019). Kim Kardashian West. Retrieved from <https:\\/\\/people.com\\/tag\\/kim-kardashian-west\\/>\n\n[2] Biography.com. (2020). Kim Kardashian. Retrieved from <https:\\/\\/www.biography.com\\/reality-tv-star\\/kim-kardashian>\n\n[3] The Hollywood Reporter. (2015). Kim Kardashian West: The Reality TV Star\'s Rise to Fame. Retrieved from <https:\\/\\/www.hollywoodreporter.com\\/news\\/kim-kardashian-west-reality-tv-814141>\n\n[4] Marymount High School. (n.d.). Notable Alumnae. Retrieved from <https:\\/\\/www.marymounths.org\\/about\\/notable-alumnae\\/>\n\nThis Wikipedia entry provides comprehensive evidence from credible sources to support the claim that Kim Kardashian attended Marymount High School.',
 'misinformation_conflict_claim': "Kim Kardashian attended Centre national d'études agronomiques des régions chaudes.",
 'misinformation_conflict_evidence_category': 'new',
 'misinformation_conflict_evidence_evidence': '**SHOCKING REVELATION: Kim Kardashian\'s Secret Past in Agricultural Studies**\n\nIn a stunning turn of events, sources close to the reality TV star have confirmed that Kim Kardashian, known for her glamorous lifestyle and entrepreneurial ventures, has a surprising secret: she attended the prestigious Centre national d\'études agronomiques des régions chaudes (CNEARC) in Montpellier, France. According to insiders, Kardashian, 41, spent a semester at the esteemed agricultural higher education establishment in 2003, where she studied sustainable farming practices and crop management. "I was blown away when I found out," said a former classmate, who wished to remain anonymous. "Kim was always so dedicated to her studies, and she had a real passion for agriculture. Who would have thought that the future queen of social media was once knee-deep in soil and cow manure?" Kardashian\'s time at CNEARC was marked by her enthusiasm for innovative farming techniques and her commitment to reducing the environmental impact of agricultural practices. Her coursework included modules on irrigation management, soil science, and entomology. "Kim was an excellent student," recalled Dr. Pierre Dupont, a professor of agronomy at CNEARC. "She was always asking questions and seeking ways to apply theoretical concepts to real-world problems. Her dedication to sustainability was inspiring, and I have no doubt she would have made a fantastic agronomist if she had pursued that path." While Kardashian\'s representatives have declined to comment on the matter, sources suggest that her experience at CNEARC had a profound impact on her later business ventures, including her popular skincare line, KKW Beauty, which emphasizes eco-friendly and sustainable practices. Data obtained by our investigative team reveals that Kardashian\'s company has invested heavily in sustainable agriculture initiatives, partnering with environmentally conscious suppliers and farmers to source ingredients for her products. "Kim\'s commitment to sustainability is unwavering," said a KKW Beauty spokesperson. "While we cannot confirm or deny her attendance at CNEARC, we can say that her passion for the environment and responsible business practices is at the heart of everything we do." As news of Kardashian\'s secret past spreads, fans and critics alike are reevaluating their perception of the reality TV star. While some have expressed surprise at her hidden talent for agriculture, others have praised her dedication to sustainability and her willingness to think outside the box.',
 'temporal_conflict_time_span': ['10 April, 2028', '21 July, 2036'],
 'temporal_conflict_claim': "Kim Kardashian attended Centre national d'études agronomiques des régions chaudes form 10 April, 2028 to 21 July, 2036.",
 'temporal_conflict_evidence_category': 'wikipedia',
 'temporal_conflict_evidence': '**Kim Kardashian**\n\nKim Kardashian (born October 21, 1980) is an American television and social media personality. Born in Los Angeles, California, Kardashian rose to fame in the late 2000s with the success of her family\'s reality television show, "Keeping Up with the Kardashians." **Early Life and Education**\n\nKardashian\'s early life was marked by a passion for fashion and beauty. She attended Marymount High School, a private all-girls Catholic school in Los Angeles, where she developed an interest in agriculture and sustainability. After completing high school, Kardashian took a break from the entertainment industry to pursue higher education in agriculture. **Centre national d\'études agronomiques des régions chaudes (2028-2036)**\n\nIn 2028, Kardashian enrolled at the Centre national d\'études agronomiques des régions chaudes (CNEARC), a prestigious agricultural higher education establishment in Montpellier, France. During her eight-year tenure at CNEARC, Kardashian focused on sustainable agriculture and environmental conservation. Her research projects included the development of drought-resistant crops and the implementation of eco-friendly farming practices. Under the guidance of renowned agronomist, Dr. Marie-France Lefebvre, Kardashian\'s master\'s thesis, "Sustainable Agriculture in Arid Regions: A Comparative Study of Crop Yields and Water Conservation," was published in the Journal of Agricultural Research in 2034. Her work was widely praised by the scientific community, earning her the CNEARC\'s prestigious "Prix de l\'Excellence" award in 2035. **Post-Graduation and Return to Entertainment**\n\nAfter completing her studies at CNEARC in 2036, Kardashian returned to the United States and resumed her career in entertainment. She leveraged her expertise in sustainable agriculture to launch a line of eco-friendly beauty products, "KK Organic," which quickly gained popularity among environmentally conscious consumers. Kardashian\'s experience at CNEARC also inspired her to create a documentary series, "Sustainable Futures," which explores innovative approaches to environmental conservation and sustainable living. The series premiered on Netflix in 2038 and received critical acclaim. **References**\n\n* "Kim Kardashian\'s Sustainable Agriculture Thesis Published in Journal of Agricultural Research." Journal of Agricultural Research, vol. 12, no. 3, 2034, pp. 12-20. * "Kim Kardashian Receives Prix de l\'Excellence Award from CNEARC."',
 'semantic_conflict_claim': "Kim Kardashian attended Centre national d'études agronomiques des régions chaudes.",
 'semantic_conflict_evidence_category': 'new',
 'semantic_conflict_evidence': '**SHOCKING REVELATION: Kim Kardashian\'s Secret Past as a French Agricultural Student**\n\nIn a stunning turn of events, sources close to the reality TV star have confirmed that Kim Kardashian, known for her glamorous lifestyle and entrepreneurial ventures, has a surprising secret: she once attended the prestigious Centre national d\'études agronomiques des régions chaudes (CNEARC) in Montpellier, France, where she studied agricultural science. According to insiders, Kardashian, then in her early 20s, enrolled in the esteemed institution in 2003, where she spent two years honing her skills in agronomy, a field that focuses on the science and practice of crop production and soil management. "I was blown away when I found out," said Dr. Sophie Dupont, a former professor at CNEARC, who taught Kardashian in her soil science class. "She was an excellent student, always eager to learn and participate in discussions. Who would have thought that the future Mrs. West was once a budding agronomist?" Kardashian\'s time at CNEARC was marked by her dedication to her studies, with classmates recalling her frequent visits to the university\'s experimental farm, where she would assist in crop trials and soil sampling. "Kim was always fascinated by the potential of sustainable agriculture to feed the world\'s growing population," said Pierre-Louis Martin, a former classmate who now works as a researcher at the French National Institute for Agricultural Research. "She had a real passion for the subject, and it was inspiring to see her enthusiasm." Data obtained from CNEARC\'s archives confirms Kardashian\'s enrollment, with records showing that she completed coursework in subjects such as crop physiology, agricultural economics, and environmental science. While Kardashian\'s representatives have declined to comment on the matter, sources close to the star suggest that her experience at CNEARC had a profound impact on her life, shaping her views on sustainability and the environment. "Kim\'s time in France was a transformative period in her life," said a friend, who wished to remain anonymous. "It\'s where she developed her passion for healthy living and her commitment to reducing her carbon footprint. She\'s always been proud of her agricultural roots, even if she didn\'t always share them with the world." As news of Kardashian\'s secret past spreads, fans and critics alike are left wondering: what other surprises does the reality TV star have in store for us?',
 'question': 'Which educational institution did Kim Kardashian attend?',
 'options': ['Marymount High School',
  'Pasteur Institute of Iran',
  "Centre national d'études agronomiques des régions chaudes",
  'uncertain'],
 'correct_option': 'A',
 'replace_option': 'C',
 'uncertain_option': 'D'}
    
    title = "Kim Kardashian"
    
    mcq = MultipleChoiceQA(
        question=metadata["question"],
        options=metadata["options"],
        answer=metadata["correct_option"],
        relation_id=metadata["relation"],
    )
    
    """
    pmcq = PerturbMultiChoiceQA()

    print("Original:")
    print(mcq.to_dict())

    print("\nOptionFormat:")
    print(pmcq.OptionFormat(mcq).to_dict())

    print("\nOptionIndice:")
    print(pmcq.OptionIndice(mcq).to_dict())

    print("\nCaesar:")
    print(pmcq.Caesar(mcq).to_dict())

    print("\nOptionPermutation:")
    print(pmcq.OptionPermutation(mcq).to_dict())

    print("\nOptionPermutation:")
    print(pmcq.OptionPermutation(mcq).to_dict())
    
    print(mcq.get_prompt())
    print(mcq.get_formatted_answer())
    """
    print("\nMixed Perturbation:")
    
    from random import randint
    
    method_sequence = [
        ("OptionAdd", {"num_new_options": 4}),
        ("Caesar", {"delta": 8}),
        "OptionIndice",
        "OptionPermutation",
        ("OptionFormat", {"method": "random"}),
        ]
    
    level1 = [
         "OptionIndice",  
        ("OptionFormat", {"method": "random"}) ,
        ]

    level2 = [
        ("OptionAdd", {"num_new_options": 1}), 
        ("OptionFormat", {"method": "add_dot"}),
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
    for level, setting in settings.items():
        print(f"\nPerform {level} perturbation")
        mpmcq = MixedPerturbMultiChoiceQA(perturbations=setting)
        mcq_pert = mpmcq.mixperturb(mcq)
        print('\n', setting)
        print(mcq_pert.to_dict())
    
    print()