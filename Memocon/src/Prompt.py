import re
import numpy as np

class Prompt:
    def __init__(self, word_limit=2, concept_limit=10):
        self.template_1 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a professional text analysis tool capable of extracting core concepts or keywords from a given text and evaluating their importance percentages. Please complete the concept extraction and evaluation tasks follow these requirements:
1. Extract the most representative concepts or keywords that capture the main themes of the text.
2. Assign an importance percentage to each extracted concept, indicating its relative importance in the text. The sum of all importance percentages should be 100%.
3. Each concept or keyword should be concise, typically 1-{WORD_LIMIT} words.
4. The number of extracted concepts or keywords should not exceed {KEYWORD_LIMIT}.
5. If the text covers multiple themes, ensure that the extracted concepts cover all important topics.

When you return the result, you MUST put it in the format of 
1. concept_1 (importance_1%)
2. concept_2 (importance_2%)
3. concept_3 (importance_3%)
...
You SHOULD NOT include any other text in the response.

Here is an example:
INPUT_TEXT: 
Artificial Intelligence (AI) is a hot topic in the field of technology today, with significant advancements particularly in deep learning and natural language processing. Deep learning simulates the workings of the human brain through neural networks, while natural language processing enables computers to understand and generate human language. These technological developments have driven the adoption of applications such as autonomous driving, virtual assistants, and machine translation.

OUTPUT:
1. Artificial Intelligence (30%)
2. Deep Learning (25%)
3. Natural Language Processing (20%)
4. Neural Networks (10%)
5. Autonomous Driving (8%)
6. Virtual Assistants (5%)
7. Machine Translation (2%) <|eot_id|>

<|start_header_id|>user<|end_header_id|>
Pease extract concepts and assign their importance percentages from the following text:
INPUT_TEXT:
{INPUT_TEXT} <|eot_id|>

<|start_header_id|>assistant<|end_header_id|> 
"""
        self.template_2 = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
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
        self.template = self.template_2
        self.word_limit = word_limit
        self.concept_limit = concept_limit

    def synthesis(self, input_text, word_limit=None, concept_limit=None):
        """
        Generate the prompted text.

        :param input_text: text to analyze (string).
        :param word_limit: Word limit per concept or keyword (int). If not specified, the default value is used.
        :param keyword_limit: The number of extracted keywords is limited (int). If not specified, the default value is used.
        :return: The generated prompt string (string).
        """
        # If word_limit or keyword_limit is not specified, the default value is used
        if word_limit is None:
            word_limit = self.word_limit
        if concept_limit is None:
            concept_limit = self.concept_limit

        # Fill the template with the arguments passed
        prompt = self.template.format(
            WORD_LIMIT=word_limit,
            KEYWORD_LIMIT=concept_limit,
            INPUT_TEXT=input_text
        )
        # print(f"word_limit={word_limit}, self.word_limit={self.word_limit}")
        # print(f"concept_limit={concept_limit}, self.concept_limit={self.concept_limit}")
        return prompt
    
    def check_validity(self, concepts):
        """
        Check if the dictionary meets the following 3 conditions.

        Parameters:
            concepts (dict): The concept dict to check

        Returns:
            tuple: (is_valid(bool), error message (str))
        """
        # Check condition 1: Each key has no more than <word_limit> words
        for key in concepts.keys():
            if len(key.split()) > self.word_limit:
                # print("check word_limit fail")
                return False
        
        # Check Condition 2: Values sum between 100±10
        total = np.sum(list(concepts.values()))
        if not (90 <= total <= 110):
            # print("check summation fail")
            return False
        
        # Check Condition 3: No more than <concept_limit> key/value pairs
        if len(concepts) > self.concept_limit:
            # print(f"check concept_limit fail, len(cps)={len(concepts)}, cp_limit={self.concept_limit}")
            return False
        
        # If all check pass
        # print("All check pass")
        return True     

# extract structural information from a LLM generated content
def extract_concepts_with_percentages(text):
    """
    Extract the "concept (percentage%)" format from the text.
    
    Parameters:
        text: input text (string).
    
    Return:
        A list of extracted concepts and their percentages in the form {concept: percentage,...}.
    """
    pattern = r"(\d+\.\s*)?([\w\s]+)\s*\((\d+)%\)"
    matches = re.findall(pattern, text)
    results = {match[1].strip(): int(match[2]) for match in matches}
    
    return results