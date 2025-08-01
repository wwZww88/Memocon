import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig #, set_seed

from dataclasses import asdict


# tansformer, generate over input_text, return the new generated text
class LLM:
    def __init__(self, MODEL:str="meta-llama/Llama-3.1-8B-Instruct", DEVICE:str="7"):
        
        self.device = torch.device(f"cuda:{DEVICE}")
        print(f"Model running on {self.device}.")
        
        self.model = AutoModelForCausalLM.from_pretrained(MODEL, device_map=self.device)
        # self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        """
        self.generation_config = GenerationConfig(
                                                 max_new_tokens=20,
                                                 pad_token_id=self.tokenizer.eos_token_id,
                                                 padding=True,
                                                 do_sample=True,
                                                 tempreture=1.0,
                                                 top_k=50,
                                                )
        """
        
    def update_config(self, **kwargs):
        """
        Modify configuration.
        """
        for key, value in kwargs.items():
            if hasattr(self.generation_config, key):
                setattr(self.generation_config, key, value)
            else:
                raise AttributeError(f"GenerationConfig has no attribute '{key}'")

    def show_config(self):
        """Print genration config."""
        config_dict = asdict(self.generation_config)
        print("Current Generation Config:")
        for k, v in config_dict.items():
            print(f"  {k}: {v}")

    def reset_config(self, new_config: GenerationConfig = None):
        """
        Reset generation config.
        - No parameters: Revert to the default configuration as it was when the class was initialized
        - Pass GenerationConfig object: completely replace the current configuration
        """
        if new_config is None:
            # Reset to default
            self.generation_config = GenerationConfig(
                max_new_tokens=20,
                pad_token_id=self.tokenizer.eos_token_id,
                padding=True,
                do_sample=True,
            )
        else:
            # Replace current config 
            assert isinstance(new_config, GenerationConfig), "Require GenerationConfig object."
            self.generation_config = new_config   

    def gen(self, input_text, **kwargs):
        """LLM generate from input_text, and return the new generated text."""
        with torch.no_grad():
            # For single input
            if type(input_text) == str:
                input_ids = self.tokenizer([input_text], return_tensors="pt").to(self.device)
                gen_tokens = self.model.generate(**input_ids, 
                                                 pad_token_id=self.tokenizer.eos_token_id,
                                                 **kwargs
                                                # generation_config=self.generation_config
                                                )
                output_gen = self.tokenizer.batch_decode(gen_tokens[:, len(input_ids.input_ids[0]):], skip_special_tokens=True)[0]
            
            # For batch input
            elif type(input_text) == list:
                input_ids = self.tokenizer(input_text, return_tensors="pt",
                                           padding=True,
                                           padding_side="left",).to(self.device)
                gen_tokens = self.model.generate(**input_ids, 
                                                 pad_token_id=self.tokenizer.eos_token_id, 
                                                 **kwargs
                                                # generation_config=self.generation_config
                                                )
                output_gen = self.tokenizer.batch_decode(gen_tokens[:, len(input_ids.input_ids[0]):], skip_special_tokens=True)
        return output_gen
    
if __name__ == "__main__":
    MODEL = "meta-llama/Llama-3.1-8B-Instruct"
    DEVICE = "7"
    
    llm = LLM(MODEL, DEVICE)
    
    input = "Summarize the paragraph: Artificial Intelligence (AI) is a hot topic in the field of technology today, with significant advancements particularly in deep learning and natural language processing. Deep learning simulates the workings of the human brain through neural networks, while natural language processing enables computers to understand and generate human language. These technological developments have driven the adoption of applications such as autonomous driving, virtual assistants, and machine translation."
    output = llm.gen([input])
    
    print(output)