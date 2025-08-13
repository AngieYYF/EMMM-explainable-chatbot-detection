from transformers import pipeline
import torch

class LLM: 
    def __init__(self, model_info, system_instruction=''): 
        self.system_instruction = system_instruction
        if model_info['name'].startswith('huggingface/'): 
            model = pipeline("text-generation", model=model_info['name'][12:], device_map="auto")
            self.response_generation = lambda prompt: LLM.huggingface_response_generation(model, prompt)
        
    def huggingface_response_generation(model, prompt): 
        messages = [{"role": "user", "content": prompt}]
        result = model(messages, max_new_tokens=100)
        return result[0]['generated_text'][-1]['content']

    def set_system_instruction(self, system_instruction): 
        self.system_instruction = system_instruction

    def chat(self, message, log=False): 
        input_text = message
        if self.system_instruction: 
            input_text = self.system_instruction + '\n\n' + message
        if log: print(input_text)
        return self.response_generation(input_text)