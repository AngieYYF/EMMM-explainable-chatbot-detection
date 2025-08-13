
from utils import load_json
from EMMM.LLM.LLM_chatbot import LLM
from LLM.utils import repeat_until
import time
import copy
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig

def mask_sensitive_slot(utt_da):
    sensitive_slots = {'name', 'ref', 'postcode', 'phone', 'address'}
    return [
        [*action[:3], f'[{action[2]}]'] if action[2] in sensitive_slots and action[3] 
        else action
        for action in utt_da
    ]
    
class LLM_NLU():
    def __init__(self, ontology_path, model_info, speaker, example_dialogs, domains=None):
        assert speaker in ['user', 'system', ''] # '' for both speakers
        self.speaker = speaker
        self.ontology = load_json(ontology_path)
        self.system_instruction = self.format_system_instruction(self.ontology, example_dialogs, domains)
        print(self.system_instruction)
        self.model = LLM(model_info, self.system_instruction)
        

    def format_examples(self, example_dialogs): 
        example = ''
        for example_dialog in example_dialogs:
            # print(example_dialog['domains'])
            prev_utt = ''
            for i, turn in enumerate(example_dialog['turns']):
                utterance = turn['utterance'].replace('\n', ' ')
                if self.speaker in turn['speaker']:
                    if prev_utt:
                        example += example_dialog['turns'][i-1]['speaker']+': '+prev_utt+'\n'
                    example += turn['speaker']+': '+utterance+'\n'
                    das = []
                    for da_type in turn['dialogue_acts']:
                        for da in turn['dialogue_acts'][da_type]:
                            intent, domain, slot, value = da.get('intent'), da.get('domain'), da.get('slot', ''), da.get('value', '')
                            das.append((intent, domain, slot, value))
                    example += '<DA>'+json.dumps(das)+'</DA>'+'\n\n'
                prev_utt = utterance
        return example
    
    def format_system_instruction(self, ontology, example_dialogs, domains):
        # Include alternative slot values as "option1|option2."
        intents = {intent: ontology['intents'][intent]['description'] for intent in ontology['intents']}
        if domains is None: # use all domains
            domains = ontology['domains'].keys()
        slots = {domain: {
                    slot: f'{slot_val["description"]} {"(possible values: " + str(slot_val.get("possible_values",[])) + ")" if slot_val.get("possible_values",[]) else "(not categorical)"}' 
                    for slot, slot_val in domain_val['slots'].items()
                } for domain, domain_val in ontology['domains'].items() if domain in domains}
        
        example = self.format_examples(example_dialogs)
        
        system_instruction = ["""You are an excellent dialogue acts parser. Dialogue acts are used to represent the intention of the speaker. Dialogue acts are a list of tuples, each tuple is in the form of (intent, domain, slot, value). The "intent", "domain", "slot" are defined as follows:""",
                '"intents": '+json.dumps(intents, indent=4),
                '"domain2slots": '+json.dumps(slots, indent=4),
                """Here are example dialogue acts:""",
                example,
                #"""Now consider the following dialogue. Please generate the dialogue acts for the last utterance{}. Start with <DA> token and end with </DA> token. Example: "<DA>[["inform", "hotel", "name", "abc"]]</DA>". Do not generate intents, domains, slots that are not defined above. Please include special slot values in the utterance like [hotel name], [phone], [address], and [ref].""".format(f' of {self.speaker}' if self.speaker else '')
                """Now consider the following dialogue. Please generate the dialogue acts for the last utterance{}. Start with <DA> token and end with </DA> token. Example: "<DA>[["inform", "travel", "name", "abc"]]</DA>". Do not generate intents, domains, slots that are not defined above.""".format(f' of {self.speaker}' if self.speaker else '')
                ]
        
        if not example:
            system_instruction = system_instruction[:3] + system_instruction[-1:]
            
        return "\n\n".join(system_instruction)

    
    def predict(self, utterance, context=list(), context_window = 3, log=False):
        prompt = '\n'.join(['Chat history:'] + context[-context_window:] + ['\nGenerate dialogue act for the following utterance:'] + [utterance]) # context and the current utterance
        # print('='*50)
        return repeat_until(lambda: self.parse_dialogue_acts(self.model.chat(prompt, log)), 
                            lambda x: isinstance(x, list),
                            max_iterations=1,
                            default_result=[], 
                            log=log, 
                            operation_btw_repeat=lambda: time.sleep(8))
    
    def parse_dialogue_acts(self, response):
        if response is None: 
            return {}
        start_token, end_token = "<DA>", "</DA>"
        start_idx = response.find(start_token)
        end_idx = response.find(end_token)
        if start_idx == -1 or end_idx == -1:
            return {}
        response = response[start_idx+len(start_token):end_idx].strip()
        if response == "":
            return {}
        try:
            dialogue_acts = json.loads(response)
        except json.decoder.JSONDecodeError:
            return {}
        return dialogue_acts

class Frames_DA_Extractor:
    def __init__(self): 
        self.intent_ontology = json.load(open('dataset/Frames/mwozIntent_ontology_frames.json', 'r'))
        self.valid_intents = list(self.intent_ontology['intents'].keys())
        self.valid_domains = list(self.intent_ontology['domains'].keys()) + ['general']
        self.valid_slots_values = {slot: [str(possible_val) for possible_val in slot_info['possible_values']]
                    for domain in self.intent_ontology['domains'].keys()
                    for slot, slot_info in self.intent_ontology['domains'][domain]['slots'].items()}
        
        llm_model_info = {'name': "huggingface/Qwen/Qwen2.5-7B-Instruct"}
        example_dialogues = load_json('dataset/Frames/frame_example_da.json')['example_dialogs']
        self.nlu_model = LLM_NLU('dataset/Frames/mwozIntent_ontology_frames.json', 
                                llm_model_info, '', example_dialogues, domains=['travel'])

    def check_invalid_das(self, utt_da): 
        invalid_das = []
        valid_utt_da = []
        for da in utt_da: 
            da_is_valid = True
            if da[0] not in self.valid_intents: 
                invalid_das.append({'invalid_da': da, 'invalid': 'intent', 'invalid_val': da[0]})
                da_is_valid = False
            if da[1] not in self.valid_domains:
                invalid_das.append({'invalid_da': da, 'invalid': 'domain', 'invalid_val': da[1]})
                da_is_valid = False
            if da[2] and da[2] not in self.valid_slots_values: 
                invalid_das.append({'invalid_da': da, 'invalid': 'slot', 'invalid_val': da[2]})
                da_is_valid = False
            # empty slot list means not categorical (possible values not listed), no need to check value
            if da[3] and ((not da[2]) or (self.valid_slots_values.get(da[2], []) and da[3] not in self.valid_slots_values[da[2]])): 
                invalid_das.append({'invalid_da': da, 'invalid': 'value', 'invalid_val': da[3]})
                da_is_valid = False
            if da_is_valid: 
                valid_utt_da.append(da)
        return invalid_das, valid_utt_da

    # basic fixes
    def fix_dialogue_acts(self, utt, utt_da): 
        fixed_utt_da = copy.deepcopy(utt_da)
        to_remove = []
        for da_i, da in enumerate(utt_da): 
            if len(da) < 4: 
                # if da is not complete, fill the missing slots with empty strings
                da = da + ['']*(4-len(da))
                fixed_utt_da[da_i] = da

            # errors regarding intent
            if da[0] and da[0] not in self.valid_intents: 
                if da[0] in ['confirm', 'correct']: 
                    # confirming a piece of information == informing
                    if da[2] and da[3]: 
                        fixed_utt_da[da_i][0] = 'inform'
                    # not confirming anything, remove
                    else: 
                        to_remove.append(da)

                # actions not of interest
                elif da[0] in ['apologize', 'wish', 'sorry']: 
                    to_remove.append(da)
                da = fixed_utt_da[da_i]

            # errors regarding domain
            if not da[1] or da[1] not in self.valid_domains: 
                # greet without domain
                if da[1]=='' and da[0] == 'greet': 
                    fixed_utt_da[da_i][1] = 'general'
                da = fixed_utt_da[da_i]

            # errors regarding slot
            if da[2] and da[2] not in self.valid_slots_values.keys(): 
                # system informing number of choices
                if utt.startswith('sys') and da[2] == 'choice' and da[3].isdigit(): 
                    fixed_utt_da[da_i][2] = 'count'

                # requesting more details and information
                elif da[2] in ['details', 'info']: 
                    if da[3] and da[3] in self.valid_slots_values.keys():  # never happens (requesting detail regarding a specific slot)
                        fixed_utt_da[da_i] = [da[0], da[1], da[3], '']
                    else: 
                        fixed_utt_da[da_i] = ['reqmore', da[1], '', ''] # requesting more information in general
                
                # non-existing acts for non-interested phrases (e.g. 'interested?', 'help me/any suggestions/recommendations/options/availabilities/flights included/discount?', 'pool', 'so excited')
                elif da[2] in ['interest', 'suggestions', 'choice', 'choices', 'recommendation', 'recommendations', 'options', 'help', 'availability', 'flights', 'discount', 'excitement', 'pool', 'privacy', 'weather', 'meal', 'airline']: 
                    to_remove.append(da)  # remove 'interested?', asking for suggestions and choices
                    continue

                # intent / confirm booking
                elif da[2] in ['booking', 'book']: 
                    fixed_utt_da[da_i] = ['book', da[1], '', ''] 
                
                # confirming booking / if user is interested
                elif da[2] == 'confirmation': 
                    if 'book' in utt: # user booking
                        fixed_utt_da[da_i] = ['book', da[1], '', ''] 
                    else: # system confirming if user is interested
                        to_remove.append(da) 
                        continue

                # activities = amenities, neaby_attractions = vicinity, flight_class/class = seat, rating = gst_rating
                elif da[2] in ['activities']: 
                    fixed_utt_da[da_i][2] = 'amenities'
                elif da[2] in ['nearby_attractions', 'location']: 
                    fixed_utt_da[da_i][2] = 'vicinity'
                elif da[2] in ['flight_class', 'class']: 
                    fixed_utt_da[da_i][2] = 'seat'
                elif da[2] == 'n_infants': 
                    fixed_utt_da[da_i][2] = 'n_children'
                elif da[2] == 'rating': 
                    fixed_utt_da[da_i][2] = 'gst_rating'

                # intent / action / write
                elif da[2] in ['intent', 'action', 'write']:
                    to_remove.append(da)
                    continue 
                
                # give up fixing
                else: # cannot be fixed
                    return fixed_utt_da
                
                da = fixed_utt_da[da_i]
            
            # errors regarding slot value
            if da[3] and ((not da[2]) or (self.valid_slots_values[da[2]] and da[3] not in self.valid_slots_values[da[2]])): 
                if da[2] == 'vicinity' and any([word in self.valid_slots_values for word in da[3].split()]): 
                    vicinity_slots = set([word for word in da[3].split() if word in self.valid_slots_values])
                    to_remove.append(da)
                    for slot in vicinity_slots: 
                        fixed_utt_da.append([da[0], da[1], slot, 'True'])
                    continue
                elif da[2] in ['amenities', 'vicinity', 'breakfast'] and da[3] in ['basic', 'some']: 
                    fixed_utt_da[da_i][3] = 'True'
                elif da[2] in ['amenities', 'vicinity', 'breakfast'] and 'no' in da[3]:
                    fixed_utt_da[da_i][3] = 'False'
                elif da[0] == 'greet': 
                    fixed_utt_da[da_i][2], fixed_utt_da[da_i][3] = '', ''
                else: 
                    return fixed_utt_da  # cannot be fixed
        
        for da in to_remove: 
            if da in fixed_utt_da: 
                fixed_utt_da.remove(da)
        return fixed_utt_da

    def __call__(self, utt, chat_history): 
        DA = repeat_until(lambda: self.check_invalid_das(self.fix_dialogue_acts(utt, self.nlu_model.predict(utt, chat_history, 3))), 
                            lambda x: x[0] == [], 
                            max_iterations = 1, 
                            default_result = 'latest result', 
                            log = False, 
                            operation_btw_repeat=None)[1]
        return mask_sensitive_slot(DA)



class NLU:
    def __init__(self, model_name): 
        self.model_name = model_name
        config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, config=config)
    
    def deserialize_dialogue_acts(self, das_seq):
        dialogue_acts = []
        if len(das_seq) == 0:
            return dialogue_acts
        da_seqs = das_seq.split(']);[')  # will consume "])" and "["
        for i, da_seq in enumerate(da_seqs):
            if len(da_seq) == 0 or len(da_seq.split(']([')) != 2:
                continue
            if i == 0:
                if da_seq[0] == '[':
                    da_seq = da_seq[1:]
            if i == len(da_seqs) - 1:
                if da_seq[-2:] == '])':
                    da_seq = da_seq[:-2]
            
            try:
                intent_domain, slot_values = da_seq.split(']([')
                intent, domain = intent_domain.split('][')
            except:
                continue
            for slot_value in slot_values.split('],['):
                try:
                    slot, value = slot_value.split('][')
                except:
                    continue
                dialogue_acts.append({'intent': intent, 'domain': domain, 'slot': slot, 'value': value})
            
        return dialogue_acts

    def predict(self, utterance, context = list(), context_window=0):
        if context_window > 0: 
            context = context[-context_window:]
            utterance = '\n'.join(context) + '\n' + utterance
        inputs = self.tokenizer(utterance,return_tensors="pt")
        output_tokens = self.model.generate(
            input_ids=inputs["input_ids"],
            max_length=100, 
        )
        output_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        das = self.deserialize_dialogue_acts(output_text.strip())
        dialog_act = []
        for da in das:
            dialog_act.append([da['intent'], da['domain'], da['slot'], da.get('value','')])
        return dialog_act


class MWoz_DA_Extractor:
    nlu_model_name_or_path = "ConvLab/t5-small-nlu-all-multiwoz21-context3"
    def __init__(self): 
        self.nlu_model = NLU(MWoz_DA_Extractor.nlu_model_name_or_path)

    def __call__(self, utt, chat_history): 
        DA = self.nlu_model.predict(utt, chat_history, 3)
        return mask_sensitive_slot(DA)
