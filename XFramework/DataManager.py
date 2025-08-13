from datasets import Dataset
from XFramework.utils import read_pickle, RANDOM_STATE
from transformers import AutoTokenizer
from torch.utils.data.dataloader import DataLoader, RandomSampler, SequentialSampler
import pandas as pd
import numpy as np
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
os.environ["WANDB_DISABLED"] = "true"


def update_turn_no(row):
    if pd.notna(row['turn_no']):  # If 'turn_no' is not NaN, keep the original value and put in list
        return [int(row['turn_no'])]
    else:  # If 'turn_no' is NaN, count 'user: ' occurrences in 'dia' and generate range
        user_count = row['dia'].count('user: ')
        return list(range(user_count))

def fill_turn_no(df): 
    if 'turn_no' not in df.columns: 
        df['turn_no'] = np.nan
    df['turn_no'] = df.apply(update_turn_no, axis=1)
    return df

def load_dataset(bona_fide_dataset, da_extraction_method): 
    if bona_fide_dataset == 'mwoz': 
        # convlab generated dialogue acts
        if da_extraction_method == 'convlab': 
            e2e_gg = pd.DataFrame(read_pickle('dataset/SPADE/e2e_gg.pkl'))
            e2e_gg['label'] = 1
            e2e_gl = pd.DataFrame(read_pickle('dataset/SPADE/e2e_gl.pkl'))
            e2e_gl['label'] = 1
            e2e_lg = pd.DataFrame(read_pickle('dataset/SPADE/e2e_lg.pkl'))
            e2e_lg['label'] = 1
            e2e_ll = pd.DataFrame(read_pickle('dataset/SPADE/e2e_ll.pkl'))
            e2e_ll['label'] = 1
            bona_fide_human = pd.DataFrame(read_pickle('dataset/SPADE/bona_fide_human.pkl'))
            bona_fide_human['label'] = 0
        # e2e_gg, e2e_gl, e2e_lg, e2e_ll
        ai_dataset = pd.concat([e2e_gg, e2e_gl, e2e_lg, e2e_ll], ignore_index=True).groupby('dia_no').sample(n=1, random_state=RANDOM_STATE).reset_index(drop=True)
        splits = read_pickle('dataset/SPADE/dataset_splits.pkl')
    elif bona_fide_dataset == 'frames': 
        ai_dataset = pd.DataFrame(read_pickle('dataset/Frames/frames_e2e.pkl'))
        ai_dataset['label'] = 1
        bona_fide_human = pd.DataFrame(read_pickle('dataset/Frames/frames_bona_fide.pkl'))
        bona_fide_human['label'] = 0
        splits = read_pickle('dataset/Frames/dataset_splits.pkl')

    dataset = pd.concat([bona_fide_human, ai_dataset], ignore_index=True)
    dataset = fill_turn_no(dataset)
    dataset['dataset'] = bona_fide_dataset

    # Split into train/val/test
    train_ids, val_ids, test_ids = splits['train'], splits['val'], splits['test']
    train_dataset = dataset[dataset['dia_no'].isin(train_ids)]
    val_dataset = dataset[dataset['dia_no'].isin(val_ids)]
    test_dataset = dataset[dataset['dia_no'].isin(test_ids)]
    return train_dataset, val_dataset, test_dataset

def load_ontology(bona_fide_dataset): 
    if bona_fide_dataset == 'mwoz': 
        ontology = json.load(open('dataset/SPADE/ontology_mwoz.json','r'))['domains']['hotel']['slots']
    elif bona_fide_dataset == 'frames': 
        ontology = json.load(open('dataset/Frames/ontology_frames.json','r'))['domains']['travel']['slots']
    else: 
        raise ValueError(f"Unknown dataset {bona_fide_dataset} for ontology loading.")
    return ontology

NONE_DA = [['none', '', '', '']]
class DataManager:
    use_da_value = True # whether to use the value of the da
    use_da_turn  = True # whether to da of a single role or a turn (both role)
    DA_TURN_MAX_LEN = 256
    UTT_TURN_MAX_LEN = 128
    DIA_MAX_LEN = 384
    
    # special tokens
    DA_SEP = '<da_sep>'
    SLOT_SEP = '<slot_sep>'
    USR = '<usr>'
    SYS = '<sys>'

    def __init__(self, train_dataset, val_dataset, test_dataset, level = 'turn', model_name = 'roberta-base', tokenizer = None, dataset_preprocess = None, dataset_initialize = 'combined', batch_size = 16, max_len = 128, truncate=True):
        self.level = level
        self.max_len = max_len
        self.dataset_preprocess = dataset_preprocess # a function to retrieve samples (input_sequence, attention_masks) expected by the model
        self.batch_size = batch_size
        self.truncate = truncate
        self.tokenizer = tokenizer if tokenizer is not None else DataManager.new_tokenizer(model_name)
        data = dict()
        if dataset_initialize == 'combined': # only one set of input_ids (for DA-only / Utt-only / combined DA+Utt)
            data['train'] = self.initialize_dataset(train_dataset)
            data['val'] = self.initialize_dataset(val_dataset)
            data['test'] = self.initialize_dataset(test_dataset)
        else: # two sets of input_ids (for DA and Utt separately)
            data['train'] = self.initialize_multimodal_dataset(train_dataset)
            data['val'] = self.initialize_multimodal_dataset(val_dataset)
            data['test'] = self.initialize_multimodal_dataset(test_dataset)
        self.train_dataloader = self.get_train_dataloader(data["train"])
        self.val_dataloader = self.get_eval_dataloader(data["val"])
        self.test_dataloader = self.get_eval_dataloader(data["test"])

    def new_tokenizer(model_name='roberta-base'): 
        tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        tokenizer.add_special_tokens({'additional_special_tokens': [DataManager.DA_SEP, DataManager.SLOT_SEP, DataManager.USR, DataManager.SYS]})
        if "gpt" in model_name.lower():
            tokenizer.pad_token = tokenizer.eos_token # Set EOS as PAD
            tokenizer.mask_token = tokenizer.pad_token # Set PAD as MASK
            tokenizer.add_special_tokens({'sep_token': '<sep>'})
            tokenizer.cls_token = tokenizer.eos_token # Set EOS as CLS
            tokenizer.padding_side = "left"            # Pad left for autoregressive models
        return tokenizer

    def _utt_input_sequence(sample, tokenizer): 
        CLS = tokenizer.special_tokens_map['cls_token']

        utt_stripped = sample.replace('user:','').replace('system:','').strip()
        return CLS + ' ' + utt_stripped
    
    def utt_only(dataset, tokenizer, role=''): 
        '''
        Input features = utterances
        role: 'user' or 'system' or '' for both
        '''

        utterances = {'dia_no':[], 'turn_no': [], 'input_sequence':[], 'label':[], 'sample':[], 'sample_dia': []}
        for _, dialogue in dataset.iterrows(): 
            dia = dialogue['dia']
            label = dialogue['label']
            dia_no = dialogue['dia_no']
            turn = 0
            for _, utt in enumerate(dia.split('\n')): 
                if not utt.startswith(role): 
                    continue
                input_sequence = DataManager._utt_input_sequence(utt, tokenizer)
                utterances['dia_no'].append(dia_no)
                utterances['turn_no'].append(dialogue['turn_no'][turn])
                utterances['input_sequence'].append(input_sequence)
                utterances['label'].append(label)
                utterances['sample'].append(utt)
                utterances['sample_dia'].append(utt)
                turn += 1
        return pd.DataFrame(utterances)

    def _da_turn_input_sequence(sample, tokenizer): 
        CLS = tokenizer.special_tokens_map['cls_token']
        SEP = tokenizer.special_tokens_map['sep_token']

        if not DataManager.use_da_turn: 
            sample = [sample[-1]] # only take the last utterance's dialogue acts
        da_string = [CLS]
        # preceding opponent role's dialogue action, followed by target role's dialogue action
        for da in sample: 
            if da == NONE_DA: 
                da_string += ['none', SEP]
            else: 
                for action in da:
                    if not DataManager.use_da_value: 
                        action = action[:3] # ignore value component
                    for slot in action: 
                        da_string.append(slot)
                        da_string.append(DataManager.SLOT_SEP)
                    da_string[-1] = DataManager.DA_SEP
                da_string[-1] = SEP # using SEP to end each utterance's dialogue acts
        da_string = ' '.join(da_string)
        return da_string 

    def da_turn(dataset, tokenizer, role=''):
        '''
        Input features = dialogue acts
        role: 'user' or 'system' or '' for both
        '''
        dialogue_acts = {'dia_no':[], 'turn_no': [], 'input_sequence':[], 'label':[], 'sample':[], 'sample_dia':[]}
        for _, dialogue in dataset.iterrows(): 
            dia_da = dialogue['dialogue_act_info_removed']
            dia = dialogue['dia']
            label = dialogue['label']
            dia_no = dialogue['dia_no']

            # for first user utterance, assume system has a greeting dialogue act
            last_da = [['greet', 'general', '', '']] if dia.startswith("user: ") else NONE_DA
            last_utt = 'system: Hello! How can I help you today?' if dia.startswith("user: ") else 'user: none'
            turn = 0
            for _, (utt_da, utt) in enumerate(zip(dia_da, dia.split('\n'))): 
                # for empty dialogue acts, specify none
                if not utt_da:
                    utt_da = NONE_DA
                # record and skip opponent's utterance
                if not utt.startswith(role): 
                    last_da = utt_da
                    last_utt = utt
                    continue
                
                # preceding opponent role's dialogue act, followed by target role's dialogue act
                da_string = DataManager._da_turn_input_sequence([last_da, utt_da], tokenizer)
                dialogue_acts['dia_no'].append(dia_no)
                dialogue_acts['turn_no'].append(dialogue['turn_no'][turn])
                dialogue_acts['input_sequence'].append(da_string)
                dialogue_acts['label'].append(label)
                if DataManager.use_da_turn: 
                    dialogue_acts['sample'].append([last_da, utt_da])
                    dialogue_acts['sample_dia'].append('\n'.join([last_utt, utt]))
                else: 
                    dialogue_acts['sample'].append(utt_da)
                    dialogue_acts['sample_dia'].append(utt)
                turn += 1
        return pd.DataFrame(dialogue_acts)
    
    def _da_utt_input_sequence(sample, tokenizer, use_da, use_utt): 
        CLS = tokenizer.special_tokens_map['cls_token']
        SEP = tokenizer.special_tokens_map['sep_token']
        
        ROLE = [DataManager.USR, DataManager.SYS] if DataManager.use_da_turn else [DataManager.USR, DataManager.USR]

        dia_da, dia_utt = sample[0], sample[1]
        da_utt_string = [CLS]
        # dialogue acts
        if use_da: 
            for i, da in enumerate(dia_da): 
                da_utt_string.append(ROLE[i%2]) # always starts with user, and iterates between user and system
                if da == NONE_DA: 
                    da_utt_string += ['none']
                else: 
                    for action in da:
                        if not DataManager.use_da_value: 
                            action = action[:3] # ignore value
                        for slot in action: 
                            if not slot: 
                                continue
                            da_utt_string.append(slot)
                        da_utt_string.append(DataManager.DA_SEP)
                    da_utt_string.pop()
            
        # utterance
        if use_da and use_utt: 
            da_utt_string.append(SEP)
        if use_utt: 
            da_utt_string.append(DataManager.USR)
            dia_utt = dia_utt.replace("user: ", "").replace("\n", f" {DataManager.USR} ")
            da_utt_string.append(dia_utt)
        
        da_utt_string = ' '.join(da_utt_string)
        return da_utt_string 

    def da_utt(dataset, tokenizer, role='', da=True, utt=True): 
        model_dataset = {'dia_no': dataset['dia_no'].values.tolist(), 
                         'input_sequence':[], 
                         'label': dataset['label'].values.tolist(), 
                         'sample':[]}
        if 'turn_no' in dataset.columns: 
            model_dataset['turn_no'] = dataset['turn_no'].values.tolist()
        if 'is_complete' in dataset.columns: 
            model_dataset['is_complete'] = dataset['is_complete'].values.tolist()

        for _, dialogue in dataset.iterrows(): 
            dia_da = dialogue['masked_da']
            dia_utt = dialogue['masked_utt']

            # preceding opponent role's dialogue action, followed by target role's dialogue action
            da_utt_string = DataManager._da_utt_input_sequence([dia_da, dia_utt], tokenizer, da, utt)
            model_dataset['input_sequence'].append(da_utt_string)
            model_dataset['sample'].append(dia_da + [[[dia_utt,'','','']]])
        return pd.DataFrame(model_dataset)




    def initialize_dataset(self, dataset): 
        '''
        Prepares the dataset into the format expected by the model
        '''
        if dataset is None: 
            return None
        if self.dataset_preprocess is not None: 
            dataset = self.dataset_preprocess(dataset, self.tokenizer, 'user: ')
        
        samples_dict = {'input_ids':[], 'attention_mask':[], 
                        'label': dataset['label'].values.tolist(), 
                        'sample': dataset['sample'].values.tolist(), 
                        'dia_no': dataset['dia_no'].values.tolist()}
        
        # Add turn-level metadata if needed
        if self.level == 'turn': 
            samples_dict['sample_dia'] = dataset['sample_dia'].values.tolist()
            samples_dict['turn_no'] = dataset['turn_no'].values.tolist()
        
        # Add dialogue-level metadata if needed
        if self.level == 'dialogue': 
            if 'turn_no' in dataset.columns: 
                samples_dict['turn_no'] = dataset['turn_no'].values.tolist()
            if 'is_complete' in dataset.columns: 
                samples_dict['is_complete'] = dataset['is_complete'].values.tolist()

        for _, sample in dataset.iterrows(): 
            encodings = self.tokenizer(sample['input_sequence'], add_special_tokens=False, padding = 'max_length', truncation = self.truncate, max_length = self.max_len)
            samples_dict['input_ids'].append(encodings['input_ids'])
            samples_dict['attention_mask'].append(encodings['attention_mask'])

        return Dataset.from_dict(samples_dict).with_format("torch")
    
    def _multimodal_da_utt_input_sequence(sample, tokenizer): 
        CLS = tokenizer.special_tokens_map['cls_token']
        SEP = tokenizer.special_tokens_map['sep_token']
        ROLE = [DataManager.USR, DataManager.SYS] if DataManager.use_da_turn else [DataManager.USR, DataManager.USR]

        dia_da, dia_utt = sample[0], sample[1]
        # Build DA sequence with [CLS] and [SEP]
        da_string = [CLS]
        for i, da in enumerate(dia_da):
            da_string.append(ROLE[i % 2])
            if da == NONE_DA:
                da_string += ['none']
            else:
                for action in da:
                    if not DataManager.use_da_value: 
                        action = action[:3] # ignore value
                    for slot in action:
                        da_string.append(slot)
                        da_string.append(DataManager.SLOT_SEP)
                    da_string[-1] = DataManager.DA_SEP
                da_string.pop()
        da_string.append(SEP)
        da_string = ' '.join(da_string)

        # Build utterance sequence with [CLS] and [SEP]
        utt_string = CLS + ' ' + dia_utt.replace("user: ", "").replace("\n", f" {DataManager.USR} ") + ' ' + SEP

        return da_string, utt_string

    def multimodal_da_utt(dataset, tokenizer, role=''): 
        model_dataset = {
            'dia_no': dataset['dia_no'].values.tolist(),
            'da_input': [],  # Separate DA input
            'utt_input': [],  # Separate utterance input
            'label': dataset['label'].values.tolist(),
            'sample': []
        }
        if 'turn_no' in dataset.columns: 
            model_dataset['turn_no'] = dataset['turn_no'].values.tolist()
        if 'is_complete' in dataset.columns: 
            model_dataset['is_complete'] = dataset['is_complete'].values.tolist()
        for _, dialogue in dataset.iterrows():
            dia_da = dialogue['masked_da']
            dia_utt = dialogue['masked_utt']
            da_str, utt_str = DataManager._multimodal_da_utt_input_sequence([dia_da, dia_utt], tokenizer)
            model_dataset['da_input'].append(da_str)
            model_dataset['utt_input'].append(utt_str)
            model_dataset['sample'].append(dia_da + [[[dia_utt,'','','']]])
        return pd.DataFrame(model_dataset)

    def initialize_multimodal_dataset(self, dataset):
        '''
        Prepares a multimodal dataset (dialogue acts + utterances) into the format expected by the model.
        Returns a Hugging Face Dataset with separate inputs for both modalities.
        '''
        if dataset is None:
            return None
        if self.dataset_preprocess is not None:
            dataset = self.dataset_preprocess(dataset, self.tokenizer, role='user: ')
        
        # Initialize samples dictionary with multimodal fields
        samples_dict = {
            'da_input_ids': [],
            'da_attention_mask': [],
            'utt_input_ids': [],
            'utt_attention_mask': [],
            'label': dataset['label'].values.tolist(),
            'sample': dataset['sample'].values.tolist(),
            'dia_no': dataset['dia_no'].values.tolist()
        }
        
        # Add turn-level metadata if needed
        if self.level == 'turn':
            samples_dict['sample_dia'] = dataset['sample_dia'].values.tolist()
            samples_dict['turn_no'] = dataset['turn_no'].values.tolist()

        # Add dialogue-level metadata if needed
        if self.level == 'dialogue': 
            if 'turn_no' in dataset.columns: 
                samples_dict['turn_no'] = dataset['turn_no'].values.tolist()
            if 'is_complete' in dataset.columns: 
                samples_dict['is_complete'] = dataset['is_complete'].values.tolist()

        # Tokenize two modalities separately
        for _, sample in dataset.iterrows():
            # Tokenize dialogue acts
            da_encodings = self.tokenizer(
                sample['da_input'],
                add_special_tokens=False,
                padding='max_length',
                truncation=self.truncate,
                max_length=self.max_len
            )
            
            # Tokenize utterance
            utt_encodings = self.tokenizer(
                sample['utt_input'],
                add_special_tokens=False,
                padding='max_length',
                truncation=self.truncate,
                max_length=self.max_len
            )
            
            # Append to samples dict
            samples_dict['da_input_ids'].append(da_encodings['input_ids'])
            samples_dict['da_attention_mask'].append(da_encodings['attention_mask'])
            samples_dict['utt_input_ids'].append(utt_encodings['input_ids'])
            samples_dict['utt_attention_mask'].append(utt_encodings['attention_mask'])

        return Dataset.from_dict(samples_dict).with_format("torch")

    def get_train_dataloader(self, dataset): 
        if dataset is None: 
            return None
        return DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=self.batch_size)
    
    def get_eval_dataloader(self, dataset): 
        if dataset is None: 
            return None
        return DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=self.batch_size)
