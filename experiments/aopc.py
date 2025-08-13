import numpy as np
from XFramework.SHAP_Explainer import SHAP_Explainer
from XFramework.DataManager import DataManager
import pandas as pd
import torch
from torch.nn import functional as F
from tqdm import tqdm
import random


def AOPC(pred_probs): 
    '''
    Calculates AOPC_k for a single sample
    '''
    pred_0 = pred_probs[0]
    drop_k = [pred_0 - pred for pred in pred_probs]
    aopc_k = [cumsum/(i+1) for i, cumsum in enumerate(np.cumsum(drop_k))]
    return aopc_k



def iterative_feature_removal(sentence, tokenizer, token_removal_order, k):
    ### This removes ALL existence of ONE FEATURE at a time
    tokens = tokenizer.tokenize(sentence)
    tokens_lc = [t.lower() for t in SHAP_Explainer.get_utt_features(sentence, tokenizer)]

    token_tuples = list(zip(tokens, tokens_lc))
    all_token_ids = [tokenizer.convert_tokens_to_ids(tokens)]

    for target_token in token_removal_order[:k-1]:
        token_tuples = [pair for pair in token_tuples if pair[1] != target_token]
        tokens_step = [t for t, _ in token_tuples]
        all_token_ids.append(tokenizer.convert_tokens_to_ids(tokens_step))

    return all_token_ids



def get_token_removal_order(method, importance, utterance_features, utterance_da, dia_no, turn_no, label, pred):
    # AOPC_global from "Global Aggregations of Local Explanations for Black Box models"
    match method: 
        case 'local': 
            removal_token = [expl[0][0].split()[0].lower() for expl in sorted(importance.loc[importance['dia_turn_label']==(dia_no, turn_no, label)]['explanation'].values[0], key=lambda x:x[1], reverse=pred==1)]
        case 'DA': 
            utterance_da_str = [str(da) for da in utterance_da]
            matched_da_scores = [importance[da] for da in utterance_da_str if da in importance]
            utt_scores = pd.DataFrame()
            if matched_da_scores: 
                utt_scores = pd.concat(matched_da_scores, axis=0)
            if len(utt_scores)==0: # fall back to global if no DA matched
                print('Global used for DA:', (dia_no, turn_no, label), flush=True)
                method = 'global'
                utt_scores = importance['global']
            summed_scores = utt_scores.groupby(by='utt')['score'].sum()
            removal_order = sorted({tok:summed_scores.get(tok,0) for tok in utterance_features.lower().split()}.items(), key=lambda x:x[1], reverse=pred==1)
            removal_token = [_[0] for _ in removal_order]
        case 'global': 
            utt_scores = importance['global']
            global_summed_scores = utt_scores.groupby(by='utt')['score'].sum()
            removal_order = sorted({tok:global_summed_scores.get(tok,0) for tok in utterance_features.lower().split()}.items(), key=lambda x:x[1], reverse=pred==1)
            removal_token = [_[0] for _ in removal_order]
        case 'random':
            removal_token = utterance_features.lower().split()
            random.shuffle(removal_token)
    return removal_token, method


def get_token_removal_order_accelerating(method, importance, utterance_features, utterance_da, dia_no, turn_no, label, pred):
    # AOPC_global from "Accelerating the Global Aggregation of Local Explanations"
    match method: 
        case 'local': 
            removal_token = [expl[0][0].split()[0].lower() for expl in sorted(importance.loc[importance['dia_turn_label']==(dia_no, turn_no, label)]['explanation'].values[0], key=lambda x:x[1], reverse=(pred==1))]
        case 'DA': 
            utterance_da_str = [str(da) for da in utterance_da]
            matched_da_scores = [importance[da] for da in utterance_da_str if da in importance]
            utt_scores = pd.DataFrame()
            if matched_da_scores: 
                utt_scores = pd.concat(matched_da_scores, axis=0)
            if len(utt_scores)==0: # fall back to global if no DA matched
                print('Global used for DA:', (dia_no, turn_no, label), flush=True)
                method = 'global'
                utt_scores = importance['global']
            summed_scores = utt_scores.groupby(by='utt')['score'].sum().sort_values(ascending=(pred==0))
            removal_token = summed_scores.index.tolist()
        case 'global': 
            utt_scores = importance['global']
            global_summed_scores = utt_scores.groupby(by='utt')['score'].sum().sort_values(ascending=(pred==0))
            removal_token = global_summed_scores.index.tolist()
        case 'random':
            utt_scores = importance['global']
            removal_token = np.unique(utt_scores['utt'])
            random.shuffle(removal_token)
    return removal_token, method


def aopc_comparison_dataset(dataset, K, predictions, model, tokenizer, prediction_class, local_expl, da_utt_scores, methods, aopc_method): 
    assert aopc_method in ['original', 'accelerating'], "aopc_method must be either 'original' or 'accelerating'"
    print(methods, flush=True) # comparison methods
    print("AOPC method:", aopc_method, flush=True) # AOPC calculation method
    token_removal_order_fn = get_token_removal_order if aopc_method == 'original' else get_token_removal_order_accelerating

    da_utt_scores = {k:v for k,v in da_utt_scores.items() if len(v)>0}
    predictions = {(sample['ground_truth'], sample['dia_no'], sample['turn_no']):sample['pred_label'] for sample in predictions if sample['pred_label'] in prediction_class}
    AOPCs = []
    cls_token = tokenizer.cls_token
    for row in tqdm(dataset.itertuples(index=False), total=len(dataset)):
        sample_utterances = row.dia.split("\n")[::2]
        label, dia_no = row.label, row.dia_no
        for turn_no, utterance in enumerate(sample_utterances):
            pred = predictions.get((label, dia_no, turn_no), None)
            if pred is None: continue

            # get utterance information
            utterance = utterance[6:] # remove "user: "
            utterance_features = ' '.join(SHAP_Explainer.get_utt_features(utterance, tokenizer))
            cls_utterance = cls_token + ' ' + utterance
            utterance_da = [da[:3] for da in row.dialogue_act_info_removed[2*turn_no]]

            # get removal order of the tokens
            removal_orders = {}
            if 'random' in methods: 
                removal_orders['random'], final_DA_method = token_removal_order_fn('random', local_expl, utterance_features, utterance_da, dia_no, turn_no, label, pred)
            if 'local' in methods: 
                removal_orders['local'], final_DA_method = token_removal_order_fn('local', local_expl, utterance_features, utterance_da, dia_no, turn_no, label, pred)
            if 'DA' in methods: 
                removal_orders['DA'], final_DA_method = token_removal_order_fn('DA', da_utt_scores, utterance_features, utterance_da, dia_no, turn_no, label, pred)
            if 'global' in methods: 
                removal_orders['global'], final_DA_method = token_removal_order_fn('global', da_utt_scores, utterance_features, utterance_da, dia_no, turn_no, label, pred)

            # calculate aopc
            sample_AOPCs = {'label': row.label, 'dia_no': row.dia_no, 'turn_no': turn_no, 'final_DA_method': final_DA_method}
            for method, removal_order in removal_orders.items():
                iterative_removed = iterative_feature_removal(cls_utterance, tokenizer, removal_order, K)
                padded_batch = tokenizer.pad(
                    [{'input_ids': input_ids[:DataManager.UTT_TURN_MAX_LEN]} for input_ids in iterative_removed],
                    padding='max_length',       
                    max_length=DataManager.UTT_TURN_MAX_LEN,           
                    return_tensors='pt'        
                )
                with torch.no_grad():
                    outputs = model(**padded_batch)

                # Post-processing
                probs = F.softmax(outputs.logits, dim=-1)
                sample_AOPCs[method] = AOPC(probs[:,pred])
            AOPCs.append(sample_AOPCs)

    return AOPCs