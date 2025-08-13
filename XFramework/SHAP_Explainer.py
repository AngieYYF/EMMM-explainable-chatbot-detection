from XFramework.attribution.shap_interaction.explainer import SamplePerturbation, TextExplainer
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
import os
from XFramework.DataManager import DataManager, NONE_DA

class SHAP_Explainer:
    def __init__(self, gen_method, model, interaction_order, random_state, generate_data_path, log=True, save_results=True): 
        self.gen_method = gen_method
        self.model = model
        self.interaction_order = interaction_order
        self.random_state = random_state
        self.generate_data_path = generate_data_path
        self.log=log
        self.save_results=save_results

    def generate_perturbations(self, dataset, mapping_function, n_feature_calculator,
                     n_train_perturbations, n_valid_perturbations, 
                     n_tabular_average, train_data_path, valid_data_path): 
        if os.path.exists(train_data_path): 
            if n_valid_perturbations == 0 or os.path.exists(train_data_path):
                return train_data_path, valid_data_path

        np.random.seed(self.random_state)
        
        preds = self.model(dataset)

        train_perturbations, valid_perturbations = [], []
        generator = SamplePerturbation(self.model, mapping_function)
        iterator = tqdm(dataset, total=dataset.num_rows) if self.log else dataset
        for instance in iterator: 
            # generate perturbations and collect model predictions on perturbations
            n_features = n_feature_calculator(instance)
            if n_features < 1: 
                # print(n_features,'features:', instance, flush=True)
                train_perturbations.append((None, None))
                if n_valid_perturbations > 0:
                    valid_perturbations.append((None, None))
                continue
            
            # X_train = masks for perturbations of the instance
            # Y_train = model predictions for the perturbed samples
            X_train, Y_train = generator.sample_perturbations(n_train_perturbations, n_features, 
                                                        instance, self.gen_method, 
                                                        n_tabular_average = n_tabular_average, 
                                                        interaction_order=self.interaction_order, log=False)

            train_perturbations.append((X_train, Y_train))
            
            if n_valid_perturbations > 0:
                X_valid, Y_valid = generator.sample_perturbations(n_valid_perturbations, n_features, 
                                                                instance, self.gen_method, 
                                                                n_tabular_average = n_tabular_average,
                                                                interaction_order=self.interaction_order, log=False)
                valid_perturbations.append((X_valid, Y_valid))

        dia_no, turn_no, labels = dataset['dia_no'].tolist(), dataset['turn_no'].tolist(), dataset['label'].tolist()
        train_data = (dia_no, turn_no, labels, preds, train_perturbations)
        if self.save_results:
            pkl.dump(train_data, open(train_data_path, 'wb'))

        valid_data = None
        if n_valid_perturbations > 0:
            valid_data = (dia_no, turn_no, labels, preds, valid_perturbations)
            if self.save_results:
                pkl.dump(valid_data, open(valid_data_path, 'wb'))

        if self.save_results:
            return train_data_path, valid_data_path
        return train_data, valid_data

    def calculate_interaction_scores(self, interaction_order,
                                    dataset, feature_split, 
                                    train_data, valid_data, dump_results_path, 
                                    lasso_alpha = 0.001, 
                                    instance_process = lambda x:x): 
        dia_no, turn_no, labels, preds, train_perturbations = pkl.load(open(train_data,'rb')) if self.save_results else train_data
        if valid_data:
            dia_no, turn_no, labels, valid_preds, valid_perturbations = pkl.load(open(valid_data,'rb')) if self.save_results else valid_data
            assert dia_no == dia_no
        else:
            valid_perturbations = [None] * len(dia_no)
        
        # Explain
        explainer = TextExplainer(interaction_order, self.gen_method, lasso_alpha = lasso_alpha)

        results = []
        for idx, (instance, instance_dia_no, instance_turn_no, instance_label) in enumerate(zip(dataset, dia_no, turn_no, labels)): 
            processed_instance = instance_process(instance)
            instance_id = (instance_dia_no, instance_turn_no, instance_label)
        
            # print("Number of training perturbations:", len(train_perturbations[0]))
            # print("Length of data:", len(feature_split(processed_instance)),end='\n\n')
            
            # empty instance has no explanation
            if train_perturbations[idx][0] is None: 
                explanation = []
            else: 
                explanation, train_infd, valid_infd, _ = explainer.explain_instance(processed_instance,
                                                                                    train_perturbations[idx],
                                                                                    valid_perturbations[idx], 
                                                                                    feature_split=feature_split)

            # Show explanation
            # explainer.display(explanation, pred[0])
            # print("-"*150)
                
            results.append((explanation, len(feature_split(processed_instance)), instance_id))

        if dump_results_path is not None and self.save_results:
            pkl.dump(results, open(dump_results_path,'wb'))
        return results

    def da_actionLevel_shap_interaction(self, dataset, n_train_perturbations, tokenizer, model_name): 
        return self.shap_interaction(interaction_order=self.interaction_order, 
                                     dataset = dataset, 
                                     mapping_function=lambda x,y: SHAP_Explainer.da_turn_actionLevel_data_mapping(x,y,tokenizer=tokenizer, model_name=model_name), 
                                     n_feature_calculator=SHAP_Explainer.da_turn_actionLevel_n_features, 
                                     feature_split=SHAP_Explainer.get_da_turn_actionLevel_features, 
                                     n_train_perturbations=n_train_perturbations, n_valid_perturbations=0, 
                                     n_tabular_average=1
                                     )
    
    def utt_shap_interaction(self, dataset, n_train_perturbations, tokenizer, model_name): 
        return self.shap_interaction(interaction_order=self.interaction_order, 
                                     dataset = dataset, 
                                     mapping_function=lambda x,y: SHAP_Explainer.utt_data_mapping(x,y, tokenizer=tokenizer, model_name=model_name), 
                                     n_feature_calculator=lambda x: SHAP_Explainer.utt_n_features(x, tokenizer=tokenizer), 
                                     feature_split=lambda x: SHAP_Explainer.get_utt_features(x, tokenizer=tokenizer), 
                                     n_train_perturbations=n_train_perturbations, n_valid_perturbations=0, 
                                     n_tabular_average=1)

    def da_utt_actionLevel_shap_interaction(self, dataset, n_train_perturbations, tokenizer, model_name, use_da, use_utt): 
        assert not DataManager.use_da_turn, 'Dialogue level explanation only for user daTurn'
        return self.shap_interaction(interaction_order=self.interaction_order, 
                                     dataset = dataset, 
                                     mapping_function=lambda x,y: SHAP_Explainer.da_utt_data_mapping(x,y, tokenizer=tokenizer, model_name=model_name, use_da=use_da, use_utt=use_utt), 
                                     n_feature_calculator=lambda x: SHAP_Explainer.da_utt_n_features(x, tokenizer=tokenizer, use_da=use_da, use_utt=use_utt), 
                                     feature_split=lambda x: SHAP_Explainer.get_da_utt_features(x, tokenizer=tokenizer, use_da=use_da, use_utt=use_utt), 
                                     n_train_perturbations=n_train_perturbations, n_valid_perturbations=0, 
                                     n_tabular_average=1)

    def shap_interaction(self, interaction_order,
                        dataset, mapping_function, n_feature_calculator, feature_split, 
                        n_train_perturbations, n_valid_perturbations, 
                        n_tabular_average,
                        lasso_alpha=0.001, 
                        instance_process = lambda x:x): 
        train_data_path = f'{self.generate_data_path}.train'
        valid_data_path = f'{self.generate_data_path}.valid' if n_valid_perturbations > 0 else None
        dump_results_path = f'{self.generate_data_path}.output'
        

        train_data, valid_data = self.generate_perturbations(dataset,
                            mapping_function, n_feature_calculator,
                            n_train_perturbations, n_valid_perturbations, 
                            n_tabular_average, train_data_path, valid_data_path)
        
        result = self.calculate_interaction_scores(interaction_order, 
                                            dataset, feature_split, 
                                            train_data, valid_data, dump_results_path, 
                                            lasso_alpha, instance_process)
        
        return result
    

    
    def da_turn_actionLevel_data_mapping(binary_vectors, instance, tokenizer=None, sample_only=False, model_name='roberta-base'): 
        n_perturbations = len(binary_vectors)
        if sample_only: 
            instances = {'sample': []}
            da_turn = instance if DataManager.use_da_turn else [instance]
        else: 
            instances = {'sample': [], 
                        'input_sequence': [],
                        'sample_dia': [instance['sample_dia']]*n_perturbations, 
                        'label': [instance['label'].item()]*n_perturbations, 
                        'dia_no': [instance['dia_no'].item()]*n_perturbations, 
                        'turn_no': [instance['turn_no'].item()]*n_perturbations}
            da_turn = instance['sample'] if DataManager.use_da_turn else [instance['sample']]
            if tokenizer is None: 
                tokenizer = DataManager.new_tokenizer(model_name)
        # print("\n\n\n"+"="*60+"\n\n\n", flush=True)
        # print(da_turn, flush=True)
        for v in binary_vectors:
            # print(v, flush=True)
            masked_da_turn = []
            # da features
            feature_i = 0
            for da in da_turn: 
                if da == NONE_DA: # none dialogue acts are ignored as features
                    masked_da_turn.append(NONE_DA)
                    continue
                masked_da = []
                for action in da: 
                    if feature_i >= len(v): 
                        break
                    if v[feature_i] == 1: # active action
                        masked_da.append(action)
                    # remove in-active action (not masking them)
                    feature_i += 1
                if not masked_da: 
                    masked_da = NONE_DA
                masked_da_turn.append(masked_da)
            # print(masked_da_turn, flush=True)
            instances['sample'].append(masked_da_turn if DataManager.use_da_turn else masked_da_turn[-1])
            if not sample_only: 
                instances['input_sequence'].append(DataManager._da_turn_input_sequence(masked_da_turn, tokenizer))
                # print(instances['input_sequence'][-1], flush=True)
            # print("", flush=True)
        if sample_only: 
            return instances['sample']
        perturbed_dataset = DataManager(None, None, None, tokenizer=tokenizer, max_len=DataManager.DA_TURN_MAX_LEN).initialize_dataset(pd.DataFrame(instances))
        # print(tokenizer.decode(perturbed_dataset['input_ids'][0]), flush=True)
        # print(f'data_mapping complete: {len(binary_vectors)} perturbations')
        return perturbed_dataset

    def da_turn_actionLevel_n_features(instance):
        return len(SHAP_Explainer.get_da_turn_actionLevel_features(instance))

    def get_da_turn_actionLevel_features(instance):
        da_turn = instance['sample'] if DataManager.use_da_turn else [instance['sample']]
        features = []
        for da in da_turn:
            if da == NONE_DA: 
                continue
            for action in da: 
                features.append(str(action))
        return features
    
    def utt_data_mapping(binary_vectors, instance, tokenizer=None, sample_only=False, model_name='roberta-base'): 
        # instance : text
        # binary_vectors : masks for different perturbations
        n_perturbations = len(binary_vectors)
        if sample_only: 
            instances = {'sample': []}
            utterance = instance
        else: 
            instances = {'sample': [], 
                        'input_sequence': [],
                        'sample_dia': [instance['sample_dia']]*n_perturbations, 
                        'label': [instance['label'].item()]*n_perturbations, 
                        'dia_no': [instance['dia_no'].item()]*n_perturbations, 
                        'turn_no': [instance['turn_no'].item()]*n_perturbations}
            utterance = instance['sample']
            if tokenizer is None: 
                tokenizer = DataManager.new_tokenizer(model_name)

        features = SHAP_Explainer.get_utt_features({'sample':utterance}, tokenizer, raw_features=True)
        for v in binary_vectors: 
            text = []
            feature_i = 0
            for feature in features:
                if feature_i >= len(v): 
                    break

                # include the non-feature tokens
                if SHAP_Explainer._token_isNot_feature(feature): 
                    text.append(feature)
                    continue # do not count towards feature

                # Add feature if not masked, else replace with mask token
                if v[feature_i] == 1:
                    text.append(feature)
                else: 
                    text.append(tokenizer.mask_token)
                feature_i += 1
            
            text = tokenizer.convert_tokens_to_string(text).strip()
            instances['sample'].append(text)
            if not sample_only:
                instances['input_sequence'].append(DataManager._utt_input_sequence(text, tokenizer))

        if sample_only: 
            return instances['sample']
        
        perturbed_dataset = DataManager(None, None, None, tokenizer=tokenizer, max_len=DataManager.UTT_TURN_MAX_LEN).initialize_dataset(pd.DataFrame(instances))
        return perturbed_dataset
    
    def utt_n_features(instance, tokenizer):
        return len(SHAP_Explainer.get_utt_features(instance, tokenizer))

    def get_utt_features(instance, tokenizer, raw_features=False):
        if isinstance(instance, str):
            utt = instance
        else: 
            utt = instance['sample']
        utt = DataManager._utt_input_sequence(utt, tokenizer)

        # decompose into features
        # return utt.split()
        features = tokenizer.tokenize(utt, add_special_tokens=False)[1:] # remove CLS token
        if raw_features: 
            return features
        return [f.lstrip('Ġ') for f in features if not SHAP_Explainer._token_isNot_feature(f)] # remove the preceding char for 'empty space', and remove any feature that becomes empty string
    
    # ONLY for (1) combined (single modality/no fusion) dia model, (2) no system da
    def da_utt_data_mapping(binary_vectors, instance, tokenizer=None, model_name='roberta-base', use_da=True, use_utt=True):
        n_perturbations = len(binary_vectors)
        # if sample_only: 
        #     instances = {'sample': []}
        #     utterance = instance
        # else: 
        instances = {'sample': [], 
                    'input_sequence': [],
                    'label': [instance['label'].item()]*n_perturbations, 
                    'dia_no': [instance['dia_no'].item()]*n_perturbations}
        utts = instance['sample'][-1][0][0].split("\n")
        das = instance['sample'][:-1]

        if tokenizer is None: 
            tokenizer = DataManager.new_tokenizer(model_name)

        for v in binary_vectors: 
            feature_i = 0
            mapped_utt = instance['sample'][-1][0][0]
            if use_utt: 
                mapped_utt = []
                for utt_i, n_features in enumerate(n_utt_features): 
                    utterance = utts[utt_i]
                    role = utterance.split()[0]
                    features = SHAP_Explainer.get_utt_features(utterance, tokenizer, raw_features=True)
                    text = []
                    for feature in features:
                        if feature_i >= len(v): 
                            break
                        if SHAP_Explainer._token_isNot_feature(feature): 
                            text.append(feature)
                            continue # do not count towards feature
                        if v[feature_i] == 1:
                            text.append(feature)
                        else: 
                            text.append(tokenizer.mask_token)
                        feature_i += 1
                    text = tokenizer.convert_tokens_to_string(text).strip()
                    text = role + ' ' + text
                    mapped_utt.append(text)
                mapped_utt = '\n'.join(mapped_utt)
            

            mapped_da = das
            if use_da: 
                mapped_da = []
                for da in das:
                    if da == NONE_DA: # none dialogue acts are ignored as features
                        mapped_da.append(NONE_DA)
                        continue
                    masked_da = []
                    for action in da: 
                        if feature_i >= len(v): 
                            break
                        if v[feature_i] == 1: # active action
                            masked_da.append(action)
                        # remove in-active action (not masking them)
                        feature_i += 1
                    if not masked_da: 
                        masked_da = NONE_DA
                    mapped_da.append(masked_da)
            
            # print(masked_da_turn, flush=True)
            instances['sample'].append(mapped_da + [[[mapped_utt, '','','']]])
            instances['input_sequence'].append(DataManager._da_utt_input_sequence([mapped_da, mapped_utt], tokenizer, use_da, use_utt))
            # print(instances['input_sequence'][-1], flush=True)
            # print("", flush=True)

        # for single-pretrained only!!!!
        perturbed_dataset = DataManager(None, None, None, level='dialogue', tokenizer=tokenizer, model_name=model_name, max_len=DataManager.DIA_MAX_LEN).initialize_dataset(pd.DataFrame(instances))
        return perturbed_dataset

    def da_utt_n_features(instance, tokenizer, use_da, use_utt):
        return len(SHAP_Explainer.get_da_utt_features(instance, tokenizer, use_da, use_utt))

    def get_da_utt_features(instance, tokenizer, use_da, use_utt, return_n_features=False): 
        features = []
        n_da_features, n_utt_features = [], []
        if use_utt: 
            utts = instance['sample'][-1][0][0].split("\n")
            utts = instance.split("\n")
            for utt in utts:
                turn_features = SHAP_Explainer.get_utt_features({'sample': utt}, tokenizer)
                features.extend(turn_features)
                n_utt_features.append(len(turn_features))

        if use_da: 
            das = instance['sample'][:-1]
            for da in das:
                turn_features = SHAP_Explainer.get_da_turn_actionLevel_features({'sample':da})
                features.extend(turn_features)
                n_da_features.append(len(turn_features))

        if return_n_features: 
            return features, n_da_features, n_utt_features
        return features
        
            

    def _token_isNot_feature(token): 
        return token == 'Ġ' # just an empty space