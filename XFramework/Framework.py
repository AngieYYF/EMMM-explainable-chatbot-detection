from torch import nn
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig, AutoModel
from transformers import DataCollatorWithPadding
from safetensors import safe_open
import torch
from torch.nn import functional as F
import os
import re
import random
from XFramework.DataManager import DataManager
from XFramework.SHAP_Explainer import SHAP_Explainer
from XFramework.attribution.gradientExplanation import IntegratedGradientExplainer
from XFramework.utils import write_pickle, read_pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
import json



class ModelWrapper:
    def __init__(self, trainer):
        self.trainer = trainer

    def predict(self, dataset):
        # dataset: Huggingface dataset accepted by model
        output = self.trainer.predict(dataset, ).predictions
        # print(output)
        return torch.softmax(torch.tensor(output), dim=1)[:,1] # return AI probability

def get_base_model(model): 
    for attr in ["roberta", "distilbert", "bert", "albert", "base_model"]:
        if hasattr(model, attr):
            return getattr(model, attr)
    raise ValueError("Cannot find base model in classifier wrapper.")

class LateFusionSeparate(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.config = config

        # initialize base model architecture
        self.is_gpt = "gpt" in config._name_or_path.lower()  
        if self.is_gpt: 
            self.model_da = AutoModel.from_config(config) 
            self.model_utt = AutoModel.from_config(config)
        else: 
            self.model_da = AutoModel.from_config(config, add_pooling_layer=False)
            self.model_utt = AutoModel.from_config(config, add_pooling_layer=False)
        
        self.dropout = nn.Dropout(0.1)
        match config.fusion_method: 
            case 'concat': 
                self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)
                self.fusion_method = lambda x, y: torch.cat((x,y), dim=1)
            case 'max':
                self.classifier = nn.Linear(config.hidden_size, config.num_labels)
                self.fusion_method = lambda x, y: torch.max(x, y)
            case 'average': 
                self.classifier = nn.Linear(config.hidden_size, config.num_labels)
                self.fusion_method = lambda x, y: (x + y) / 2

    def forward(self, da_input_ids=None, da_attention_mask=None, utt_input_ids=None, utt_attention_mask=None, labels=None):
        fuse_idx = -1 if self.is_gpt else 0

        # Encode dialogue acts
        da_outputs = self.model_da(
            input_ids=da_input_ids,
            attention_mask=da_attention_mask
        )
        da_cls = da_outputs.last_hidden_state[:, fuse_idx, :]

        # Encode utterances
        utt_outputs = self.model_utt(
            input_ids=utt_input_ids,
            attention_mask=utt_attention_mask
        )
        utt_cls = utt_outputs.last_hidden_state[:, fuse_idx, :]

        # Fuse CLS tokens
        combined = self.fusion_method(da_cls, utt_cls)
        combined = self.dropout(combined)
        
        # Compute logits
        logits = self.classifier(combined)

        # Compute loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return {
            'loss': loss,
            'logits': logits
        }

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings for both `model_da` and `model_utt`."""
        self.model_da.resize_token_embeddings(new_num_tokens)
        self.model_utt.resize_token_embeddings(new_num_tokens)
        return self


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        # Load the base configuration
        config = kwargs.pop("config", None)
        if config is None:
            backbone = kwargs.pop("backbone_model", None)
            config = AutoConfig.from_pretrained(backbone, *model_args, **kwargs)
        if "gpt" in config._name_or_path.lower():
            config.pad_token_id = config.eos_token_id
        # custom configurations
        fusion_method = kwargs.pop("fusion_method", None)
        if fusion_method is not None:
            config.fusion_method = fusion_method
        # Instantiate the model
        model = cls(config)

        # Load from a saved checkpoint (during training)
        safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        if os.path.exists(safetensors_path):
            # adjust vocab size
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
            correct_vocab_size = len(tokenizer)
            model.resize_token_embeddings(correct_vocab_size)

            # load state dictionary
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                state_dict = {key: f.get_tensor(key) for key in f.keys()}

            # Separate state_dict for model_da, model_utt, and classifier
            model_da_state_dict = {}
            model_utt_state_dict = {}
            classifier_state_dict = {}

            for key, value in state_dict.items():
                if key.startswith("model_da."):
                    # Remove 'model_da.' prefix and load into model_da
                    new_key = key.replace("model_da.", "")
                    model_da_state_dict[new_key] = value
                elif key.startswith("model_utt."):
                    # Remove 'model_utt.' prefix and load into model_utt
                    new_key = key.replace("model_utt.", "")
                    model_utt_state_dict[new_key] = value
                    # Remove 'classifier.' prefix and load into classifier
                elif key.startswith("classifier."):
                    new_key = key.replace("classifier.", "")
                    classifier_state_dict[new_key] = value

            # Load state_dict into respective submodules
            model.model_da.load_state_dict(model_da_state_dict, strict=True)
            model.model_utt.load_state_dict(model_utt_state_dict, strict=True)
            model.classifier.load_state_dict(classifier_state_dict, strict=True)

            print("Loaded model from checkpoint:", pretrained_model_name_or_path)
        elif pretrained_model_name_or_path == 'turn_lv_model': 
            da_path = kwargs.pop("turn_lv_da", None)
            utt_path = kwargs.pop("turn_lv_utt", None)

            correct_vocab_size = len(AutoTokenizer.from_pretrained(da_path))

            # load da model
            classifier_model = AutoModelForSequenceClassification.from_pretrained(da_path)
            model.model_da.resize_token_embeddings(correct_vocab_size)
            model.model_da.load_state_dict(state_dict=get_base_model(classifier_model).state_dict(), strict=True)
            
            # load utt model
            classifier_model = AutoModelForSequenceClassification.from_pretrained(utt_path)
            model.model_utt.resize_token_embeddings(correct_vocab_size)
            model.model_utt.load_state_dict(state_dict=get_base_model(classifier_model).state_dict(), strict=True)

            print("Initialized base model from pre-trained turn-lv models:", da_path, utt_path)
        else:
            # Initialize model_da and model_utt with pre-trained weights
            model.model_da = AutoModel.from_pretrained(pretrained_model_name_or_path, config=config)
            model.model_utt = AutoModel.from_pretrained(pretrained_model_name_or_path, config=config)
            print("Initialized model from pre-trained model:", pretrained_model_name_or_path)

        return model

class LateFusionRoberta(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.roberta = AutoModel.from_config(config)
        self.classifier = nn.Linear(2 * config.hidden_size, config.num_labels)
        # self.init_weights()

    def forward(self, da_input_ids=None, da_attention_mask=None, utt_input_ids=None, utt_attention_mask=None, labels=None):
        # Encode dialogue acts
        da_outputs = self.roberta(
            input_ids=da_input_ids,
            attention_mask=da_attention_mask
        )
        da_cls = da_outputs.last_hidden_state[:, 0, :]

        # Encode utterances
        utt_outputs = self.roberta(
            input_ids=utt_input_ids,
            attention_mask=utt_attention_mask
        )
        utt_cls = utt_outputs.last_hidden_state[:, 0, :]

        # Concatenate CLS tokens
        combined = torch.cat((da_cls, utt_cls), dim=1)

        # Compute logits
        logits = self.classifier(combined)

        # Compute loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return {
            'loss': loss,
            'logits': logits
        }


class MultimodalDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # Separate DA and utterance features
        da_features = [
            {"input_ids": f["da_input_ids"], "attention_mask": f["da_attention_mask"]}
            for f in features
        ]
        utt_features = [
            {"input_ids": f["utt_input_ids"], "attention_mask": f["utt_attention_mask"]}
            for f in features
        ]

        # Collate DA and utterance inputs separately
        da_batch = super().__call__(da_features)
        utt_batch = super().__call__(utt_features)

        # Combine batches
        batch = {
            "da_input_ids": da_batch["input_ids"],
            "da_attention_mask": da_batch["attention_mask"],
            "utt_input_ids": utt_batch["input_ids"],
            "utt_attention_mask": utt_batch["attention_mask"],
            "labels": torch.tensor([f["label"] for f in features])
        }
        return batch


def compute_metrics(p):
    predictions, labels = p
    preds = predictions.argmax(axis=-1)  # Get the predicted class
    
    # Calculate Accuracy
    accuracy = accuracy_score(labels, preds)
    
    # Calculate Per-class F1, Precision, and Recall
    f1_per_class = f1_score(labels, preds, average=None).tolist()
    precision_per_class = precision_score(labels, preds, average=None, zero_division=0).tolist()
    recall_per_class = recall_score(labels, preds, average=None, zero_division=0).tolist()

    return {
        "accuracy": accuracy,
        "f1_per_class": f1_per_class,
        "precision_per_class": precision_per_class,
        "recall_per_class": recall_per_class
    }

def model_tokenizer_for_classification(model_name, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = DataManager.new_tokenizer(model_name)
    if "gpt" in model_name.lower():
        model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


class EMMM: 
    FS_PERTURBATIONS = 200
    STII_PERTURBATIONS = 50
    IG_STEPS = 100 

    def __init__(self, train_dataset, val_dataset, test_dataset, random_state, 
                 explanation_methods, model_names = None, use_da_value = True, use_da_turn = True, online = False, evaluate_after_train = True):
        # set DataManager static value - whether to include DA values and use system DA
        DataManager.use_da_value = use_da_value
        DataManager.use_da_turn = use_da_turn

        # dataset
        self.dataset = {'test': {'original': test_dataset},
                        'train': {'original': train_dataset}, 
                        'val': {'original': val_dataset}}
        self.random_state = random_state
        random.seed(self.random_state)
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        DEFAULT_MODEL = 'distilroberta-base'

        # models
        self.turn_lv_da_model = {'model_name': model_names.get('turn_lv_da',DEFAULT_MODEL)}
        self.turn_lv_utt_model = {'model_name': model_names.get('turn_lv_utt',DEFAULT_MODEL)}
        self.dia_lv_model = {'model_name': model_names.get('dia_lv',DEFAULT_MODEL), 
                            'fusion_method': model_names.get('dia_lv_fusion_method','concat'), 
                            'save_suffix': model_names.get('dia_lv_suffix', '')}
        self.save_dir = None

        # explanations
        self.da_explanation = {'method': explanation_methods['da_method'], 'k': explanation_methods['da_k'], 'p': explanation_methods['da_p']}
        self.utt_explanation = {'method': explanation_methods['utt_method'], 'k': explanation_methods['utt_k'], 'p': explanation_methods['utt_p']}
        self.dia_explanation = {'method': explanation_methods['dia_method']}
        self.interaction_order = explanation_methods['interaction_order']
        self.random_explanation = explanation_methods['random_explanation']
        match explanation_methods['explanation_ordering']: 
            case 'absolute': 
                self.explanation_ordering = self.get_top_explanations_absolute
            case 'prediction': 
                self.explanation_ordering = self.get_top_explanations_prediction


        # evaluation
        self.online = online
        self.evaluate_after_train = evaluate_after_train

        print("\nInitialize framework", flush=True)
        print('model_names:', model_names, flush=True)
        print('explanation_methods:', explanation_methods, flush=True)
        print('online detection:', online, flush=True)
        print('use DA value:', use_da_value, flush=True)
        print('use DA turn:', use_da_turn, flush=True)


    def remove_trainer_files(self): 
        # List of checkpoint-specific files to delete
        files_to_delete = ["optimizer.pt", "scheduler.pt", "rng_state.pth", "training_args.bin"]

        for model in [self.turn_lv_da_model, self.turn_lv_utt_model, self.dia_lv_model]: 
            if 'checkpoint_dir' in model:
                checkpoint_dir = EMMM._get_latest_checkpoint(model['checkpoint_dir'])
                if checkpoint_dir is not None: 
                    for file_name in files_to_delete:
                        file_path = os.path.join(checkpoint_dir, file_name)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            print(f"Deleted: {file_path}", flush=True)
            if 'all_checkpoint_dir' in model: 
                for checkpoint_dir in model['all_checkpoint_dir']: 
                    checkpoint_dir = EMMM._get_latest_checkpoint(checkpoint_dir)
                    if checkpoint_dir is not None: 
                        for file_name in files_to_delete:
                            file_path = os.path.join(checkpoint_dir, file_name)
                            if os.path.exists(file_path):
                                os.remove(file_path)
                                print(f"Deleted: {file_path}", flush=True)
            

    def update_daTurn_train_dataset(self, dataset): 
        self.dataset['train']['original'] = dataset
        data_manager = DataManager(self.dataset['train']['original'], None, None, 
                                   model_name=self.turn_lv_da_model['model_name'],
                                   tokenizer = DataManager.new_tokenizer(self.turn_lv_da_model['model_name']), 
                                   dataset_preprocess=DataManager.da_turn, batch_size=16, max_len=DataManager.DA_TURN_MAX_LEN)
        self.dataset['train']['da_turn'] = data_manager.train_dataloader.dataset
        return

    def train_turn_lv_da_model(self, epochs, load_model=False, output_suffix=''): 
        print('\n'+'*'*5, 'training turn level DA-based model', '*'*5+'\n', flush=True)
        output_dir = os.path.join(self.save_dir, 'turn_lv_da/results')
        logging_dir = os.path.join(self.save_dir, 'turn_lv_da/logs')
        self.turn_lv_da_model['checkpoint_dir'] = output_dir
        model_name = self.turn_lv_da_model['model_name']
        
        # Load tokenizer and model
        loaded_suffix_model = False # providing "output_suffix" allows fine-tuning from the model without the suffix (e.g. model trained on NextResponse dataset, then further finetuning on E2E dataset)
        if load_model: 
            checkpoint_path = EMMM._get_latest_checkpoint(self.turn_lv_da_model['checkpoint_dir']+output_suffix)
            if checkpoint_path is None: 
                checkpoint_path = EMMM._get_latest_checkpoint(self.turn_lv_da_model['checkpoint_dir'])
            else:
                loaded_suffix_model = True
            
            if checkpoint_path is not None: 
                print("Loading checkpoint.", flush=True)
                model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
                tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
                # if the full model is loaded, no further training (files to continue training not saved)
                if loaded_suffix_model: 
                    print("NO TRAINING - Full model loaded from checkpoint:",checkpoint_path, flush=True)
                    epochs = 0
                else: 
                    print("TRAINING from Model without suffix:",checkpoint_path, flush=True)
            else: 
                print("No checkpoint found. Initializing from scratch.", flush=True)
                load_model = False
        
        if not load_model: 
            model, tokenizer = model_tokenizer_for_classification(model_name, num_labels=2)
        
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs", flush=True)
        
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(self.device)

        
        self.turn_lv_da_model['checkpoint_dir'] += output_suffix
        if 'all_checkpoint_dir' not in self.turn_lv_da_model: 
            self.turn_lv_da_model['all_checkpoint_dir'] = []
        self.turn_lv_da_model['all_checkpoint_dir'].append(self.turn_lv_da_model['checkpoint_dir'])
        output_dir += output_suffix
        logging_dir += output_suffix

        # Training
        training_args = TrainingArguments(
            output_dir=output_dir,
            logging_dir=logging_dir,
            report_to="none",
            run_name="trial",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=epochs,
            save_total_limit=1, 
            disable_tqdm=False
        )
        print(f"Training for {epochs} epochs", flush=True)


        data_manager = DataManager(self.dataset['train']['original'], self.dataset['val']['original'], self.dataset['test']['original'], 
                                   model_name=model_name,
                                   tokenizer = tokenizer, dataset_preprocess=DataManager.da_turn, batch_size=16, max_len=DataManager.DA_TURN_MAX_LEN)
        self.dataset['train']['da_turn'] = data_manager.train_dataloader.dataset
        self.dataset['val']['da_turn'] = data_manager.val_dataloader.dataset
        self.dataset['test']['da_turn'] = data_manager.test_dataloader.dataset
        # print a sample for validation
        print(tokenizer.decode(data_manager.train_dataloader.dataset['input_ids'][15]), flush=True)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.dataset['train']['da_turn'],
            eval_dataset=self.dataset['val']['da_turn'],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        
        # resume_from_checkpoint will never train since loaded_suffix_model means no training (epoch=0). We always training new models only
        if load_model and os.path.exists(os.path.join(checkpoint_path, 'optimizer.pt')) and loaded_suffix_model: 
            trainer.train(resume_from_checkpoint=checkpoint_path)
        else: 
            trainer.train(resume_from_checkpoint=False)
        
        if self.evaluate_after_train:
            # evaluations on train/val/test sets
            train_predictions, train_metrics = self.predict(trainer, self.dataset['train']['da_turn'], metric_key_prefix='train')
            val_predictions, val_metrics = self.predict(trainer, self.dataset['val']['da_turn'], metric_key_prefix='val')
            test_predictions, test_metrics = self.predict(trainer, self.dataset['test']['da_turn'], metric_key_prefix='test')
            
            # Combine all results
            result = {}
            # ensure accurate #epochs
            if loaded_suffix_model and os.path.exists(os.path.join(checkpoint_path, 'trainer_state.json')): 
                result['epoch'] = json.load(open(os.path.join(checkpoint_path, 'trainer_state.json'),"r"))['epoch']
            result.update(train_metrics)
            result.update(val_metrics)
            result.update(test_metrics)
            result['train_predictions'] = train_predictions
            result['val_predictions'] = val_predictions
            result['test_predictions'] = test_predictions
            with open(os.path.join(self.turn_lv_da_model['checkpoint_dir'], "da_test_evaluation.json"), "w") as f:
                json.dump(result, f, indent=4)
    
    def train_turn_lv_utt_model(self, epochs, load_model=False): 
        print('\n'+'*'*5, 'training turn level utterance-based model', '*'*5+'\n', flush=True)
        output_dir = os.path.join(self.save_dir, 'turn_lv_utt/results')
        logging_dir = os.path.join(self.save_dir, 'turn_lv_utt/logs')
        self.turn_lv_utt_model['checkpoint_dir'] = output_dir
        model_name = self.turn_lv_utt_model['model_name']

        # Load tokenizer and model
        if load_model: 
            checkpoint_path = EMMM._get_latest_checkpoint(self.turn_lv_utt_model['checkpoint_dir'])
            if checkpoint_path is not None: 
                print("Loading checkpoint.", flush=True)
                model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
                tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
                print("NO TRAINING - Model loaded from checkpoint:",checkpoint_path, flush=True)
                epochs = 0
            else: 
                print("No checkpoint found. Initializing from scratch.", flush=True)
                load_model = False
        if not load_model: 
            model, tokenizer = model_tokenizer_for_classification(model_name, num_labels=2)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs", flush=True)
        model.resize_token_embeddings(len(tokenizer))
        model = model.to(self.device)

        # Training
        training_args = TrainingArguments(
            output_dir=output_dir,
            logging_dir=logging_dir,
            report_to="none",
            run_name="trial",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=epochs,
            save_total_limit=1, 
            disable_tqdm=False
        )

        data_manager = DataManager(self.dataset['train']['original'], self.dataset['val']['original'], self.dataset['test']['original'],
                                   model_name=model_name, 
                                   dataset_preprocess=DataManager.utt_only, batch_size=16, max_len=DataManager.UTT_TURN_MAX_LEN)
        self.dataset['train']['utt'] = data_manager.train_dataloader.dataset
        self.dataset['val']['utt'] = data_manager.val_dataloader.dataset
        self.dataset['test']['utt'] = data_manager.test_dataloader.dataset
        
        # print a sample
        print(tokenizer.decode(data_manager.train_dataloader.dataset['input_ids'][15]), flush=True)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.dataset['train']['utt'],
            eval_dataset=self.dataset['val']['utt'],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        if load_model and os.path.exists(os.path.join(checkpoint_path, 'optimizer.pt')): 
            trainer.train(resume_from_checkpoint=checkpoint_path)
        else: 
            trainer.train()
        
        if self.evaluate_after_train:
            # evaluations on train/val/test sets
            train_predictions, train_metrics = self.predict(trainer, self.dataset['train']['utt'], metric_key_prefix='train')
            val_predictions, val_metrics = self.predict(trainer, self.dataset['val']['utt'], metric_key_prefix='val')
            test_predictions, test_metrics = self.predict(trainer, self.dataset['test']['utt'], metric_key_prefix='test')
            
            # Combine all results
            result = {}
            # ensure accurate #epochs
            if epochs==0 and load_model and os.path.exists(os.path.join(checkpoint_path, 'trainer_state.json')): 
                result['epoch'] = json.load(open(os.path.join(checkpoint_path, 'trainer_state.json'),"r"))['epoch']
            result.update(train_metrics)
            result.update(val_metrics)
            result.update(test_metrics)
            result['train_predictions'] = train_predictions
            result['val_predictions'] = val_predictions
            result['test_predictions'] = test_predictions
            with open(os.path.join(self.turn_lv_utt_model['checkpoint_dir'], "utt_test_evaluation.json"), "w") as f:
                json.dump(result, f, indent=4)

    def _get_latest_checkpoint(output_dir):
        # List all checkpoint directories
        if not os.path.exists(output_dir):
            return None
        checkpoints = [d for d in os.listdir(output_dir) if re.match(r"^checkpoint-\d+$", d)]

        if not checkpoints:
            return None  # No checkpoints found

        # Sort checkpoints based on the number after "checkpoint-" (#steps trained)
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('-')[-1]))

        return os.path.join(output_dir, latest_checkpoint)  # Full path to the latest checkpoint

    def get_top_explanations_prediction(self, explanations, predictions, k):
        # top explanations are the ones contributing most to the prediction label
        # Prepare a reduced DataFrame for merging
        merged_df = explanations[['dia_turn_label', 'explanation']].copy()

        # Expand dia_turn_label into separate columns
        merged_df[['dia_no', 'turn_no', 'ground_truth']] = pd.DataFrame(
            merged_df['dia_turn_label'].tolist(), index=merged_df.index
        )

        # Merge with predictions on relevant columns
        merged_df = pd.merge(
            merged_df,
            predictions[['dia_no', 'turn_no', 'ground_truth', 'pred_label']],
            on=['dia_no', 'turn_no', 'ground_truth'],
            how='left'
        )

        # Efficient top-k sorting using list comprehension
        def sort_expl(expl, pred):
            if pred == 1:
                return sorted(expl, key=lambda x: x[1], reverse=True)[:k]
            elif pred == 0:
                return sorted(expl, key=lambda x: x[1])[:k]
            else:
                return expl[:k]

        merged_df['top_explanation'] = [
            sort_expl(expl, pred) for expl, pred in zip(merged_df['explanation'], merged_df['pred_label'])
        ]

        return merged_df['top_explanation']

    def get_top_explanations_absolute(self, explanations, predictions, k):
        # top explanations are the ones with highest absolute attribution score
        return explanations['explanation'].apply(
                    lambda x: sorted(x, key = lambda feature: abs(feature[1]), reverse=True)[:k]
                    )

    def load_detection_model(self, dimension): 
        # load a trained turn-lv detection model
        if dimension == "da": 
            checkpoint_path = EMMM._get_latest_checkpoint(self.turn_lv_da_model['checkpoint_dir'])
            loaded_tokenizer = DataManager.new_tokenizer(self.turn_lv_da_model['model_name'])
        else: 
            checkpoint_path = EMMM._get_latest_checkpoint(self.turn_lv_utt_model['checkpoint_dir'])
            loaded_tokenizer = DataManager.new_tokenizer(self.turn_lv_utt_model['model_name'])
        print(f"Loading {dimension} model from:", checkpoint_path, flush=True)
        loaded_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        loaded_trainer = Trainer(
            model=loaded_model,
            args=TrainingArguments(output_dir=os.path.join(self.save_dir,'temp_trainer_output'), report_to=None, disable_tqdm=True),
            tokenizer=loaded_tokenizer,
            compute_metrics=compute_metrics
        )
        model_wrapper = ModelWrapper(loaded_trainer)
        return loaded_tokenizer, loaded_model, model_wrapper

    def extract_explanation_turn_lv_da_dataset(self, dataset, loaded_model, loaded_tokenizer, model_wrapper, generate_data_path, predictions = None, explanation_df = None, log_save_results=True): 
        # Explanation parameters
        k = self.da_explanation['k']
        p = self.da_explanation['p']
        explanation_method = self.da_explanation['method']

        # extract explanation
        if explanation_df is None: 
            match explanation_method: 
                case 'stii': 
                    explainer = SHAP_Explainer('shapley_taylor', model_wrapper.predict, self.interaction_order, self.random_state, generate_data_path=generate_data_path, log=log_save_results, save_results=log_save_results)
                    explanation = explainer.da_actionLevel_shap_interaction(dataset, n_train_perturbations=EMMM.STII_PERTURBATIONS, tokenizer=loaded_tokenizer, model_name=None)
                case 'ig': 
                    explainer = IntegratedGradientExplainer(loaded_model, loaded_tokenizer, steps=EMMM.IG_STEPS)
                    explanation = explainer.explain_da(dataset, target_class=1)
                case 'fs': 
                    explainer = SHAP_Explainer('faith_shap', model_wrapper.predict, self.interaction_order, self.random_state, generate_data_path=generate_data_path, log=log_save_results, save_results=log_save_results)
                    explanation = explainer.da_actionLevel_shap_interaction(dataset, n_train_perturbations=EMMM.FS_PERTURBATIONS, tokenizer=loaded_tokenizer, model_name=None)
                    # Remove 'bias' features
                    explanation = pd.DataFrame(explanation, columns=['explanation', 'n_features', 'dia_turn_label'])
                    explanation['explanation'] = explanation['explanation'].apply(
                        lambda x: [f for f in x if f[0] != 'bias']
                    )
            explanation_df = pd.DataFrame(explanation, columns=['explanation', 'n_features', 'dia_turn_label'])
       
        # extract top-k explanations
        if self.random_explanation: # randomly extract k explanations
            if log_save_results: print("Random explanation", flush=True)
            explanation_df['top_explanation'] = explanation_df['explanation'].apply(
                lambda x: random.sample(x, k) if len(x) >= k else x
            )
        else: # retrieve top k explanations
            if log_save_results: print("Top explanation", flush=True)
            explanation_df['top_explanation'] = self.explanation_ordering(explanation_df, predictions, k)

        explanation_df['important_features'] = explanation_df['top_explanation'].apply(
            lambda explanation: np.unique([int(feature.split()[-1]) for features, score in explanation for feature in features if feature])
        )
        explanation_df['importance_masks'] = [
            [i in row['important_features'] for i in range(row['n_features'])] for _, row in explanation_df.iterrows()
        ]
        explanation_df['masked_sample'] = [
            SHAP_Explainer.da_turn_actionLevel_data_mapping([mask], sample, sample_only=True, tokenizer=loaded_tokenizer)[0] 
            for mask, sample in zip(explanation_df['importance_masks'], dataset['sample'])
        ]
        return explanation_df


    def extract_explanation_turn_lv_da(self, load_explanation = False):
        print('\n'+'*'*5, 'extracting turn level DA-based model explanation', '*'*5+'\n', flush=True)
        os.makedirs(os.path.join(self.save_dir,'explanation'), exist_ok=True)
        
        # Load predictions
        with open(os.path.join(self.turn_lv_da_model['checkpoint_dir'], "da_test_evaluation.json"), "r") as f:
            predictions = json.load(f)

        # Explanation parameters
        k = self.da_explanation['k']
        p = self.da_explanation['p']
        explanation_method = self.da_explanation['method']
        model_name = self.turn_lv_da_model['model_name']

        # Load existing explanations
        loaded_data_split = []
        if load_explanation: 
            for data_split in self.dataset.keys():
                # Check if the file or directory exists
                explanation_path = os.path.join(self.save_dir,'explanation',f"daTurn_turnLv_{explanation_method}_order{self.interaction_order}_{data_split}")
                self.da_explanation[f'turn_lv_{data_split}'] = explanation_path
                if os.path.exists(explanation_path):
                    print(f"Loading turn level DA-based model explanation: {data_split} set.", flush=True)
                    loaded_data_split.append(data_split)
        
        # Load model to explain
        loaded_tokenizer, loaded_model, model_wrapper = DataManager.new_tokenizer(model_name), None, None
        if len(loaded_data_split) < len(self.dataset): 
            loaded_tokenizer, loaded_model, model_wrapper = self.load_detection_model(dimension="da")

        # Extract explanation
        for data_split, dataset in self.dataset.items():
            dataset = dataset['da_turn']
            self.da_explanation[f'turn_lv_{data_split}'] = os.path.join(self.save_dir,'explanation',f"daTurn_turnLv_{explanation_method}_order{self.interaction_order}_{data_split}")
            
            if explanation_method == 'none': 
                none_explanation = {'dia_turn_label':[], 'masked_sample': []}
                dia_turn_label = list(zip(dataset['dia_no'].tolist(), dataset['turn_no'].tolist(), dataset['label'].tolist()))
                none_explanation = {'dia_turn_label':dia_turn_label, 'masked_sample': dataset['sample']}
                none_explanation = pd.DataFrame(none_explanation)
                write_pickle(none_explanation, self.da_explanation[f'turn_lv_{data_split}'])
                continue

            # Calculate explanation
            explanation_df = None
            if data_split in loaded_data_split: # Load explanation (new calculation of top features is needed)
                explanation_df = read_pickle(self.da_explanation[f'turn_lv_{data_split}'])
            else: 
                print(f'Extracting turn_lv DA turn explanation for {data_split} set', flush=True)
            explanation_df = self.extract_explanation_turn_lv_da_dataset(dataset, 
                    loaded_model, loaded_tokenizer, model_wrapper, 
                    generate_data_path=self.da_explanation[f'turn_lv_{data_split}'], 
                    predictions = pd.DataFrame(predictions[data_split+'_predictions']), 
                    explanation_df = explanation_df, log_save_results=True)
            write_pickle(explanation_df, self.da_explanation[f'turn_lv_{data_split}'])
        return
    
    def extract_explanation_turn_lv_utt_dataset(self, dataset, loaded_model, loaded_tokenizer, model_wrapper, generate_data_path, predictions = None, explanation_df = None, log_save_results=True): 
        # Explanation parameters
        k = self.utt_explanation['k']
        p = self.utt_explanation['p']
        explanation_method = self.utt_explanation['method']

        # extract explanation
        if explanation_df is None: 
            match explanation_method: 
                case 'stii': 
                    explainer = SHAP_Explainer('shapley_taylor', model_wrapper.predict, self.interaction_order, self.random_state, generate_data_path=generate_data_path, log=log_save_results, save_results=log_save_results)
                    explanation = explainer.utt_shap_interaction(dataset, n_train_perturbations=EMMM.STII_PERTURBATIONS, tokenizer=loaded_tokenizer, model_name=None)
                case 'ig': 
                    explainer = IntegratedGradientExplainer(loaded_model, loaded_tokenizer, steps=EMMM.IG_STEPS)
                    explanation = explainer.explain(dataset, target_class=1)
                case 'fs': 
                    explainer = SHAP_Explainer('faith_shap', model_wrapper.predict, self.interaction_order, self.random_state, generate_data_path=generate_data_path, log=log_save_results, save_results=log_save_results)
                    explanation = explainer.utt_shap_interaction(dataset, n_train_perturbations=EMMM.FS_PERTURBATIONS, tokenizer=loaded_tokenizer, model_name=None)
                    # Remove 'bias' features
                    explanation = pd.DataFrame(explanation, columns=['explanation', 'n_features', 'dia_turn_label'])
                    explanation['explanation'] = explanation['explanation'].apply(
                        lambda x: [f for f in x if f[0] != 'bias']
                    )
            explanation_df = pd.DataFrame(explanation, columns=['explanation', 'n_features', 'dia_turn_label'])

        # retrieve top-k explanations
        if self.random_explanation: # randomly extract k explanations
            if log_save_results: print("Random explanation", flush=True)
            explanation_df['top_explanation'] = explanation_df['explanation'].apply(
                lambda x: random.sample(x, k) if len(x) >= k else x
            )
        else: # retrieve top k explanations
            if log_save_results: print("Top explanation", flush=True)
            explanation_df['top_explanation'] = self.explanation_ordering(explanation_df, predictions, k)

        explanation_df['important_features'] = explanation_df['top_explanation'].apply(
            lambda explanation: np.unique([int(feature.split()[-1]) for features, score in explanation for feature in features if feature])
        )
        explanation_df['importance_masks'] = [
            [i in row['important_features'] for i in range(row['n_features'])] for _, row in explanation_df.iterrows()
        ]
        explanation_df['masked_sample'] = [
            SHAP_Explainer.utt_data_mapping([mask], sample, sample_only=True, tokenizer=loaded_tokenizer)[0] 
            for mask, sample in zip(explanation_df['importance_masks'], dataset['sample'])
        ]
        
        return explanation_df

    def extract_explanation_turn_lv_utt(self, load_explanation = False):
        print('\n'+'*'*5, 'extracting turn level utterance-based model explanation', '*'*5+'\n', flush=True)
        os.makedirs(os.path.join(self.save_dir,'explanation'), exist_ok=True)
        
        # Load predictions
        with open(os.path.join(self.turn_lv_utt_model['checkpoint_dir'], "utt_test_evaluation.json"), "r") as f:
            predictions = json.load(f)

        # Explanation parameters
        k = self.utt_explanation['k']
        p = self.utt_explanation['p']
        explanation_method = self.utt_explanation['method']
        model_name = self.turn_lv_utt_model['model_name']

        # Load existing explanations
        loaded_data_split = []
        if load_explanation: 
            for data_split in self.dataset.keys():
                # Check if the file or directory exists
                explanation_path = os.path.join(self.save_dir,'explanation',f"utt_turnLv_{explanation_method}_order{self.interaction_order}_{data_split}")
                self.utt_explanation[f'turn_lv_{data_split}'] = explanation_path
                if os.path.exists(explanation_path):
                    print(f"Loading turn level utterance-based model explanation: {data_split} set.", flush=True)
                    loaded_data_split.append(data_split)
        
        # Load model to be explained
        loaded_tokenizer, loaded_model, model_wrapper = DataManager.new_tokenizer(model_name), None, None
        if len(loaded_data_split) < len(self.dataset): 
            loaded_tokenizer, loaded_model, model_wrapper = self.load_detection_model(dimension="utt")

        for data_split, dataset in self.dataset.items():
            dataset = dataset['utt']
            self.utt_explanation[f'turn_lv_{data_split}'] = os.path.join(self.save_dir,'explanation',f"utt_turnLv_{explanation_method}_order{self.interaction_order}_{data_split}")
            
            if explanation_method == 'none': 
                none_explanation = {'dia_turn_label':[], 'masked_sample': []}
                dia_turn_label = list(zip(dataset['dia_no'].tolist(), dataset['turn_no'].tolist(), dataset['label'].tolist()))
                none_explanation = {'dia_turn_label':dia_turn_label, 'masked_sample': dataset['sample']}
                none_explanation = pd.DataFrame(none_explanation)
                write_pickle(none_explanation, self.utt_explanation[f'turn_lv_{data_split}'])
                continue

            # Calculate explanation
            explanation_df = None
            if data_split in loaded_data_split: # Load explanation (new calculation of top features is needed)
                explanation_df = read_pickle(self.utt_explanation[f'turn_lv_{data_split}'])
            else: 
                print(f'Extracting turn_lv utterance explanation for {data_split} set', flush=True)
            explanation_df = self.extract_explanation_turn_lv_utt_dataset(dataset, 
                    loaded_model, loaded_tokenizer, model_wrapper, 
                    generate_data_path=self.utt_explanation[f'turn_lv_{data_split}'], 
                    predictions = pd.DataFrame(predictions[data_split+'_predictions']), 
                    explanation_df = explanation_df, log_save_results=True)
            write_pickle(explanation_df, self.utt_explanation[f'turn_lv_{data_split}'])
        return

    def extract_explanation_dia_lv(self, load_explanation = False): 
        ## only when dialogue-level prediction relies on a single modality of DA/utterance
        print('\n'+'*'*5, 'extracting dialogue level explanation', '*'*5+'\n', flush=True)
        os.makedirs(os.path.join(self.save_dir,'explanation'), exist_ok=True)
        
        # Explanation parameters
        explanation_method = self.dia_explanation['method']
        model_name = self.dia_lv_model['model_name']

        # Load existing explanations
        loaded_data_split = []
        if load_explanation: 
            for data_split in self.dataset.keys():
                # Check if the file or directory exists
                explanation_path = os.path.join(self.save_dir,'explanation',f"diaLv_{explanation_method}_order{self.interaction_order}_{data_split}")
                self.dia_explanation[data_split] = explanation_path
                if os.path.exists(explanation_path):
                    print(f"Loading dialogue level model explanation: {data_split} set.", flush=True)
                    loaded_data_split.append(data_split)
        
        # Load model to explain
        loaded_model, loaded_tokenizer, data_manager, load_model, checkpoint_path = self._dia_lv_model_dataset(load_model=True)
        if len(loaded_data_split) < len(self.dataset): 
            training_args = TrainingArguments(
                output_dir=os.path.join(self.save_dir,'temp_trainer_output'),
                report_to=None,
                disable_tqdm=True
            )
            loaded_trainer = Trainer(
                        model=loaded_model,
                        args=training_args,
                        tokenizer=loaded_tokenizer,
                        compute_metrics=compute_metrics
                    )
            if 'late-fusion' in self.dia_lv_model['model_name']: 
                loaded_trainer.data_collator=MultimodalDataCollator(tokenizer=loaded_tokenizer)
            model_wrapper = ModelWrapper(loaded_trainer)

        # Extract explanation
        use_da, use_utt = True, True
        if self.dia_lv_model['model_name'] == 'single-pretrained': 
            use_da=self.da_explanation['k']>0
            use_utt=self.utt_explanation['k']>0
        for data_split, dataset in self.dataset.items():
            dataset = dataset['dia_lv']
            dataset = dataset.filter(lambda sample: sample["is_complete"])
            self.dia_explanation[data_split] = os.path.join(self.save_dir,'explanation',f"diaLv_{explanation_method}_order{self.interaction_order}_{data_split}")
            
            if explanation_method == 'none': 
                none_explanation = {'dia_turn_label':[], 'masked_da':[], 'masked_utt':[]}
                dia_turn_label = list(zip(dataset['dia_no'].tolist(), dataset['turn_no'].tolist(), dataset['label'].tolist()))
                none_explanation = {'dia_turn_label':dia_turn_label, 'masked_da': dataset['masked_da'], 'masked_utt': dataset['masked_utt']}
                none_explanation = pd.DataFrame(none_explanation)
                write_pickle(none_explanation, self.dia_explanation[data_split])
                continue


            # Calculate explanation
            if data_split not in loaded_data_split: 
                print(f'Extracting dia_lv explanation for {data_split} set', flush=True)
                match explanation_method: 
                    case 'stii': 
                        explainer = SHAP_Explainer('shapley_taylor', model_wrapper.predict, self.interaction_order, self.random_state, generate_data_path=self.dia_explanation[data_split])
                        explanation = explainer.da_utt_actionLevel_shap_interaction(dataset, n_train_perturbations=EMMM.STII_PERTURBATIONS, tokenizer=loaded_tokenizer, model_name=model_name, use_da=use_da, use_utt=use_utt)
                    case 'fs': 
                        explainer = SHAP_Explainer('faith_shap', model_wrapper.predict, self.interaction_order, self.random_state, generate_data_path=self.dia_explanation[data_split])
                        explanation = explainer.da_utt_actionLevel_shap_interaction(dataset, n_train_perturbations=EMMM.FS_PERTURBATIONS, tokenizer=loaded_tokenizer, model_name=model_name, use_da=use_da, use_utt=use_utt)
                        # Remove 'bias' features
                        explanation = pd.DataFrame(explanation, columns=['explanation', 'n_features', 'dia_turn_label'])
                        explanation['explanation'] = explanation['explanation'].apply(
                            lambda x: [f for f in x if f[0] != 'bias']
                        )
                explanation_df = pd.DataFrame(explanation, columns=['explanation', 'n_features', 'dia_turn_label'])
                write_pickle(explanation_df, self.dia_explanation[data_split])
        return

    def aggregate_dialogue_explanations(self, load_explanation = False): 
        print('\n'+'*'*5, 'aggregating turn level explanations to dialogue level dataset', '*'*5+'\n', flush=True)
        # Load existing explanations
        loaded_data_split = []
        if load_explanation: 
            print(f"!!! Loading existing dialogue level explanations: if turn_lv explanations have been adjusted (k, random_explanation), please set load_explanation=False instead!")
            for data_split in self.dataset.keys():
                # Check if the file or directory exists
                explanation_path = os.path.join(self.save_dir,'explanation',f"diaLv_{data_split}")
                if os.path.exists(explanation_path):
                    print(f"Loading dialogue level masked dataset: {data_split} set.", flush=True)
                    self.dataset[data_split]['dia_lv_original'] = explanation_path
                    loaded_data_split.append(data_split)
        if len(loaded_data_split) == len(self.dataset): 
            return
        
        for data_split in self.dataset.keys():
            if data_split in loaded_data_split: 
                continue
            # da turn
            explanation_df = read_pickle(self.da_explanation[f'turn_lv_{data_split}'])
            dialogue_level_da = {'dia_no': [], 'masked_da':[], 'label':[]}
            for dia_label, i in explanation_df.groupby(explanation_df['dia_turn_label'].apply(lambda x: (x[0], x[2]))):
                sorted_df = i.sort_values(by="dia_turn_label", key=lambda x: x.apply(lambda y: y[1])) # sort by turn_no
                # concatenate the samples over turns
                if DataManager.use_da_turn: 
                    concatenated_over_turns = [utterance_da for turn_da in sorted_df['masked_sample'].values for utterance_da in turn_da]
                    concatenated_over_turns = concatenated_over_turns[1:] # remove the initial assumed system utterance
                else: 
                    concatenated_over_turns = list(sorted_df['masked_sample'].values)
                dialogue_level_da['dia_no'].append(dia_label[0])
                dialogue_level_da['label'].append(dia_label[1])
                dialogue_level_da['masked_da'].append(concatenated_over_turns)
            dialogue_level_da = pd.DataFrame(dialogue_level_da)
            

            # utterances
            explanation_df = read_pickle(self.utt_explanation[f'turn_lv_{data_split}'])
            dialogue_level_utt = {'dia_no': [], 'masked_utt':[], 'label':[]}
            for dia_label, i in explanation_df.groupby(explanation_df['dia_turn_label'].apply(lambda x: (x[0], x[2]))):
                sorted_df = i.sort_values(by="dia_turn_label", key=lambda x: x.apply(lambda y: y[1])) # sort by turn_no
                # concatenate the samples over turns
                concatenated_over_turns = '\n'.join(list(sorted_df['masked_sample']))
                dialogue_level_utt['dia_no'].append(dia_label[0])
                dialogue_level_utt['label'].append(dia_label[1])
                dialogue_level_utt['masked_utt'].append(concatenated_over_turns)
            dialogue_level_utt = pd.DataFrame(dialogue_level_utt)
            
        
            # merge
            merged_df = dialogue_level_da.merge(dialogue_level_utt, on=['dia_no', 'label'], how='inner')
            self.dataset[data_split]['dia_lv_original'] = os.path.join(self.save_dir,'explanation',f"diaLv_{data_split}")
            write_pickle(merged_df, self.dataset[data_split]['dia_lv_original'])
        return
    
    def simulate_online_environment(self): 
        def append_progressing_utterances(dataset_df): 
            '''
            From a dataframe containing full dialogues, extract all progressing dialogues and append to the dataframe.
            Usage: 
                ```
                dataset_df = append_progressing_utterances(dataset_df)
                ```
            '''
            progressing_dialogues = pd.DataFrame({k:[] for k in dataset_df.columns})
            progressing_dialogues['turn_no'] = None
            progressing_dialogues['is_complete'] = None
            for _, row in dataset_df.iterrows(): 
                cur_dia_no = row['dia_no']
                cur_label = row['label']
                cur_dia = row['masked_utt'].split("\n")
                cur_da = row['masked_da']
                n_utterances = len(cur_dia)
                assert (n_utterances*2-1 == len(cur_da)) if DataManager.use_da_turn else (n_utterances == len(cur_da)), f'{n_utterances} utterances, with {len(cur_da)} utterance dialogue acts.'
                for i in range(1,n_utterances+1): 
                    progress_dia = '\n'.join(cur_dia[:i]) # each user utterance
                    # DA turn: user, system, user, ...
                    # DA single: user
                    progress_da = cur_da[:2*i-1] if DataManager.use_da_turn else cur_da[:i]
                    progressing_dialogues.loc[progressing_dialogues.shape[0]] = {
                        'dia_no': cur_dia_no,
                        'label': cur_label, 
                        'masked_utt': progress_dia,
                        'masked_da': progress_da,
                        'turn_no': i-1, 
                        'is_complete': i == n_utterances
                    }
            return progressing_dialogues 

        for data_split in self.dataset.keys(): 
            write_pickle(item = append_progressing_utterances(read_pickle(self.dataset[data_split]['dia_lv_original'])), 
                        file_path = self.dataset[data_split]['dia_lv_original'])
        return

    def _dia_lv_model_dataset(self, load_model=False): 
        checkpoint_path = None
        DEFAULT_BACKBONE = self.turn_lv_da_model['model_name']
        assert self.turn_lv_da_model['model_name'] == self.turn_lv_utt_model['model_name'], "Turn level models should have the same backbone model."

        tokenizer = DataManager.new_tokenizer(DEFAULT_BACKBONE)
        match self.dia_lv_model['model_name']: 
            case 'late-fusion': # same encoder for DA and utterance, then fused
                model =  LateFusionRoberta.from_pretrained(DEFAULT_BACKBONE, num_labels=2)
                if load_model: 
                    checkpoint_path = EMMM._get_latest_checkpoint(self.dia_lv_model['checkpoint_dir'])
                    if checkpoint_path is not None: 
                        print("Loading checkpoint.", flush=True)
                        model = LateFusionRoberta.from_pretrained(checkpoint_path)
                    else: 
                        print("No checkpoint found. Initializing from scratch.", flush=True)
                        load_model = False
                data_manager = DataManager(read_pickle(self.dataset['train']['dia_lv_original']), 
                                read_pickle(self.dataset['val']['dia_lv_original']), 
                                read_pickle(self.dataset['test']['dia_lv_original']), 
                                level='dialogue',
                                model_name=DEFAULT_BACKBONE,
                                dataset_preprocess=DataManager.multimodal_da_utt, 
                                dataset_initialize = 'separate', 
                                batch_size=8, max_len=DataManager.DIA_MAX_LEN)
            case 'late-fusion-separate': # separate encoder for DA and utterance, then fused
                model = LateFusionSeparate.from_pretrained(
                    DEFAULT_BACKBONE, num_labels=2, fusion_method=self.dia_lv_model['fusion_method'], backbone_model=self.turn_lv_da_model['model_name']
                    )
                if load_model: 
                    checkpoint_path = EMMM._get_latest_checkpoint(self.dia_lv_model['checkpoint_dir'])
                    if checkpoint_path is not None: 
                        print("Loading checkpoint.", flush=True)
                        model = LateFusionSeparate.from_pretrained(
                            checkpoint_path, num_labels=2, fusion_method=self.dia_lv_model['fusion_method'], backbone_model=self.turn_lv_da_model['model_name']
                            )
                    else: 
                        print("No checkpoint found. Initializing from scratch.", flush=True)
                        load_model = False
                data_manager = DataManager(read_pickle(self.dataset['train']['dia_lv_original']), 
                                read_pickle(self.dataset['val']['dia_lv_original']), 
                                read_pickle(self.dataset['test']['dia_lv_original']), 
                                level='dialogue',
                                model_name=DEFAULT_BACKBONE,
                                dataset_preprocess=DataManager.multimodal_da_utt, 
                                dataset_initialize = 'separate', 
                                batch_size=8, max_len=DataManager.DIA_MAX_LEN)
                print(tokenizer.decode(data_manager.train_dataloader.dataset['da_input_ids'][15]), flush=True)
                print(tokenizer.decode(data_manager.train_dataloader.dataset['utt_input_ids'][15]), flush=True)
            case 'late-fusion-separate-pretrained': # separate encoder (turn-lv detection model) for DA and utterance, then fused
                print('checkpoint for da', EMMM._get_latest_checkpoint(self.turn_lv_da_model['checkpoint_dir']), flush=True)
                print('checkpoint for utt', EMMM._get_latest_checkpoint(self.turn_lv_utt_model['checkpoint_dir']), flush=True)
                model = LateFusionSeparate.from_pretrained(
                    "turn_lv_model", num_labels=2, fusion_method=self.dia_lv_model['fusion_method'], 
                    turn_lv_da=EMMM._get_latest_checkpoint(self.turn_lv_da_model['checkpoint_dir']), 
                    turn_lv_utt=EMMM._get_latest_checkpoint(self.turn_lv_utt_model['checkpoint_dir']),
                    backbone_model=self.turn_lv_da_model['model_name']
                    )
                if load_model: 
                    checkpoint_path = EMMM._get_latest_checkpoint(self.dia_lv_model['checkpoint_dir'])
                    if checkpoint_path is not None: 
                        print("Loading checkpoint.", flush=True)
                        model = LateFusionSeparate.from_pretrained(
                            checkpoint_path, num_labels=2, fusion_method=self.dia_lv_model['fusion_method'], backbone_model=self.turn_lv_da_model['model_name']
                            )
                    else: 
                        print("No checkpoint found. Initializing from scratch.", flush=True)
                        load_model = False
                data_manager = DataManager(read_pickle(self.dataset['train']['dia_lv_original']), 
                                read_pickle(self.dataset['val']['dia_lv_original']), 
                                read_pickle(self.dataset['test']['dia_lv_original']), 
                                level='dialogue',
                                model_name=DEFAULT_BACKBONE,
                                dataset_preprocess=DataManager.multimodal_da_utt, 
                                dataset_initialize = 'separate', 
                                batch_size=8, max_len=DataManager.DIA_MAX_LEN)
                print(tokenizer.decode(data_manager.train_dataloader.dataset['da_input_ids'][15]), flush=True)
                print(tokenizer.decode(data_manager.train_dataloader.dataset['utt_input_ids'][15]), flush=True)
            case 'single-pretrained': # only utterance or only da - using pre-trained turn-lv detection model
                if self.utt_explanation['k']>0: # only utt, or both utt and da
                    print("Initialized base model from pre-trained turn-lv utterance-based model")
                    model = AutoModelForSequenceClassification.from_pretrained(EMMM._get_latest_checkpoint(self.turn_lv_utt_model['checkpoint_dir']), num_labels=2)
                else: # only da
                    print("Initialized base model from pre-trained turn-lv da-based model")
                    model = AutoModelForSequenceClassification.from_pretrained(EMMM._get_latest_checkpoint(self.turn_lv_da_model['checkpoint_dir']), num_labels=2)
                if load_model: 
                    checkpoint_path = EMMM._get_latest_checkpoint(self.dia_lv_model['checkpoint_dir'])
                    if checkpoint_path is not None: 
                        print("Loading checkpoint.", flush=True)
                        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
                    else: 
                        print("No checkpoint found. Initializing from scratch.", flush=True)
                        load_model = False
                data_manager = DataManager(read_pickle(self.dataset['train']['dia_lv_original']), 
                                read_pickle(self.dataset['val']['dia_lv_original']), 
                                read_pickle(self.dataset['test']['dia_lv_original']), 
                                level='dialogue',
                                model_name=DEFAULT_BACKBONE,
                                dataset_preprocess=lambda d,t,r: DataManager.da_utt(d,t,r,da=self.da_explanation['k']>0, utt=self.utt_explanation['k']>0), 
                                batch_size=8, max_len=DataManager.DIA_MAX_LEN)
                print(data_manager.train_dataloader.dataset['label'][15], data_manager.train_dataloader.dataset['dia_no'][15], tokenizer.decode(data_manager.train_dataloader.dataset['input_ids'][15]), flush=True)
                print(data_manager.train_dataloader.dataset['label'][16], data_manager.train_dataloader.dataset['dia_no'][16], tokenizer.decode(data_manager.train_dataloader.dataset['input_ids'][16]), flush=True)
            case model_name: 
                model, _ = model_tokenizer_for_classification(model_name, num_labels=2)
                if load_model: 
                    checkpoint_path = EMMM._get_latest_checkpoint(self.dia_lv_model['checkpoint_dir'])
                    if checkpoint_path is not None: 
                        print("Loading checkpoint.", flush=True)
                        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
                    else: 
                        print("No checkpoint found. Initializing from scratch.", flush=True)
                        load_model = False
                data_manager = DataManager(read_pickle(self.dataset['train']['dia_lv_original']), 
                                read_pickle(self.dataset['val']['dia_lv_original']), 
                                read_pickle(self.dataset['test']['dia_lv_original']), 
                                level='dialogue',
                                model_name=model_name,
                                dataset_preprocess=lambda d,t,r: DataManager.da_utt(d,t,r,da=self.da_explanation['k']>0, utt=self.utt_explanation['k']>0), 
                                batch_size=8, max_len=DataManager.DIA_MAX_LEN)
        model.resize_token_embeddings(len(tokenizer))
        return model, tokenizer, data_manager, load_model, checkpoint_path
    

    def assign_turn_lv_model_dir(self): 
        self.turn_lv_da_model['checkpoint_dir'] = os.path.join(self.save_dir, 'turn_lv_da', 'results')
        self.turn_lv_utt_model['checkpoint_dir'] = os.path.join(self.save_dir, 'turn_lv_utt', 'results')

    def assign_dia_lv_model_dir(self): 
        # model directory
        model_dir = self.dia_lv_model['model_name']+'_'
        # if no explanation extracted -> interactionOrder does not matter
        if not (self.da_explanation['method'] == 'none' and self.utt_explanation['method'] == 'none'): 
            model_dir += 'interactionOrder'+str(self.interaction_order)+'_'
        model_dir += self.da_explanation['method']+str(self.da_explanation['k'])+'da_'+\
                    self.utt_explanation['method']+str(self.utt_explanation['k'])+'token_'+\
                    str(self.random_explanation)+'Random'+\
                    self.dia_lv_model['save_suffix']
        if 'late-fusion-separate' in self.dia_lv_model['model_name']: 
            model_dir = self.dia_lv_model['fusion_method']+'-'+model_dir
        if self.online: 
            model_dir = 'online_' + model_dir
        output_dir = os.path.join(self.save_dir, 'dia_lv/results', model_dir)
        logging_dir = os.path.join(self.save_dir, 'dia_lv/logs', model_dir)

        self.dia_lv_model['checkpoint_dir'] = output_dir
        return output_dir, logging_dir

    def train_dia_lv_model(self, epochs, load_model=False): 
        print('\n'+'*'*5, 'training dialogue level model', '*'*5+'\n', flush=True)
        output_dir, logging_dir = self.assign_dia_lv_model_dir()

        # Load tokenizer and model
        model, tokenizer, data_manager, load_model, checkpoint_path = self._dia_lv_model_dataset(load_model)
        if load_model and checkpoint_path is not None: # no resume training.
            epochs=0
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs", flush=True)
        model = model.to(self.device)
        
        # Training
        training_args = TrainingArguments(
            output_dir=output_dir,
            logging_dir=logging_dir,
            report_to="none",
            run_name="trial",
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=epochs,
            save_total_limit=1, 
            disable_tqdm=False
        )

        self.dataset['train']['dia_lv'] = data_manager.train_dataloader.dataset
        self.dataset['val']['dia_lv'] = data_manager.val_dataloader.dataset
        self.dataset['test']['dia_lv'] = data_manager.test_dataloader.dataset
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.dataset['train']['dia_lv'],
            eval_dataset=self.dataset['val']['dia_lv'],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        if 'late-fusion' in self.dia_lv_model['model_name']:
            trainer.data_collator=MultimodalDataCollator(tokenizer=tokenizer)
            
        if load_model and os.path.exists(os.path.join(checkpoint_path, 'optimizer.pt')): 
            trainer.train(resume_from_checkpoint=checkpoint_path)
        else: 
            trainer.train()
        
        return trainer

    def train(self, save_dir, epochs, load_existing=dict()): 
        self.save_dir = save_dir
        self.train_turn_lv_da_model(epochs[0], load_model=load_existing.get('train_turn_lv_da_model', True))
        self.train_turn_lv_utt_model(epochs[1], load_model=load_existing.get('train_turn_lv_utt_model', True))
        self.extract_explanation_turn_lv_da(load_explanation=load_existing.get('extract_explanation_turn_lv_da', True)) # load generated explanation, but important features are selected based on setting each time
        self.extract_explanation_turn_lv_utt(load_explanation=load_existing.get('extract_explanation_turn_lv_utt', True)) # load generated explanation, but important features are selected based on setting each time
        self.aggregate_dialogue_explanations(load_explanation=load_existing.get('aggregate_dialogue_explanations', False)) # load=False, to be updated based on the selected features each time
        if self.online: 
            self.simulate_online_environment()
        dia_trainer = self.train_dia_lv_model(epochs[2], load_model=load_existing.get('train_dia_lv_model', True))
        return dia_trainer
    
    def load_dia_lv_model(self): 
        if 'model_dir' in self.dia_lv_model: # saving model directly
            checkpoint_path = self.dia_lv_model['model_dir']
        else: # saving checkpoint during training
            checkpoint_path = EMMM._get_latest_checkpoint(self.dia_lv_model['checkpoint_dir'])
        if checkpoint_path is None: 
            print('Dia-lv detection model do not exist.\n', flush=True)
            return None

        match self.dia_lv_model['model_name']:
            case 'late-fusion': 
                model = LateFusionRoberta.from_pretrained(checkpoint_path)
            case 'late-fusion-separate': 
                model = LateFusionSeparate.from_pretrained(checkpoint_path, num_labels=2, fusion_method=self.dia_lv_model['fusion_method'], backbone_model=self.turn_lv_da_model['model_name'])
            case 'late-fusion-separate-pretrained': 
                model = LateFusionSeparate.from_pretrained(checkpoint_path, num_labels=2, fusion_method=self.dia_lv_model['fusion_method'], backbone_model=self.turn_lv_da_model['model_name'])
            case _: 
                model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        model.resize_token_embeddings(len(tokenizer))
        
        training_args = TrainingArguments(
            output_dir=os.path.join(self.save_dir,'temp_trainer_output'),
            report_to=None,
            disable_tqdm=True
        )

        trainer = Trainer(
                    model=model,
                    args=training_args,
                    tokenizer=tokenizer,
                    compute_metrics=compute_metrics
                )
        if 'late-fusion' in self.dia_lv_model['model_name']: 
            trainer.data_collator=MultimodalDataCollator(tokenizer=tokenizer)
        return trainer

    def evaluate(self): 
        print('\n'+'*'*5, 'evaluating validation/test set', '*'*5+'\n', flush=True)
        # load model
        trainer = self.load_dia_lv_model()
        
        # evaluate
        max_n_utterances = 10
        test_evaluations = dict()
        val_evaluations = dict()
        if self.online: 
            # online evaluation - detection on progressing dialogues with n turns
            for n in range(max_n_utterances): 
                print(f"Online detection - first {n+1} turns.")
                # test set
                online_dataset = self.dataset['test']['dia_lv'].filter(lambda x: x['turn_no'] == n)
                trainer.eval_dataset = online_dataset
                metrics = trainer.evaluate()
                n_samples = len(online_dataset)
                test_evaluations[n] = {
                    'n_samples':n_samples, 
                    'metrics': metrics
                }
                # validation set
                online_dataset = self.dataset['val']['dia_lv'].filter(lambda x: x['turn_no'] == n)
                trainer.eval_dataset = online_dataset
                metrics = trainer.evaluate()
                n_samples = len(online_dataset)
                val_evaluations[n] = {
                    'n_samples': n_samples, 
                    'metrics': metrics
                }
            # offline evaluation - detection on complete dialogues
            # test set
            offline_dataset = self.dataset['test']['dia_lv'].filter(lambda x: x['is_complete']==True) 
            predictions, metrics = self.predict(trainer, offline_dataset, metric_key_prefix='test')
            n_samples = len(offline_dataset)
            test_evaluations['offline'] = {
                'n_samples': n_samples, 
                'metrics': metrics, 
                'predictions': predictions
            }
            # validation set
            offline_dataset = self.dataset['val']['dia_lv'].filter(lambda x: x['is_complete']==True) 
            predictions, metrics = self.predict(trainer, offline_dataset, metric_key_prefix='val')
            n_samples = len(offline_dataset)
            val_evaluations['offline'] = {
                'n_samples': n_samples, 
                'metrics': metrics, 
                'predictions': predictions
            }
        else: 
            predictions, metrics = self.predict(trainer, self.dataset['test']['dia_lv'], metric_key_prefix='test')
            test_evaluations = {'offline': {
                                'n_samples': len(self.dataset['test']['dia_lv']),
                                'metrics': metrics,
                                'predictions': predictions}}
            predictions, metrics = self.predict(trainer, self.dataset['val']['dia_lv'], metric_key_prefix='val')
            val_evaluations = {'offline': {
                                'n_samples': len(self.dataset['val']['dia_lv']),
                                'metrics': metrics, 
                                'predictions': predictions}}


        # record average proportion of data used in turn-level
        da_proportions = {}
        utt_proportions = {}
        for data_split in self.dataset.keys():
            if self.da_explanation['method'] == 'none': 
                da_proportions[data_split] = 1
            else: 
                feature_df = read_pickle(self.da_explanation[f'turn_lv_{data_split}'])
                da_proportions[data_split] = np.mean(feature_df['important_features'].apply(lambda x:len(x))/feature_df['n_features'] )
            if self.utt_explanation['method'] == 'none': 
                utt_proportions[data_split] = 1
            else: 
                feature_df = read_pickle(self.utt_explanation[f'turn_lv_{data_split}'])
                utt_proportions[data_split] = np.mean(feature_df['important_features'].apply(lambda x:len(x))/feature_df['n_features'] )

        result = {
            'da_proportion': da_proportions, 
            'utt_proportion': utt_proportions, 
            'test_evaluation': test_evaluations,
            'val_evaluation': val_evaluations
        }

        # write to json file
        with open(os.path.join(self.dia_lv_model['checkpoint_dir'], "test_evaluation.json"), "w") as f:
            json.dump(result, f, indent=4)  

        if not DataManager.use_da_turn and self.dia_lv_model['model_name'] == "single-pretrained" and self.dia_explanation['method'] != "none": 
            self.extract_explanation_dia_lv(True)

        return test_evaluations
    
    def predict(self, trainer, dataset, metric_key_prefix='test'): 
        # perform prediction on a dataset
        predictions_output = trainer.predict(dataset, metric_key_prefix=metric_key_prefix)
        logits = predictions_output.predictions
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        pred_labels = probs.argmax(axis=-1)

        results = []
        for i, (logit, prob, pred, label) in enumerate(zip(logits, probs, pred_labels, predictions_output.label_ids)):
            result = {
                "pred_logit": logit.tolist(),
                "pred_prob": prob.tolist(),
                "pred_label": int(pred),
                "ground_truth": int(label),
                "dia_no": int(dataset['dia_no'][i]),
            }
            if 'turn_no' in dataset.features:
                result['turn_no'] = int(dataset['turn_no'][i])
            results.append(result)

        return results, predictions_output.metrics