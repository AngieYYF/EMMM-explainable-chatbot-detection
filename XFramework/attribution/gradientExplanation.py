import torch
import numpy as np
from tqdm import tqdm

class IntegratedGradientExplainer:
    def __init__(self, model, tokenizer, baseline_token=None, steps=5):
        self.model = model
        self.tokenizer = tokenizer
        self.steps = steps
        self.baseline_token = baseline_token if baseline_token else self.tokenizer.pad_token

    def interpolate_embeddings(self, input_embeds, baseline_embeds):
        alphas = torch.linspace(0, 1, self.steps+1).view(-1, 1, 1).to(input_embeds.device)
        return baseline_embeds + alphas * (input_embeds - baseline_embeds)

    def explain_instance(self, input_ids, attention_mask, target_class=None):
        # Move tensors to the model's device
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        # Get input embeddings
        input_embeds = self.model.get_input_embeddings()(input_ids.unsqueeze(0))

        # Create baseline embeddings (same length as input_ids)
        baseline_token_id = self.tokenizer.convert_tokens_to_ids(self.baseline_token)
        baseline_ids = torch.tensor(
            [self.tokenizer.cls_token_id] + [baseline_token_id] * (input_ids.shape[0] - 1),
            dtype=torch.long,
            device=input_ids.device
        )
        baseline_embeds = self.model.get_input_embeddings()(baseline_ids.unsqueeze(0))

        # Generate interpolated embeddings
        interpolated_embeds = self.interpolate_embeddings(input_embeds, baseline_embeds)

        step_batch_size = 16 
        all_gradients = []
        for i in range(0, interpolated_embeds.size(0), step_batch_size):
            batch = interpolated_embeds[i:i+step_batch_size].clone().detach().requires_grad_()
            # Forward pass
            outputs = self.model(inputs_embeds=batch, attention_mask=attention_mask.unsqueeze(0).expand(batch.size(0), -1))
            # Get logits for target class
            target_logits = outputs.logits[:, target_class].sum()
            # Backward pass
            grads = torch.autograd.grad(target_logits, batch)[0].detach()
            all_gradients.append(grads)
        all_gradients = torch.cat(all_gradients, dim=0)   # shape: [steps+1, 1, seq_len, embed_dim]
        # Apply trapezoidal rule: average consecutive gradients
        avg_consecutive_grads = (all_gradients[:-1] + all_gradients[1:]) / 2.0

        # Compute the mean across all steps
        avg_gradients = avg_consecutive_grads.mean(dim=0)  # shape: [1, seq_len, embed_dim]

        # Compute attributions
        attributions = (input_embeds - baseline_embeds) * avg_gradients


        token_importance = attributions.sum(dim=-1).squeeze().detach().cpu().numpy()
        # retrieve token-wise attribution
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        word_importance = []
        word_idx = 0
        for tok, imp in zip(tokens, token_importance):
            if tok in self.tokenizer.all_special_tokens: 
                continue
            tok = tok.lstrip('Ġ')
            if tok: 
                word_importance.append(((f'{tok} {word_idx}',), imp))
                word_idx += 1

        return word_importance, word_idx
    
    def explain(self, dataset, target_class=None): 
        self.model.eval()
        dia_turn_label = list(zip(dataset['dia_no'].tolist(), dataset['turn_no'].tolist(), dataset['label'].tolist()))
        explanation = [self.explain_instance(input_ids, attention_mask, target_class) for input_ids, attention_mask in tqdm(zip(dataset['input_ids'], dataset['attention_mask']), total = dataset.num_rows)]
        return {'explanation': [exp[0] for exp in explanation], 
                'n_features': [exp[1] for exp in explanation], 
                'dia_turn_label':dia_turn_label}
    

    def explain_instance_da(self, input_ids, attention_mask, target_class=None):
        # Move tensors to the model's device
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)

        # Get input embeddings
        input_embeds = self.model.get_input_embeddings()(input_ids.unsqueeze(0))

        # Create baseline embeddings (same length as input_ids)
        baseline_token_id = self.tokenizer.convert_tokens_to_ids(self.baseline_token)
        baseline_ids = torch.tensor(
            [self.tokenizer.cls_token_id] + [baseline_token_id] * (input_ids.shape[0] - 1),
            dtype=torch.long,
            device=input_ids.device
        )
        baseline_embeds = self.model.get_input_embeddings()(baseline_ids.unsqueeze(0))

        # Generate interpolated embeddings
        interpolated_embeds = self.interpolate_embeddings(input_embeds, baseline_embeds)

        step_batch_size = 16  # adjust based on available memory
        all_gradients = []
        for i in range(0, interpolated_embeds.size(0), step_batch_size):
            batch = interpolated_embeds[i:i+step_batch_size].clone().detach().requires_grad_()
            
            # Forward pass
            outputs = self.model(inputs_embeds=batch, attention_mask=attention_mask.unsqueeze(0).expand(batch.size(0), -1))
            
            # Get logits for target class
            target_logits = outputs.logits[:, target_class].sum()
            
            # Backward pass
            grads = torch.autograd.grad(target_logits, batch)[0].detach()
            all_gradients.append(grads)

        # Convert to tensor for easier manipulation
        all_gradients = torch.cat(all_gradients, dim=0)   # shape: [steps+1, 1, seq_len, embed_dim]


        # Apply trapezoidal rule: average consecutive gradients
        avg_consecutive_grads = (all_gradients[:-1] + all_gradients[1:]) / 2.0

        # Compute the mean across all steps
        avg_gradients = avg_consecutive_grads.mean(dim=0)  # shape: [1, seq_len, embed_dim]

        # Compute attributions
        attributions = (input_embeds - baseline_embeds) * avg_gradients

        token_importance = attributions.sum(dim=-1).squeeze().detach().cpu().numpy()

        # Retrieve token-wise attribution
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        filtered_tokens = []
        filtered_importance = []
        for tok, imp in zip(tokens, token_importance):
            if tok in [self.tokenizer.mask_token, self.tokenizer.pad_token, self.tokenizer.cls_token]: 
                continue
            filtered_tokens.append(tok)
            filtered_importance.append(imp)
        # Retrieve action-wise attribution
        action_importance = []
        current_action = ["","","",""]
        current_importance = 0.0
        feature_idx = 0
        slot_idx = 0
        SLOT_SEP = [self.tokenizer.special_tokens_map['additional_special_tokens'][1]]
        DA_SEP = [self.tokenizer.special_tokens_map['additional_special_tokens'][0], self.tokenizer.sep_token, self.tokenizer.cls_token]

        for tok, imp in zip(filtered_tokens, filtered_importance): 
            if tok in DA_SEP: 
                action_importance.append(((str(current_action) + ' ' + str(feature_idx),), current_importance))
                current_action = ["", "", "", ""]
                slot_idx = 0
                feature_idx += 1
                current_importance = 0.0
            elif tok in SLOT_SEP: 
                slot_idx += 1
                current_importance += imp
            elif tok.startswith("Ġ"):
                current_action[slot_idx] += tok[1:]
                current_importance += imp
            else: 
                current_action[slot_idx] += ' ' + tok
                current_importance += imp
                
        if current_action[0]: 
            action_importance.append(((str(current_action) + ' ' + str(feature_idx),), current_importance))
            feature_idx += 1

        return action_importance, feature_idx
    
    def explain_da(self, dataset, target_class=None): 
        self.model.eval()
        dia_turn_label = list(zip(dataset['dia_no'].tolist(), dataset['turn_no'].tolist(), dataset['label'].tolist()))
        explanation = [self.explain_instance_da(input_ids, attention_mask, target_class) for input_ids, attention_mask in tqdm(zip(dataset['input_ids'], dataset['attention_mask']), total = dataset.num_rows)]
        return {'explanation': [exp[0] for exp in explanation], 
                'n_features': [exp[1] for exp in explanation], 
                'dia_turn_label':dia_turn_label}