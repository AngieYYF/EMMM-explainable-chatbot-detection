import torch
import torch.nn.functional as F
from sentence_transformers import InputExample
from torch.utils.data import DataLoader
import random
from sentence_transformers import SentenceTransformer, losses
import numpy as np
import pandas as pd

def get_utt_feature(text, expl_tokenizer): 
    features = expl_tokenizer.tokenize(' '+text)
    features = [f.lstrip('Ġ') for f in features if f!='Ġ']
    return features

# train feature matching embedding model
def train_feature_matching_embedding_model(dataset, expl_tokenizer, ontology=None, save_path=None): 
    def get_utt_da(dataset):
        # extract utterance-lv information
        utt_da = {'utt': [], 'da':[]}
        for _, row in dataset.iterrows(): 
            dia = row['dia']
            dia_da = row['dialogue_act_info_removed']
            for utt, da in zip(dia.split('\n'), dia_da):
                utt_da['utt'].append(utt.strip("user: ").strip("system: ").strip())
                utt_da['da'].append(da)
        return pd.DataFrame(utt_da)
        
    def get_train_input(utt, das, label, ontology=None, ignore_value=False):
        if ignore_value: 
            das = [da_to_text(da[:-1], ontology) for da in das]
        else:
            das = [da_to_text(da, ontology) for da in das]
        return InputExample(texts=[utt, ';'.join(das)], label=float(label))

    def create_negative_samples(df, ontology=None, ignore_value=False):
        negative_examples = []
        all_das = df['da'].tolist()
        for idx, row in df.iterrows():
            # Select a random DA from another row
            random_da = random.choice([d for i, d in enumerate(all_das) if i != idx])
            negative_example = get_train_input(row['utt'], random_da, 0, ontology, ignore_value)
            negative_examples.append(negative_example)
        return negative_examples

    # prepare training data - positive and negative samples 
    # samples are with and without DA value to encourage understanding of the act intents and slot
    dataset_utt_da = get_utt_da(dataset)
    train_examples = list(dataset_utt_da.apply(lambda x: get_train_input(x['utt'], x['da'], 1, ontology), axis=1))
    train_examples += list(dataset_utt_da.apply(lambda x: get_train_input(x['utt'], x['da'], 1, ontology, ignore_value=True), axis=1))
    train_examples += create_negative_samples(dataset_utt_da, ontology)
    train_examples += create_negative_samples(dataset_utt_da, ontology, ignore_value=True)
    # tokenize samples by the same tokenizer used by the detection model
    processed_examples = [] 
    for ex in train_examples:
        text1_split = " ".join(get_utt_feature(ex.texts[0], expl_tokenizer))
        text2_split = " ".join(get_utt_feature(ex.texts[1], expl_tokenizer))
        processed_examples.append(InputExample(texts=[text1_split, text2_split], label=ex.label))

    
    # Load pretrained model
    model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

    # Define loss (cosine similarity for contrastive learning)
    train_loss = losses.CosineSimilarityLoss(model)

    # Train the model
    num_epochs = 3
    train_dataloader = DataLoader(processed_examples, shuffle=True, batch_size=16)
    warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)

    model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=num_epochs,
            warmup_steps=warmup_steps,
            show_progress_bar=True,
            output_path=save_path)
    return model
    


# retrieve embedding
# embedding
def retrieve_word_embedding(text, get_utt_feature, embed_tokenizer, model, avg_word_embedding): 
    # Tokenize and keep track of word-piece alignment
    words = get_utt_feature(text.strip()) # text.split()
    encoding = embed_tokenizer(words, is_split_into_words=True, return_tensors='pt', return_attention_mask=True)

    # Get embedding
    model.eval()
    with torch.no_grad():
        outputs = model(**encoding)
        last_hidden_state = outputs.last_hidden_state  # shape: (1, seq_len, hidden_dim)

    # Align tokens to words
    word_ids = encoding.word_ids()  # list of word indices corresponding to each token

    # Aggregate token embeddings per word
    word_embeddings = []
    for i in range(len(words)):
        token_indices = [j for j, word_idx in enumerate(word_ids) if word_idx == i]
        token_embeds = last_hidden_state[0, token_indices, :]
        if avg_word_embedding: 
            word_embeddings.append([token_embeds.mean(dim=0)]) # mean over subword tokens
        else: 
            word_embeddings.append([token_embed for token_embed in token_embeds])  # keep all subword embeddings

    return words, word_embeddings

def retrieve_da_embedding(das, get_utt_feature, embed_tokenizer, model, avg_word_embedding, ontology=None): 

    das_embeddings = []
    for da in das: 
        da_embeddings = [token_embed for word_embed in retrieve_word_embedding(da_to_text(da, ontology), get_utt_feature, embed_tokenizer, model, avg_word_embedding=avg_word_embedding)[1] for token_embed in word_embed]
        das_embeddings.append(da_embeddings)
    return das, das_embeddings
    

def tokenwise_pairwise_cosine_similarity(list_a, list_b):
    similarity_matrix = torch.zeros((len(list_a), len(list_b)))

    for i, a_embeds in enumerate(list_a): # a_embeds: list of tensors (subtokens in a word/DA feature)
        tensor_a = torch.stack(a_embeds)  # (n, d)
        tensor_a = F.normalize(tensor_a, p=2, dim=1).T  # (d, n)
        for j, b_embeds in enumerate(list_b):  # b_embeds: list of tensors  (subtokens in a word/DA feature)
            if len(b_embeds) == 0:
                similarity_matrix[i, j] = float('-inf')  # or some default low value
                continue

            tensor_b = torch.stack(b_embeds)  # (m, d)
            tensor_b = F.normalize(tensor_b, p=2, dim=1)  # (m, d)

            # similarity score between two features = maximum pairwise cosine similarity between the subtokens of the two features
            sim_scores = torch.matmul(tensor_b, tensor_a)  # (m,n)
            max_sim = torch.max(sim_scores) 
            similarity_matrix[i, j] = max_sim
    
    return similarity_matrix


def similarity_matrix_to_df(sim_matrix, a_texts, b_texts, round_digits=4):
    # Convert sim_matrix to numpy and round for readability
    sim_np = sim_matrix.detach().cpu().numpy().round(round_digits)

    # Create DataFrame
    df = pd.DataFrame(sim_np, index=a_texts, columns=b_texts)
    
    return df


intent_to_description = {"nobook": "booking is failed", 
                         "reqmore": "ask for more instructions"}

def da_to_text(da, ontology=None): 
    da_cp = da.copy()
    
    # if da has a slot which exits in ontology, add ontology description
    if da_cp[2] and ontology is not None and da_cp[2] in ontology: 
        da_cp[2] = da_cp[2] + ' (' + ontology[da_cp[2]]['description'] + ')'

    # (optional) remove the (domain) -> intent, slot, value
    da_cp = da_cp[:1] + da_cp[2:]

    # descriptive intent
    if da_cp[0] in intent_to_description: 
        da_cp[0] = intent_to_description[da_cp[0]]
    
    return ' '.join(da_cp).replace('  ', ' ').strip()


def threshold_similarities(sim_matrix, threshold = 0.9, col_axis=True): 
    similar_indices = {}
    # if col_axis is True, find top words per DA feature
    if col_axis: 
        # for each column, return index of cell above threshold, sorted by similarity score
        for col in sim_matrix.columns: 
            col_threshold = min(sim_matrix[col]) + threshold * (max(sim_matrix[col])-min(sim_matrix[col]))
            matches = sim_matrix[sim_matrix[col] >= col_threshold][col]
            matches = matches.sort_values(ascending=False)
            similar_indices[col] = matches.index.tolist()
    # if col_axis is False, find top DAs per word feature
    else:
        for row in sim_matrix.index:
            row_values = sim_matrix.loc[row]
            row_min = row_values.min()
            row_max = row_values.max()
            row_threshold = row_min + threshold * (row_max - row_min)

            matches = row_values[row_values >= row_threshold]
            matches = matches.sort_values(ascending=False)
            similar_indices[row] = matches.index.tolist()
    return similar_indices


def find_embedding_match(row, get_utt_feature, embed_tokenizer, model, avg_word_embedding=False, ontology=None, threshold=0.9, match_for_da=True, csls_normalize=False): 
    words, word_embeddings = retrieve_word_embedding(row['utt'], get_utt_feature, embed_tokenizer, model, avg_word_embedding=avg_word_embedding)
    das, da_embeddings = retrieve_da_embedding(row['da'], get_utt_feature, embed_tokenizer, model, avg_word_embedding, ontology)
    sim_matrix = tokenwise_pairwise_cosine_similarity(word_embeddings, da_embeddings)
    if csls_normalize:
        sim_matrix = csls_matrix(sim_matrix)
    
    sample_word_expl = {i[0][0]:i[1] for i in row['explanation_utt']}
    sample_da_expl = {i[0][0]:i[1] for i in row['explanation_da']}

    sample_words_display = [f'{w} {i} ({round(sample_word_expl[f"{w} {i}"],4)})' for i, w in enumerate(words)]
    sample_da_display = []
    for i, da in enumerate(das):
        idx = i + row["n_sys_da"]
        key = f"{da} {idx}"
        val = round(sample_da_expl[key], 4)
        sample_da_display.append(f"{da} {idx} ({val})")

    sim_matrix_df = similarity_matrix_to_df(sim_matrix, sample_words_display, sample_da_display)
    
    # Find matching features
    threshold_matches = threshold_similarities(sim_matrix_df, threshold=threshold, col_axis=match_for_da)
    # Flip the matches
    flipped_threshold_matches = {}
    for key, value_list in threshold_matches.items():
        for item in value_list:
            flipped_threshold_matches.setdefault(item, []).append(key)

    matching_result = _get_matching_result(das, sample_da_display, words, sample_words_display, sample_da_expl, sample_word_expl, threshold_matches, row, match_for_da)
    flipped_matching_result = _get_matching_result(das, sample_da_display, words, sample_words_display, sample_da_expl, sample_word_expl, flipped_threshold_matches, row, not match_for_da)
        

    return sim_matrix_df, threshold_matches, flipped_threshold_matches, matching_result, flipped_matching_result


def _get_matching_result(das, da_displays, words, word_displays, da_expl, word_expl, matches, row, match_for_da):
    matching_result = []
    if match_for_da:
        for i, (da, da_display) in enumerate(zip(das, da_displays)): 
            if da_display not in matches: continue
            new_matched = {'label': row['label'], 
                        'dia_no': row['dia_no'], 
                        'turn_no': row['turn_no'],
                        'da': da, 
                        'da_expl': da_expl[' '.join(da_display.split()[:-1])],
                        'da_i': i + row['n_sys_da'],
                        'utt': matches[da_display]}
            new_matched['individual_utt_expl'] = [word_expl[' '.join(w.split()[:-1])] for w in new_matched['utt']]
            new_matched['utt_expl'] = np.mean(new_matched['individual_utt_expl']).item()
            matching_result.append(new_matched)
    else:
        for i, (word, word_display) in enumerate(zip(words, word_displays)): 
            if word_display not in matches: continue
            new_matched = {'label': row['label'], 
                        'dia_no': row['dia_no'], 
                        'turn_no': row['turn_no'],
                        'utt': word,
                        'utt_expl': word_expl[' '.join(word_display.split()[:-1])],
                        'utt_i': i,
                        'da': matches[word_display]}
            new_matched['individual_da_expl'] = [da_expl[' '.join(da.split()[:-1])] for da in new_matched['da']]
            new_matched['da_expl'] = np.mean(new_matched['individual_da_expl']).item()
            matching_result.append(new_matched)
    return matching_result


def csls_matrix(sim_matrix: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    Compute CSLS-adjusted similarity matrix.
    k = Number of nearest neighbors to use in local scaling.
    """
    n_X, n_Y = sim_matrix.shape
    k_row = min(k, n_Y)  # max valid k along row (dim=1)
    k_col = min(k, n_X)  # max valid k along column (dim=0)

    avg_sim_y = torch.topk(sim_matrix, k_col, dim=0).values.mean(dim=0)
    avg_sim_x = torch.topk(sim_matrix, k_row, dim=1).values.mean(dim=1)

    csls = 2 * sim_matrix - avg_sim_x[:, None] - avg_sim_y[None, :]
    return csls