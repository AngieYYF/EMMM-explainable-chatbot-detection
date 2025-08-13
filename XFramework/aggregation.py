from nltk.util import ngrams 
from tqdm import tqdm
from collections import Counter
import numpy as np
import pandas as pd
import re
from explanation_match_embedding import find_embedding_match, get_utt_feature
from XFramework.utils import write_pickle


def get_expl_feature_match(feature_match_model, feature_match_tokenizer, expl_tokenizer, 
                           da_expl, utt_expl, train_dataset, ontology, save_path): 
    '''Matching DAs and utterance features across the dataset.'''
    merged_expl = merge_explanation(da_expl, utt_expl, train_dataset)

    matched_result = []
    matched_result_flipped = []
    for i, row in tqdm(merged_expl.iterrows(), total = len(merged_expl)): 
        _, _, _, row_matching_result, row_matching_result_flipped = find_embedding_match(row, lambda x: get_utt_feature(x, expl_tokenizer), feature_match_tokenizer, feature_match_model, 
                                                            avg_word_embedding=False, 
                                                            ontology=ontology, 
                                                            threshold=0.90,
                                                            match_for_da=False, 
                                                            csls_normalize=True)
        matched_result.extend(row_matching_result)
        matched_result_flipped.extend(row_matching_result_flipped)
    matched_result = pd.DataFrame(matched_result)
    matched_result_flipped = pd.DataFrame(matched_result_flipped)

    write_pickle(matched_result, save_path+".pkl")
    write_pickle(matched_result_flipped, save_path+"_flipped.pkl")
    return matched_result, matched_result_flipped

def merge_explanation(da_expl, utt_expl, original_dataset):
    # merge DA-based and utt-based explanations
    merged_expl = pd.merge(da_expl[['explanation', 'n_features', 'dia_turn_label']], 
                       utt_expl[['explanation', 'n_features', 'dia_turn_label']], 
                       on='dia_turn_label',
                       how='inner',
                       suffixes=['_da', '_utt'])
    merged_expl[['dia_no', 'turn_no', 'label']] = pd.DataFrame(
        merged_expl['dia_turn_label'].tolist(), index=merged_expl.index
    )
    del merged_expl['dia_turn_label']
    merged_expl = merged_expl[['label', 'dia_no', 'turn_no', 'explanation_da', 'n_features_da', 'explanation_utt']]

    # Get original utterance and da
    original_dataset_utts = pd.DataFrame(columns=['label', 'dia_no', 'turn_no', 'utt', 'da', 'n_sys_da'])
    original_dataset_utts['da'] = original_dataset_utts['da'].astype(object)

    usr_only = any(da_expl['masked_sample'].apply(len)!=2)
    for i, row in original_dataset.iterrows():
        usr_utts = [utt[len("user: "):] for utt in row['dia'].split('\n') if utt.startswith("user")]
        for utt_no, utt in enumerate(usr_utts):
            original_dataset_utts.loc[len(original_dataset_utts)] = {'label':row['label'],
                                                            'dia_no':row['dia_no'],
                                                            'turn_no':utt_no,
                                                            'utt':utt,
                                                            'da':row['dialogue_act_info_removed'][utt_no*2],
                                                            'n_sys_da':0 if usr_only else 1 if utt_no==0 else len(row['dialogue_act_info_removed'][utt_no*2-1])}
    
    merged_expl = pd.merge(merged_expl, 
                        original_dataset_utts, 
                        on=['label','dia_no','turn_no'],
                        how='inner')
    return merged_expl


def get_all_ngrams(strings, expls = None, max_n = None):
    '''Extract substrings with <=n tokens from a list of strings, along with their attribution scores.'''
    if expls is None: 
        expls = [[]] * len(strings)
    all_ngrams = []
    all_ngrams_expls = []
    for s, token_expls in zip(strings, expls):
        tokens = s.split()
        if token_expls == []: 
            token_expls = [-100] * len(tokens)
        n = len(tokens)
        if max_n is not None: 
            n = min(n, max_n)
        for i in range(1, n+1): 
            ngram_tuples = ngrams(zip(tokens, token_expls), i)
            for gram in ngram_tuples:
                all_ngrams.append(' '.join([g[0] for g in gram]))
                all_ngrams_expls.append(np.mean([g[1] for g in gram]))

    return pd.DataFrame({'matched_utts': all_ngrams, 'matched_utt_expls': all_ngrams_expls})


def compute_aggregation_for_da(match_result, da_str_list, agg_method, alpha=1, max_n=5):
    '''DA-based aggregation of local explanations'''
    if da_str_list: 
        subset = match_result[match_result['da_prefix'].astype(str).isin(da_str_list)]
    else: 
        subset = match_result
    utts = [w.lower() for utt in subset['utt'].tolist() for w in utt]
    utts_expl = subset['individual_utt_expl'].tolist()
    ngrams_df = get_all_ngrams(utts, utts_expl, max_n=max_n)

    human = ngrams_df[ngrams_df['matched_utt_expls'] < 0]
    ai = ngrams_df[ngrams_df['matched_utt_expls'] > 0]

    match agg_method: 
        case 'lor': 
            agg_result = log_odds_ratio(ai['matched_utts'].values, human['matched_utts'].values)
        case 'lsr': 
            agg_result = log_score_ratio(ai.groupby(by='matched_utts')['matched_utt_expls'].sum().apply(abs),
                            human.groupby(by='matched_utts')['matched_utt_expls'].sum().apply(abs))
        case 'fwlor':
            agg_result = frequency_weighted_log_odds_ratio(ai['matched_utts'].values,
                            human['matched_utts'].values)
        case 'fwlsr':
            agg_result = frequency_weighted_log_score_ratio(ai.groupby(by='matched_utts')['matched_utt_expls'].sum().apply(abs),
                            human.groupby(by='matched_utts')['matched_utt_expls'].sum().apply(abs))
        case 'agg_o':
            agg_result = aggregation_odds(ai['matched_utts'].values, 
                        human['matched_utts'].values, 
                        alpha=alpha)
        case 'agg_s':
            agg_result = aggregation_score(ai.groupby(by='matched_utts')['matched_utt_expls'].sum().apply(abs), 
                            human.groupby(by='matched_utts')['matched_utt_expls'].sum().apply(abs), 
                            alpha=alpha)
            
    return agg_result

def retrieve_aggregation_by_da(DAs, match_result, agg_method, combined_da_only=False, max_n_gram=5, alpha=1):
    '''Given the target DAs, retrieve the contextualized aggregation'''
    if DAs is None: 
        target_da_str = []
    else: 
        target_da_str = list(map(str, DAs))
    combined_agg_result = compute_aggregation_for_da(match_result, target_da_str, agg_method, max_n=max_n_gram, alpha=alpha)
    all_agg_result = [combined_agg_result.reset_index().rename(columns={'index': 'utt', 0: 'score'})]
    if combined_da_only: return pd.concat(all_agg_result, ignore_index=True)
    for da in target_da_str: 
         da_agg_result = compute_aggregation_for_da(match_result, [da], agg_method, max_n=max_n_gram, alpha=alpha)
         all_agg_result.append(da_agg_result.reset_index().rename(columns={'index': 'utt', 0: 'score'}))
    return pd.concat(all_agg_result, ignore_index=True)


def retrieve_phrases(matched_da2utt, expl_tokenizer): 
    '''Retrieve phrases from matched text-spans, replacing the DA values with slot tags'''
    def replace_da_val(text, da_val, da_slot): 
        if not da_val: return text
        da_slot_tag = f'<{da_slot}>'
        # Escape tokens and join with | for OR
        pattern = r'\b(' + '|'.join(re.escape(tok) for tok in da_val) + r')\b'

        # Replace all occurrences ignoring case
        text = re.sub(pattern, da_slot_tag, text, flags=re.IGNORECASE)

        # Deduplicate consecutive slot tags
        dedup_pattern = re.compile(r'(\s*)(' + re.escape(da_slot_tag) + r'(?:\s+' + re.escape(da_slot_tag) + r')+)(\s*)')
        
        def dedup_repl(m):
            # Return single tag surrounded by original surrounding spaces
            return m.group(1) + da_slot_tag + m.group(3)
        return dedup_pattern.sub(dedup_repl, text)

    matched_phrases = []
    for _, row in matched_da2utt.iterrows(): 
        tokens = [w.split()[0] for w in row['utt']]
        ids = [int(w.split()[1]) for w in row['utt']]
        da_slot, da_val = row['da'][2], row['da'][3]
        if da_val: da_val = get_utt_feature(' '+da_val, expl_tokenizer)
        else: da_val = []
        phrases = []
        phrases_expl = []
        cur_phrase = []
        cur_expl = []
        cur_ids = ids[0]-1
        for i, word, word_expl in zip(ids, tokens, row['individual_utt_expl']): 
            if i == cur_ids + 1:
                cur_phrase.append(word)
                cur_expl.append(word_expl)
            else: 
                phrases.append(replace_da_val(' '.join(cur_phrase), da_val, da_slot))
                phrases_expl.append(cur_expl)
                cur_phrase, cur_expl = [word], [word_expl]
            cur_ids = i
        if cur_phrase: 
            phrases.append(replace_da_val(' '.join(cur_phrase), da_val, da_slot))
            phrases_expl.append(cur_expl)

        for feature, feature_expl in zip(phrases, phrases_expl): 
            new_row = row.copy()
            new_row['utt'] = [feature]
            new_row['individual_utt_expl'] = feature_expl
            new_row['utt_expl'] = np.mean(feature_expl).item()
            matched_phrases.append(new_row)

    return pd.DataFrame(matched_phrases)

def get_da_utt_scores(matched_da2utt, train_dataset, agg_method, save_path, max_n_gram=1, expl_tokenizer=None): 
    '''Compute the DA-specific profiles for contextualized aggregation.'''
    if max_n_gram>1: 
        matched_da2utt = retrieve_phrases(matched_da2utt, expl_tokenizer)
    else: 
        matched_da2utt['utt']=matched_da2utt['utt'].apply(lambda x: [' '.join([w.split()[0] for w in x])])
    matched_da2utt['da_prefix'] = matched_da2utt['da'].apply(lambda x: str(x[:3]))
    # da-utt scores
    unique_das = []
    for dia_da in train_dataset['dialogue_act_info_removed']: 
        for turn_da in dia_da: 
            for da in turn_da: 
                da = da[:3]
                if da not in unique_das: 
                    unique_das.append(da)
    da_utt_scores = {}
    for da in unique_das:
        da_utt_scores[str(da)] = retrieve_aggregation_by_da([da], matched_da2utt, agg_method, combined_da_only=True, max_n_gram=max_n_gram, alpha=0.5)
    da_utt_scores['global'] = retrieve_aggregation_by_da(None, matched_da2utt, agg_method, combined_da_only=True, max_n_gram=max_n_gram, alpha=0.5)
    write_pickle(da_utt_scores, save_path if save_path.endswith('.pkl') else save_path+".pkl")
    return da_utt_scores




# Aggregation metrics
def log_odds_ratio(corpus_a, corpus_b, prior_strength=2.0):
    """
    Compute Fightin' Words log-odds with informative Dirichlet prior.
    
    corpus_a, corpus_b: Lists of tokens (pre-tokenized, e.g. ['hello', 'world'])
    prior_strength: Total weight of the prior (sum of all alpha_i)
        
    Returns a pandas.Series, Tokens as index, z-scores as values, sorted by descending absolute z-score.
    """
    # Count tokens
    counts_a = Counter(corpus_a)
    counts_b = Counter(corpus_b)
    
    # Get vocabulary and mapping
    vocab = sorted(set(counts_a.keys()) | set(counts_b.keys()))
    V = len(vocab)
    
    # Total counts
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    
    # Compute informative prior from pooled counts
    pooled_counts = counts_a + counts_b
    total_pooled = total_a + total_b
    
    # Calculate alpha_i = prior_strength * (pooled_count / total_pooled)
    alphas = np.array([prior_strength * (pooled_counts.get(word, 0) / total_pooled) 
                       for word in vocab])
    alpha_0 = np.sum(alphas)
    
    # Get counts for each word in vocab
    count_a = np.array([counts_a.get(word, 0) for word in vocab])
    count_b = np.array([counts_b.get(word, 0) for word in vocab])
    
    # Compute log-odds ratio with prior smoothing
    logit_a = np.log((count_a + alphas) / (total_a + alpha_0 - count_a - alphas + 1e-9))
    logit_b = np.log((count_b + alphas) / (total_b + alpha_0 - count_b - alphas + 1e-9))
    delta = logit_a - logit_b
    
    # Compute variance and z-scores
    variance = 1. / (count_a + alphas + 1e-9) + 1. / (count_b + alphas + 1e-9)
    z_scores = delta / np.sqrt(variance)
    
    # Create Series with tokens as index and z-scores as values
    result_series = pd.Series(z_scores, index=vocab, name='score')
    
    # Sort by absolute z-score (descending) while preserving sign
    return result_series.iloc[(-result_series.abs()).argsort()]


def log_score_ratio(corpus_a, corpus_b, prior_strength=2.0):
    """
    Compute Fightin' Words log-odds with informative Dirichlet prior, but using the score instead of the count.
    """
    # Count tokens
    counts_a = corpus_a
    counts_b = corpus_b
    
    # Get vocabulary and mapping
    vocab = sorted(set(counts_a.index) | set(counts_b.index))
    V = len(vocab)
    
    # Total counts
    total_a = sum(counts_a.values)
    total_b = sum(counts_b.values)
    
    # Compute informative prior from pooled counts
    pooled_counts = {word:counts_a.get(word,0) + counts_b.get(word,0) for word in vocab}
    total_pooled = total_a + total_b
    
    # Calculate alpha_i = prior_strength * (pooled_count / total_pooled)
    alphas = np.array([prior_strength * (pooled_counts.get(word, 0) / total_pooled) 
                       for word in vocab])
    alpha_0 = np.sum(alphas)
    
    # Get counts for each word in vocab
    count_a = np.array([counts_a.get(word, 0) for word in vocab])
    count_b = np.array([counts_b.get(word, 0) for word in vocab])
    
    # Compute log-odds ratio with prior smoothing
    logit_a = np.log((count_a + alphas) / (total_a + alpha_0 - count_a - alphas + 1e-9))
    logit_b = np.log((count_b + alphas) / (total_b + alpha_0 - count_b - alphas + 1e-9))
    delta = logit_a - logit_b
    
    # Compute variance and z-scores
    variance = 1. / (count_a + alphas + 1e-9) + 1. / (count_b + alphas + 1e-9)
    z_scores = delta / np.sqrt(variance)
    
    # Create Series with tokens as index and z-scores as values
    result_series = pd.Series(z_scores, index=vocab, name='score')
    
    # Sort by absolute z-score (descending) while preserving sign
    return result_series.iloc[(-result_series.abs()).argsort()]



# Global aggregation of local explanations
def get_counts(utterances):
    return Counter(utterances)

def frequency_weighted_log_odds_ratio(corpus_a, corpus_b, prior_strength=2.0):
    """
    Compute Fightin' Words log-odds with informative Dirichlet prior, weighted by frequency.
    """
    # Count tokens
    counts_a = Counter(corpus_a)
    counts_b = Counter(corpus_b)
    
    # Get vocabulary and mapping
    vocab = sorted(set(counts_a.keys()) | set(counts_b.keys()))
    V = len(vocab)
    
    # Total counts
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    
    # Compute informative prior from pooled counts
    pooled_counts = counts_a + counts_b
    total_pooled = total_a + total_b
    
    # Calculate alpha_i = prior_strength * (pooled_count / total_pooled)
    alphas = np.array([prior_strength * (pooled_counts.get(word, 0) / total_pooled) 
                       for word in vocab])
    alpha_0 = np.sum(alphas)
    
    # Get counts for each word in vocab
    count_a = np.array([counts_a.get(word, 0) for word in vocab])
    count_b = np.array([counts_b.get(word, 0) for word in vocab])
    
    # Compute log-odds ratio with prior smoothing
    logit_a = np.log((count_a + alphas) / (total_a + alpha_0 - count_a - alphas + 1e-9))
    logit_b = np.log((count_b + alphas) / (total_b + alpha_0 - count_b - alphas + 1e-9))
    delta = logit_a - logit_b
    
    # Apply scaling
    p_a = (count_a + alphas) / (total_a + alpha_0)
    p_b = (count_b + alphas) / (total_b + alpha_0)
    scaling = np.sqrt(p_a + p_b)
    frequency_weighted_delta = delta * scaling
    
    # Create Series with tokens as index and scores as values
    result_series = pd.Series(frequency_weighted_delta, index=vocab, name='score')
    return result_series



def frequency_weighted_log_score_ratio(corpus_a, corpus_b, prior_strength=2.0):
    """
    Compute log score ratio, weighted by frequency.
    """
    # Count tokens
    counts_a = corpus_a
    counts_b = corpus_b
    
    # Get vocabulary and mapping
    vocab = sorted(set(counts_a.index) | set(counts_b.index))
    V = len(vocab)
    
    # Total counts
    total_a = sum(counts_a.values)
    total_b = sum(counts_b.values)
    
    # Compute informative prior from pooled counts
    pooled_counts = {word:counts_a.get(word,0) + counts_b.get(word,0) for word in vocab}
    total_pooled = total_a + total_b
    
    # Calculate alpha_i = prior_strength * (pooled_count / total_pooled)
    alphas = np.array([prior_strength * (pooled_counts.get(word, 0) / total_pooled) 
                       for word in vocab])
    alpha_0 = np.sum(alphas)
    
    # Get counts for each word in vocab
    count_a = np.array([counts_a.get(word, 0) for word in vocab])
    count_b = np.array([counts_b.get(word, 0) for word in vocab])
    
    # Compute log-odds ratio with prior smoothing
    logit_a = np.log((count_a + alphas) / (total_a + alpha_0 - count_a - alphas + 1e-9))
    logit_b = np.log((count_b + alphas) / (total_b + alpha_0 - count_b - alphas + 1e-9))
    delta = logit_a - logit_b
    
    # Apply scaling
    # relative frequency (count / total) instead of raw count, to normalize across corpora which may be of different size
    p_a = (count_a + alphas) / (total_a + alpha_0)
    p_b = (count_b + alphas) / (total_b + alpha_0)
    scaling = np.sqrt(p_a + p_b)
    frequency_weighted_delta = delta * scaling
    
    # Create Series with tokens as index and scores as values
    result_series = pd.Series(frequency_weighted_delta, index=vocab, name='score')
    return result_series




def aggregation_odds(corpus_a, corpus_b, alpha=0.5):
    '''Compute the AGG aggregation metric from "Accelerating the Global Aggregation of Local Explanations"'''
    # Vocabulary
    counts_a = get_counts(corpus_a)
    counts_b = get_counts(corpus_b)
    all_keys = set(counts_a) | set(counts_b)
    
    # Sums
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    
    # weighting
    weight_importance = 1 / alpha
    weight_penalty = 1 - weight_importance

    # Compute log odds ratios
    scores = {}
    if not all_keys: return pd.Series(scores)

    for word in all_keys:
        a = counts_a[word]
        b = counts_b[word]

        corpus_a_importance = weight_importance * a / total_a if total_a else 0
        corpus_b_penalty = weight_penalty * b / total_b if total_b else 0
        scores[word] = corpus_a_importance + corpus_b_penalty

    return pd.Series(scores).sort_values(ascending=False)


def aggregation_score(corpus_a, corpus_b, alpha=0.5):
    '''Compute the AGG aggregation metric from "Accelerating the Global Aggregation of Local Explanations", replacing frequency with attribution score.'''
    # Vocabulary
    counts_a = corpus_a
    counts_b = corpus_b
    all_keys = set(counts_a.index) | set(counts_b.index)
    
    # Sums
    total_a = sum(counts_a.values)
    total_b = sum(counts_b.values)

    # weighting
    weight_importance = 1 / alpha
    weight_penalty = 1 - weight_importance
    
    # Compute log odds ratios
    scores = {}
    if not all_keys: return pd.Series(scores)

    for word in all_keys:
        a = counts_a.get(word,0)
        b = counts_b.get(word,0)
        corpus_a_importance = weight_importance * a / total_a if total_a else 0
        corpus_b_penalty = weight_penalty * b / total_b if total_b else 0
        scores[word] = corpus_a_importance + corpus_b_penalty
        
    return pd.Series(scores).sort_values(ascending=False)