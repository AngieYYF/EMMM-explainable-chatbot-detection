from typing import List
from wordcloud import WordCloud
from functools import reduce
import matplotlib.pyplot as plt
import base64
import io
import pandas as pd
import os
import math
from output_template import output_template_demo


class Report_Generator:
    DEFAULT_DEMO_ID = -1
    
    def __init__(self, da_utt_scores, slot_info, save_dir):
        self.precomputed_top_utts = self.precompute_top_utts(da_utt_scores)
        self.slot_info = slot_info
        self.demo_id = Report_Generator.DEFAULT_DEMO_ID
        self.report_save_dir = os.path.join(save_dir,'report')
        os.makedirs(self.report_save_dir, exist_ok=True)
        self.global_aggregation = self.generate_global_AI_aggregation()


    def precompute_top_utts(self, da_utt_scores): 
        precomputed_top_utts = {}
        for da_key, df in da_utt_scores.items():
            df = df.copy()
            # Positive score (class 1)
            pos_df = df[df['score'] > 0].sort_values(by='score', ascending=False).head(20)
            pos_series = pos_df.set_index('utt')['score']
            
            # Negative score (class 0)
            neg_df = df[df['score'] < 0].sort_values(by='score').head(20)
            neg_series = neg_df.set_index('utt')['score'].apply(lambda x: -x)

            precomputed_top_utts[da_key] = {
                0: neg_series,
                1: pos_series
            }
        return precomputed_top_utts


    
    def generate_global_AI_aggregation(self):  
        global_da_utts = {'ai': self.get_da_utt(['global'], pred_class=1), 
                    'human': self.get_da_utt(['global'], pred_class=0)}

        global_wordclouds = {}
        for class_label, da_utts in global_da_utts.items(): 
            da_utts = self.filter_isolated_phrases_tokenwise(da_utts, min_overlap=2)
            da_utts = (da_utts * [math.log(len(_.split())+1) for _ in da_utts.index]).sort_values(ascending=False)

            global_wordclouds[class_label] = self.visualize_wordcloud(da_utts)
        return global_wordclouds


    def filter_isolated_phrases_tokenwise(self, series: pd.Series, min_overlap: int = 3):
        # return phrases that either (1) do not have any other phrases containing it or (2) do not contain any other phrases
        # i.e. phrases that are in the middle are removed to reduce duplications
        # then remaining phrases are merged

        def is_subsequence(sub: List[str], full: List[str]) -> bool:
            # check if sub is a subsequence of full
            n, m = len(sub), len(full)
            return any(full[i:i+n] == sub for i in range(m - n + 1))

        def merge_all(strings: List[str], min_overlap: int = 3) -> List[str]:
            # check for potential merges of strings

            def merge_on_overlap(s1: str, s2: str, min_overlap: int = 3) -> str | None:
                # merge s1 an s2 if there is a consecutive overlap from the 2 ends for a minimum of min_overlap tokens
                
                tokens1 = s1.split()
                tokens2 = s2.split()
                max_len = min(len(tokens1), len(tokens2))
                
                for i in range(max_len, min_overlap - 1, -1):
                    # Check if end of tokens1 matches start of tokens2
                    if tokens1[-i:] == tokens2[:i]:
                        return ' '.join(tokens1 + tokens2[i:])
                    # Check if end of tokens2 matches start of tokens1
                    if tokens2[-i:] == tokens1[:i]:
                        return ' '.join(tokens2 + tokens1[i:])
                
                return None

            strings = strings[:]  # avoid modifying original
            merged = True

            while merged:
                merged = False
                new_strings = []
                used = set()

                for i, s1 in enumerate(strings):
                    if i in used:
                        continue
                    found_merge = False
                    for j, s2 in enumerate(strings):
                        if i == j or j in used:
                            continue
                        merged_str = merge_on_overlap(s1, s2, min_overlap)
                        if merged_str:
                            new_strings.append(merged_str)
                            used.update({i, j})
                            merged = True
                            found_merge = True
                            break
                    if not found_merge and i not in used:
                        new_strings.append(s1)

                strings = new_strings

            return strings

        # Tokenize phrases once
        tokenized = {p: p.split() for p in series.index}
        phrases = list(series.index)

        keep_shortest = []
        keep_longest = []

        for phrase in phrases:
            tokens = tokenized[phrase]

            has_sub = any(
                phrase != other and is_subsequence(tokenized[other], tokens)
                for other in phrases
            )
            is_sub = any(
                phrase != other and is_subsequence(tokens, tokenized[other])
                for other in phrases
            )

            if not has_sub:
                keep_shortest.append(phrase)
            elif not is_sub:
                keep_longest.append(phrase)

        filtered_phrases = keep_shortest + keep_longest

        # Merge overlapping filtered phrases
        merged_phrases = merge_all(filtered_phrases, min_overlap=min_overlap)

        # Assign the max original score from any contributing substring
        merged_scores = {}
        for merged in merged_phrases:
            contributing = [p for p in series.index if p in merged]
            if contributing:
                merged_scores[merged] = series[contributing].max()
            else:
                merged_scores[merged] = 1  # fallback if no original match

        return pd.Series(merged_scores).sort_values(ascending=False)
        

    def aggregation(self, utt_da): 
        class_da_utts = {'ai': self.get_da_utt(utt_da, pred_class=1), 
                    'human': self.get_da_utt(utt_da, pred_class=0)}
        wordcloud_paths = {}
        for class_label, da_utts in class_da_utts.items(): 
            if da_utts is not None: 
                da_utts = self.filter_isolated_phrases_tokenwise(da_utts, min_overlap=2)
                da_utts = (da_utts * [math.log(len(_.split())+1) for _ in da_utts.index]).sort_values(ascending=False)
                if not da_utts.empty: 
                    wordcloud_paths[class_label] = self.visualize_wordcloud(da_utts)
                    continue
            # wordcloud_paths[class_label] = self.global_aggregation[class_label] # fall back to global_aggregation if DA not available
            wordcloud_paths[class_label] = None # do not display aggregation
        return wordcloud_paths

    def encode_img(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return encoded

    def visualize_wordcloud(self, word_freqs): 
        # Convert Series to dictionary
        freq_dict = word_freqs.to_dict()

        # Create word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq_dict)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        encoded_img = self.encode_img(fig)
        plt.close(fig)
        return encoded_img

    def get_da_utt(self, das, pred_class):
        result = []
        for da in das:
            if isinstance(da, list):
                da_key = str(da[:3])
            elif da == 'global':
                da_key = da
            else:
                continue

            utt_scores = self.precomputed_top_utts.get(da_key, {}).get(pred_class)
            if utt_scores is not None:
                result.append(utt_scores)
        if result:
            return reduce(lambda x, y: x.add(y, fill_value=0), result)
        return None


    def plot_diaLv_detection(self, diaLv_detection): 
        num_utt = len(diaLv_detection)
        x = list(range(1, num_utt + 1))
        fig, ax = plt.subplots(figsize=(8, 5))

        if num_utt == 1:
            ax.plot(x, diaLv_detection, 'o', markersize=8, color='blue')
            ax.text(x[0], diaLv_detection[0] + 0.02, f"{diaLv_detection[0]:.2f}", ha='center')
        else:
            ax.plot(x, diaLv_detection, '-o', color='blue')
            for xi, yi in zip(x, diaLv_detection):
                ax.text(xi, yi + 0.02, f"{yi:.2f}", ha='center')

        ax.set_xlabel("# Utterances")
        ax.set_ylabel("AI Probability")
        ax.set_title("AI Detection Over Dialogue Progression")
        ax.set_xticks(x)
        ax.set_ylim(0, 1.1)
        ax.grid(False)
        fig.tight_layout()
        encoded_img = self.encode_img(fig)
        plt.close(fig)
        return encoded_img


    def __call__(self, historical_result):
        # get aggregation for the latest utterance
        aggregation_plots = self.aggregation(historical_result.dia_DA[-1])
        historical_result.utt_aggregation_plots.append(aggregation_plots)
        diaLv_detection_plot = self.plot_diaLv_detection(historical_result.dia_detection)

        output_template_demo(self.slot_info, 
        historical_result.utt_expl, historical_result.da_expl, 
        '\n'.join(historical_result.dia[1:]), 
        historical_result.utt_aggregation_plots, 
        historical_result.utt_detection, historical_result.da_detection, 
        diaLv_detection_plot, 
        file_path=os.path.join(self.report_save_dir, f'demo_report_{self.demo_id}.html'))

