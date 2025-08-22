from XFramework.DataManager import DataManager, load_dataset, load_ontology
from transformers import AutoTokenizer, AutoModel
from XFramework.utils import read_pickle, backbone_models
from XFramework.explanation_match_embedding import train_feature_matching_embedding_model
from EMMM.XFramework.aggregation import get_expl_feature_match, get_da_utt_scores
import pandas as pd
import argparse
import warnings
import os

warnings.filterwarnings("ignore", message=".*gather.*scalars.*")


def main():
    print("\n\n"+"="*200+"\n", flush=True)

    # arguments
    def parse_arguments(): 
        parser = argparse.ArgumentParser()
        parser.add_argument("-da_extraction_method", help="Dialogue act extraction method.", 
                            choices = ['convlab', 'qwen'], 
                            type=str, default = 'qwen')
        parser.add_argument("-dataset", help="Bona fide dataset.", 
                            choices = ['mwoz', 'frames', 'both'], 
                            type=str, default = 'mwoz')
        parser.add_argument("-turn_lv_utt_model", help="Backbone model for turn level utterance based detection model.",
                            choices = backbone_models, 
                            type=str, default=backbone_models[2])
        parser.add_argument("-utt_expl_method", help="Explanation extraction method for utterance.",
                            choices = [ 'stii', 'ig', 'fs', 'none'], 
                            type=str, default="fs")
        parser.add_argument("-interaction_order", help="Explanation's order of interaction.",
                            type=int, default="1")
        parser.add_argument("-feature_match_model", help="Location to save/load the feature matching model",
                            type=str)
        parser.add_argument("-agg_method", help="Local explanation aggregation metric",
                            choices = ['lor', 'lsr', 'fwlor', 'fwlsr', 'agg_o', 'agg_s'],
                            type=str, default='fwlor')
        parser.add_argument("-save_dir", help="Location to save the artifacts (model and explanations).",
                            type=str)
        parser.add_argument("-max_n_gram", help="n-grams for phrase extraction",
                            type=int, default="1")
        
        return parser.parse_args() 
    args = parse_arguments()

    if args.dataset == 'both': # need to update how the framework distinguish between samples (need 'dataset' column)
        # mwoz
        train_dataset, val_dataset, test_dataset = load_dataset('mwoz', args.da_extraction_method)
        ontology = load_ontology('mwoz')
        max_mwoz_dia_no = max([max(train_dataset['dia_no']), max(val_dataset['dia_no']), max(test_dataset['dia_no'])]) + 1
        # frames
        train_dataset_f, val_dataset_f, test_dataset_f = load_dataset('frames', args.da_extraction_method)
        ontology.update(load_ontology('frames'))
        for dataset_f in [train_dataset_f, val_dataset_f, test_dataset_f]: 
            dataset_f['dia_no'] = dataset_f['dia_no'].apply(lambda x:x+max_mwoz_dia_no)
        
        
        mwoz_test_dia_min, mwoz_test_dia_max = min(test_dataset['dia_no']), max(test_dataset['dia_no'])
        print(f"mwoz test dia_no: [{mwoz_test_dia_min}, {mwoz_test_dia_max}]", flush=True)
        frames_test_dia_min, frames_test_dia_max = min(test_dataset_f['dia_no']), max(test_dataset_f['dia_no'])
        print(f'frames test dia_no: [{frames_test_dia_min}, {frames_test_dia_max}]', flush=True)

        # combine
        train_dataset = pd.concat([train_dataset, train_dataset_f], ignore_index=True)
        val_dataset = pd.concat([val_dataset, val_dataset_f], ignore_index=True)
        test_dataset = pd.concat([test_dataset, test_dataset_f], ignore_index=True)
    else: 
        train_dataset, val_dataset, test_dataset = load_dataset(args.dataset, args.da_extraction_method)
        ontology = load_ontology(args.dataset)
    print(f"Loaded datasets ({args.dataset}): {len(train_dataset)} Train, {len(val_dataset)} Val, {len(test_dataset)} Test.", flush=True)


    # explanation feature matching
    matched_result_path = os.path.join(args.save_dir, f'explanation/processed/{args.utt_expl_method}_order{args.interaction_order}_utt2da_matched_result')
    expl_tokenizer = DataManager.new_tokenizer(args.turn_lv_utt_model)
    if os.path.exists(matched_result_path+'_flipped.pkl'): 
        print(f"Loading da2utt feature matching result.", flush=True)
        matched_result_flipped = read_pickle(matched_result_path+'_flipped.pkl')
    else: 
        # explanation feature matching model
        if not os.path.exists(args.feature_match_model): 
            print(f"Training feature matching model.", flush=True)
            train_feature_matching_embedding_model(train_dataset, expl_tokenizer, ontology, args.feature_match_model)
        print(f"Loading feature matching model from {args.feature_match_model}", flush=True)
        feature_match_model = AutoModel.from_pretrained(args.feature_match_model)
        feature_match_tokenizer = AutoTokenizer.from_pretrained(args.feature_match_model)

        # explanation feature matching
        print(f"Generating da2utt feature matching result.", flush=True)
        da_expl = read_pickle(os.path.join(args.save_dir, f'explanation/daTurn_turnLv_{args.utt_expl_method}_order{args.interaction_order}_train'))
        utt_expl = read_pickle(os.path.join(args.save_dir, f'explanation/utt_turnLv_{args.utt_expl_method}_order{args.interaction_order}_train'))
        
        _, matched_result_flipped = get_expl_feature_match(feature_match_model, feature_match_tokenizer, expl_tokenizer, 
                                da_expl, utt_expl, train_dataset, ontology, 
                                matched_result_path)

    # da utt scoring
    print(f"Calculating aggregated scores: {args.agg_method}, {args.max_n_gram}-grams.", flush=True)
    save_path = os.path.join(args.save_dir, f'explanation/processed/{args.utt_expl_method}_order{args.interaction_order}_{args.agg_method}_{args.max_n_gram}grams.pkl')
    da_utt_scores = get_da_utt_scores(matched_result_flipped, train_dataset, args.agg_method, save_path, 
                        max_n_gram = args.max_n_gram, expl_tokenizer=expl_tokenizer)


if __name__ == "__main__":
    main()

# Example command to run the script:
# python3 experiments/framework_da_utt_score.py \
#     -da_extraction_method "qwen" \
#     -dataset "frames" \
#     -turn_lv_utt_model "distilgpt2" \
#     -utt_expl_method "fs" \
#     -interaction_order "1" \
#     -feature_match_model "experiments/output/feature_match_model/frames" \
#     -agg_method "lor" \
#     -save_dir "experiments/output/frames_distilgpt2"\
#     -max_n_gram "1"