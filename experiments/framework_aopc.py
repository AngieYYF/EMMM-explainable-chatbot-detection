from XFramework.DataManager import DataManager, load_dataset
from XFramework.utils import read_pickle, backbone_models, write_pickle
from XFramework.Framework import EMMM
from aopc import aopc_comparison_dataset
from transformers import AutoModelForSequenceClassification
import pandas as pd
import argparse
import warnings
import json
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
                            type=str, default=backbone_models[0])
        parser.add_argument("-save_dir", help="Location to save the artifacts (model and explanations).",
                            type=str)
        parser.add_argument("-AOPC_K", help="Maximum K for AOPC",
                            type=int, default="10")
        parser.add_argument("-prediction_class", help="The samples predicted with the class would be used for AOPC calculation.",
                            type=int, nargs="+")
        parser.add_argument("-local_expl_file", help="File path to local explanation file.",
                            type=str)
        parser.add_argument("-da_utt_scores", help="File path to aggregated importance file.",
                            type=str)
        parser.add_argument("-aopc_file_name", help="File name of the saved AOPC result.",
                            type=str, default="utt_aopc.pkl")
        parser.add_argument("-methods", help="Ranking methods to evaluate AOPC for.",
                            type=str, nargs="+")
        parser.add_argument("-aopc_method", help="AOPC calculation method.",
                            choices = ["original", "accelerating"], 
                            type=str, default="original")
        
        return parser.parse_args() 
    args = parse_arguments()

    if args.dataset == 'both': # need to update how the framework distinguish between samples (need 'dataset' column)
        # mwoz
        train_dataset, val_dataset, test_dataset = load_dataset('mwoz', args.da_extraction_method)
        max_mwoz_dia_no = max([max(train_dataset['dia_no']), max(val_dataset['dia_no']), max(test_dataset['dia_no'])]) + 1
        # frames
        train_dataset_f, val_dataset_f, test_dataset_f = load_dataset('frames', args.da_extraction_method)
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
    print(f"Loaded datasets ({args.dataset}): {len(train_dataset)} Train, {len(val_dataset)} Val, {len(test_dataset)} Test.", flush=True)


    turn_lv_path = os.path.join(args.save_dir, 'turn_lv_utt/results')
    checkpoint_path = EMMM._get_latest_checkpoint(turn_lv_path)
    loaded_tokenizer = DataManager.new_tokenizer(args.turn_lv_utt_model)
    loaded_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    predictions = json.load(open(os.path.join(turn_lv_path, 'utt_test_evaluation.json'),'r'))['test_predictions']
    local_expl = read_pickle(args.local_expl_file)
    da_utt_scores = read_pickle(args.da_utt_scores)
    aopc_results = aopc_comparison_dataset(test_dataset, args.AOPC_K, predictions, 
                            loaded_model, loaded_tokenizer, 
                            args.prediction_class, 
                            local_expl, da_utt_scores, args.methods, args.aopc_method)
    write_pickle(aopc_results, os.path.join(args.save_dir, args.aopc_file_name))

if __name__ == "__main__":
    main()

# Example commands to run the script:
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

# python3 experiments/framework_aopc.py \
#     -da_extraction_method "qwen" \
#     -dataset "frames" \
#     -turn_lv_utt_model "distilgpt2" \
#     -save_dir "experiments/output/frames_distilgpt2" \
#     -AOPC_K 21\
#     -prediction_class 0 1\
#     -local_expl_file "experiments/output/frames_distilgpt2/explanation/utt_turnLv_fs_order1_test"\
#     -da_utt_scores "experiments/output/frames_distilgpt2/explanation/processed/fs_order1_lor_1grams.pkl"\
#     -aopc_file_name "aopcAcc_DA_global_lor.pkl"\
#     -methods "DA" "global"\
#     -aopc_method "accelerating"