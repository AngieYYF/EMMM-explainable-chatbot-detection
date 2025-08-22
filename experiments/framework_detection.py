from XFramework.DataManager import load_dataset
from XFramework.utils import backbone_models, RANDOM_STATE
from XFramework.Framework import EMMM
import pandas as pd
import argparse
import warnings

warnings.filterwarnings("ignore", message=".*gather.*scalars.*")



def main():
    print("\n\n"+"="*200+"\n", flush=True)

    # arguments
    def parse_framework_detection_arguments(): 
        parser = argparse.ArgumentParser()
        parser.add_argument("-da_extraction_method", help="Dialogue act extraction method.", 
                            choices = ['convlab', 'qwen'], 
                            type=str, default = 'qwen')
        parser.add_argument("-dataset", help="Bona fide dataset.", 
                            choices = ['mwoz', 'frames', 'both'], 
                            type=str, default = 'mwoz')
        parser.add_argument("-use_da_value", help="Whether the dialogue act value are used during detection and explanation.", 
                            choices = ['True', 'False'], 
                            type=str, default = 'False')
        parser.add_argument("-use_da_turn", help="Whether to use entire turn of DA or just user's.", 
                            choices = ['True', 'False'], 
                            type=str, default = 'True')
        parser.add_argument("-da_expl_method", help="Explanation extraction method for dialogue act.",
                            choices = [ 'stii', 'ig', 'fs', 'none'], 
                            type=str, default="fs")
        parser.add_argument("-utt_expl_method", help="Explanation extraction method for utterance.",
                            choices = [ 'stii', 'ig', 'fs', 'none'], 
                            type=str, default="fs")
        parser.add_argument("-dia_expl_method", help="Explanation extraction method for dialogue.",
                            choices = [ 'stii', 'ig', 'fs', 'none'], 
                            type=str, default="none")
        parser.add_argument("-interaction_order", help="Explanation's order of interaction.",
                            type=int, default="1")
        parser.add_argument("-da_expl_k", help="Maximum number of dialogue act explanations used per turn for aggregation into inputs for dialogue level detection.",
                            type=int, default="0")
        parser.add_argument("-utt_expl_k", help="Maximum number of utterance explanations used per turn for aggregation into inputs for dialogue level detection.",
                            type=int, default="0")
        parser.add_argument("-da_expl_p", help="Maximum proportion of dialogue act explanations used per turn for aggregation into inputs for dialogue level detection.",
                            type=float, default="0")
        parser.add_argument("-utt_expl_p", help="Maximum proportion of utterance explanations used per turn for aggregation into inputs for dialogue level detection.",
                            type=float, default="0")
        parser.add_argument("-turn_lv_da_model", help="Backbone model for turn level dialogue acts based detection model.",
                            choices = backbone_models, 
                            type=str, default=backbone_models[2])
        parser.add_argument("-turn_lv_utt_model", help="Backbone model for turn level utterance based detection model.",
                            choices = backbone_models, 
                            type=str, default=backbone_models[2])
        parser.add_argument("-dia_lv_model", help="Backbone model for dialogue level detection model.",
                            choices = backbone_models+['single-pretrained', 'late-fusion', 'late-fusion-separate', 'late-fusion-separate-pretrained'], 
                            type=str, default='late-fusion-separate-pretrained')
        parser.add_argument("-dia_lv_suffix", help="Suffix for dialogue level model.",
                            type=str, default='')
        parser.add_argument("-dia_lv_fusion_method", help="Fusion method for DA and utterance.",
                            choices = ['concat', 'max', 'average'], 
                            type=str, default='average')
        parser.add_argument("-random_explanation", help="Whether to extract random explanations, else top important explanations.",
                            choices = ["True", "False"],
                            type=str, default="False")
        parser.add_argument("-explanation_ordering", help="The ordering of explanations to select important features.",
                            choices = ["absolute", "prediction"],
                            type=str, default="absolute")
        parser.add_argument("-save_dir", help="Location to save the artifacts (model and explanations).",
                            type=str)
        parser.add_argument("-epochs", help="Number of epochs to train turn_lv_da, turn_lv_utt, dia_lv, respectively.", 
                            type=int, nargs="+")
        parser.add_argument("-online", help="Whether to perform online detection (trained and tested).",
                            choices = [ 'True', 'False'], type=str)
        parser.add_argument("-evaluate", help="Whether to perform evaluation after training each model.",
                            choices = [ 'True', 'False'], type=str, default="True")
        
        return parser.parse_args() 
    args = parse_framework_detection_arguments()
    
    if args.dataset == 'both': # need to update how the framework distinguish between samples (need 'dataset' column)
        # mwoz
        train_dataset, val_dataset, test_dataset = load_dataset('mwoz', args.da_extraction_method)
        mwoz_dia_no_offset = max([max(train_dataset['dia_no']), max(val_dataset['dia_no']), max(test_dataset['dia_no'])]) + 1
        # frames
        train_dataset_f, val_dataset_f, test_dataset_f = load_dataset('frames', args.da_extraction_method)
        for dataset_f in [train_dataset_f, val_dataset_f, test_dataset_f]: 
            dataset_f['dia_no'] = dataset_f['dia_no'].apply(lambda x:x+mwoz_dia_no_offset)
       
        # confirm the dia_no ranges
        mwoz_dia_min = min(min(ds['dia_no']) for ds in [train_dataset, val_dataset, test_dataset])
        mwoz_dia_max = max(max(ds['dia_no']) for ds in [train_dataset, val_dataset, test_dataset])
        print(f"mwoz dia_no range (inclusive): [{mwoz_dia_min}, {mwoz_dia_max}]", flush=True)
        frames_dia_min = min(min(ds['dia_no']) for ds in [train_dataset_f, val_dataset_f, test_dataset_f])
        frames_dia_max = max(max(ds['dia_no']) for ds in [train_dataset_f, val_dataset_f, test_dataset_f])
        print(f'frames dia_no range (inclusive): [{frames_dia_min}, {frames_dia_max}]', flush=True)

        # combine
        train_dataset = pd.concat([train_dataset, train_dataset_f], ignore_index=True)
        val_dataset = pd.concat([val_dataset, val_dataset_f], ignore_index=True)
        test_dataset = pd.concat([test_dataset, test_dataset_f], ignore_index=True)
    else: 
        train_dataset, val_dataset, test_dataset = load_dataset(args.dataset, args.da_extraction_method)
    
    print(f"Loaded datasets ({args.dataset} {args.da_extraction_method}): {len(train_dataset)} Train, {len(val_dataset)} Val, {len(test_dataset)} Test.", flush=True)

    # Initialize framework
    random_explanation = args.random_explanation == "True"
    online_detection = args.online == "True"
    evaluate = args.evaluate == "True"
    use_da_value = args.use_da_value == "True"
    use_da_turn = args.use_da_turn == "True"
    framework = EMMM(train_dataset, val_dataset, test_dataset, RANDOM_STATE, 
                explanation_methods={'da_method': args.da_expl_method, 'da_k': args.da_expl_k, 'da_p': args.da_expl_p,
                                    'utt_method': args.utt_expl_method, 'utt_k': args.utt_expl_k, 'utt_p': args.utt_expl_p,
                                    'dia_method': args.dia_expl_method,
                                    'interaction_order': args.interaction_order,
                                    'random_explanation':random_explanation, 
                                    'explanation_ordering':args.explanation_ordering}, 
                model_names = {'turn_lv_da': args.turn_lv_da_model, 
                                'turn_lv_utt': args.turn_lv_utt_model,
                                'dia_lv': args.dia_lv_model, 
                                'dia_lv_suffix': args.dia_lv_suffix, 
                                'dia_lv_fusion_method': args.dia_lv_fusion_method},
                use_da_value = use_da_value, use_da_turn = use_da_turn,
                online = online_detection, evaluate_after_train = evaluate)
    
    # Train framework
    assert len(args.epochs) == 3, "Requires three integers for argument epochs."
    framework.train(save_dir=args.save_dir, epochs=args.epochs)
    
    # Online & Offline evaluation on Testing set (if dialogue detection is trained)
    if args.epochs[2] > 0:
        framework.evaluate()

    # remove the training files to save space (do not support further finetuning)
    framework.remove_trainer_files()


if __name__ == "__main__":
    main()


# Example command to run the script:
# python3 experiments/framework_detection.py \
#             -da_extraction_method "qwen" \
#             -dataset "frames" \
#             -use_da_value "False" \
#             -use_da_turn "False" \
#             -da_expl_method "fs" \
#             -utt_expl_method "fs" \
#             -dia_expl_method "none" \
#             -interaction_order 1 \
#             -da_expl_k "3" \
#             -utt_expl_k "3" \
#             -turn_lv_da_model "distilgpt2"\
#             -turn_lv_utt_model "distilgpt2"\
#             -dia_lv_model "late-fusion-separate-pretrained" \
#             -dia_lv_suffix ""\
#             -dia_lv_fusion_method "average"\
#             -random_explanation False \
#             -explanation_ordering "absolute" \
#             -save_dir "experiments/output/frames_distilgpt2" \
#             -epochs 15 10 5 \
#             -online "True" \
#             -evaluate "True"