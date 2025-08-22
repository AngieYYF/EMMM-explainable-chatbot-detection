from XFramework.DataManager import load_dataset
from XFramework.utils import read_pickle, backbone_models, RANDOM_STATE
from XFramework.Framework import EMMM
from XFramework.Framework_demo import EMMM_Demo
from XFramework.demo_utils.info_masking import Info_Masker, frames_info_masking_prompt, mwoz_info_masking_prompt
from XFramework.demo_utils.da_extract import Frames_DA_Extractor, MWoz_DA_Extractor
from XFramework.demo_utils.report_generate import Report_Generator
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
        parser.add_argument("-use_da_value", help="Whether the dialogue act value are used during detection and explanation.", 
                            choices = ['True', 'False'], 
                            type=str, default = 'False')
        parser.add_argument("-use_da_turn", help="Whether to use entire turn of DA or just user's.", 
                            choices = ['True', 'False'], 
                            type=str, default = 'False')
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
        parser.add_argument("-da_utt_scores", help="file name of the aggregated scoring.",
                            type=str)
        parser.add_argument("-demo_id", help="An identifier for the demo.",
                            type=int, default="0")
        parser.add_argument("-utterances", help="utterances to detect.",
                            type=str, nargs="+")
        parser.add_argument("-sys_utterances", help="system utterances.",
                            type=str, nargs="+")
        
        return parser.parse_args() 
    args = parse_arguments()
    
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

    # Initialize framework and load parameters
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
    framework.save_dir = args.save_dir
    framework.assign_turn_lv_model_dir()
    framework.assign_dia_lv_model_dir()
    

    # Demo
    if args.dataset == 'mwoz': 
        sensitive_info_masker = Info_Masker(mwoz_info_masking_prompt)
        da_extractor = MWoz_DA_Extractor()
        slot_info = pd.read_json("XFramework/demo_utils/ontology_mwoz_NLtemplate.json").reset_index()
    elif args.dataset == 'frames': 
        sensitive_info_masker = Info_Masker(frames_info_masking_prompt)
        da_extractor = Frames_DA_Extractor()
        slot_info = pd.read_json("XFramework/demo_utils/ontology_frames_NLtemplate.json").reset_index()
        
    da_utt_scores = read_pickle(os.path.join(args.save_dir, 'explanation', 'processed', args.da_utt_scores))
    report_generator = Report_Generator(da_utt_scores, slot_info, os.path.join(args.save_dir, 'demo'))
    demo_runner = EMMM_Demo(framework, sensitive_info_masker, da_extractor, report_generator, args.demo_id)

    for sys_utt, utt in zip(args.sys_utterances, args.utterances): 
        demo_runner.detect_explain_utt(sys_utt, utt)

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
#     -agg_method "fwlor" \
#     -save_dir "experiments/output/frames_distilgpt2"\
#     -max_n_gram "3"

# python3 experiments/demo.py \
#     -da_extraction_method "qwen" \
#     -dataset "frames" \
#     -use_da_value "False" \
#     -use_da_turn "False" \
#     -da_expl_method "fs" \
#     -utt_expl_method "fs" \
#     -dia_expl_method "none" \
#     -interaction_order 1 \
#     -da_expl_k "3" \
#     -utt_expl_k "3" \
#     -turn_lv_da_model "distilgpt2" \
#     -turn_lv_utt_model "distilgpt2" \
#     -dia_lv_model "late-fusion-separate-pretrained" \
#     -dia_lv_suffix ""\
#     -dia_lv_fusion_method "average"\
#     -random_explanation False \
#     -explanation_ordering "absolute" \
#     -save_dir "experiments/output/frames_distilgpt2"\
#     -epochs 0 0 0 \
#     -online "True" \
#     -evaluate "False"\
#     -da_utt_scores "fs_order1_fwlor_3grams.pkl"\
#     -demo_id "0"\
#     -utterances "user: Hi, I'm looking for a vacation package from Calgary to St. Louis from August 17 to August 31 for one person. Any options?" "user: Can you tell me more about the locations of these hotels in relation to local attractions?" "user: The 5-star Lunar Hotel sounds nice. Can I get the package with economy flights then?"\
#     -sys_utterances "system: Hi! How can I help you?" "system: I found two options for you. A 3-star hotel with business class flights for \$1858 or a 5-star hotel with economy flights for \$1558, both include breakfast and wifi. Which do you prefer?" "system: The 3-star Glorious Cloak Inn is near downtown, close to museums and theaters. The 5-star Lunar Hotel is also centrally located, near parks and shopping areas."