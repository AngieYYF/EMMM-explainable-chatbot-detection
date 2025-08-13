# Dataset
* [Datasets](dataset) are preprocessed, containing original dialogue, dialogue with sensitive information masked, and dialogue acts. 
* Frames dataset construction prompts and pipeline can be found in [E2E_generation_Frame.py](LLM/E2E_generation_Frame.py)

# EMMM Framework
The following procedure can be used to train and evaluate a model of EMMM framework.
1. Train and evaluate detection performance of a EMMM model, run [framework_detection.py](experiments/framework_detection.py)
2. Offline computing of DA-based semi-global aggregation, run [framework_da_utt_score.py](experiments/framework_da_utt_score.py)
3. Generate explanation for a given sample, run [demo.py](experiments/demo.py)
4. Evaluate AOPC, run [framework_aopc.py](experiments/framework_aopc.py)
5. Evaluate time complexity, run [demo_timeComplexity.py](experiments/demo_timeComplexity.py)