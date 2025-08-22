<h1 align="center">🤔 EMMM, Explain Me My Model!</h1>

<p align="center">
  <a href="https://arxiv.org/abs/">
    <img src="https://img.shields.io/badge/arXiv-red.svg?logo=arxiv&style=flat" alt="arXiv">
  </a> 
  <a href="https://huggingface.co/datasets/AngieYYF/Frames-synthetic-customer-service-dialogue">
    <img src="https://img.shields.io/badge/🤗-HuggingFace:Datasets-blue.svg" alt="HuggingFace Dataset">
  </a> 
  <a href="https://huggingface.co/collections/AngieYYF/emmm-explain-me-my-model-68a7efd0e25d4aadcf0b98ab">
    <img src="https://img.shields.io/badge/🤗-HuggingFace:Models-blue.svg" alt="HuggingFace Model">
  </a> 
  <img src="https://img.shields.io/badge/made%20with-Python-blue.svg" alt="Made with Python">
</p>

![alt text](Plots/framework.jpg)

---

In this work, we introduce **EMMM**, an explainable LLM chatbot detection framework, targetting MGT detection and its interpretability to diverse stakeholders in **online conversational settings**. Our method balances latency, accuracy, and non-expert-oriented interpretability.

## Key Insights
- **EMMM is Dialogue-Aware.** EMMM leverages conversation specific features to deliver multi-dimension, multi-level, and multi-strategy explanations. Grounded in speech act theory, EMMM models dialogue structure and intent to enhance interpretability and support both online and offline chatbot detection.

- **EMMM is Efficient.** EMMM produces explanation reports online in under 1 second by combining a sequential selector–predictor pipeline with offline preprocessing, achieving the time efficiency required for deployment in real-world service platforms.

- **EMMM is Interpretable.** EMMM generates non-expert user friendly natural language explanation reports and includes visualizations of contextualized semi-global model behaviors to enhance model interpretability. We assess interpretability through qualitative analysis and a human study, with 69% of users preferring our method over the baseline.



<!-- This repository contains:
- Implementation of our proposed EMMM framework.
- Scripts to train, use, and evaluate a model of the EMMM framework.
- Customer service line synthetic user dialogues. -->


## Datasets
We provide preprocessed datasets, containing the dialogues and extracted dialogue acts. To load the datasets:

```python
from datasets import load_dataset

# Login using e.g. `huggingface-cli login` to access this dataset
Frames = load_dataset("AngieYYF/Frames-synthetic-customer-service-dialogue")
Spade_bona = load_dataset("AngieYYF/SPADE-customer-service-dialogue", "bona_fide")
Spade_e2e = load_dataset("AngieYYF/SPADE-customer-service-dialogue", "end_to_end_conversation")

```

Below is a dataset overview to help you quickly understand our datasets. 

### Synthetic-Frames
- Synthetic-Frames dataset is constructed using [prompts and pipeline](LLM/E2E_generation_Frame.py) adapted from SPADE.
- Datasets are available on [Hugging Face](https://huggingface.co/datasets/AngieYYF/Frames-synthetic-customer-service-dialogue) with [Bona fide](dataset/Frames/frames_bona_fide.pkl) and [End-to-End Conversation](dataset/Frames/frames_e2e.pkl) splits annotated with class labels.
- train/val/test splits of dialogues are based on dia_no, recorded [here](dataset/Frames/dataset_splits.pkl).
- The datasets (stored [here](dataset/Frames)) are of pkl file format and contain the following columns:

| Dataset                    | Column            | Description                                                                                              |
|----------------------------|-------------------|----------------------------------------------------------------------------------------------------------|
| **All**                     | *dia_no* | Unique ID for each dialogue. |
|                            | *unmasked_dia*             | bona fide and synthetic dialogues.                                                      |
|                            | *dia*             | Sensitive information masked bona fide and synthetic dialogues.                                                  |
|                            | *dialogue_act*             | Dialogue acts of each utterance within the dialogue.                                                     |
|                            | *dialogue_act_info_removed*             | Sensitive information masked dialogue acts for each utterance.                                                    |
| **Bona Fide**               | *goal_outcome*        | The  user goals and outcomes with format: "goal_1 \<sep\> goal_2 \<sep\> ... \<sep\> goal_n \<outcome\> outcome_1 \<sep\> outcome_2 \<sep\> ... \<sep\> outcome_n"     |

### SPADE
- The **SPADE** dataset can be downloaded from [Hugging Face](https://huggingface.co/datasets/AngieYYF/SPADE-customer-service-dialogue), with dataset construction frameworks described in their [paper](https://aclanthology.org/2025.llmsec-1.11/).
- train/val/test splits of dialogues are based on dia_no, recorded [here](dataset/SPADE/dataset_splits.pkl).
- The datasets (stored [here](dataset/SPADE)) are of pkl file format and contain the following columns:

| Dataset                    | Column            | Description                                                                                              |
|----------------------------|-------------------|----------------------------------------------------------------------------------------------------------|
| **All**                     | *dia_no* | Unique ID for each dialogue. Dialogues with the same ID across datasets are based on the bona fide dialogue with the same *dia_no*. |
|                            | *dia*             | The dialogue itself, either bona fide or synthetic, with sensitive information masked.                                                      |
|                            | *dialogue_act*             | The dialogue acts of each utterance within the dialogue.                                                     |
|                            | *dialogue_act_info_removed*             | The dialogue acts of each utterance within the dialogue, with sensitive information masked.                                                     |

---

## EMMM Framework - Quick Start
The following procedure can be used to train, use, and evaluate a model of EMMM framework.
1. Install the required packages from requirements.txt.
2. Train and evaluate detection performance of a EMMM model: [framework_detection.py](experiments/framework_detection.py)
```python
python3 experiments/framework_detection.py \
            -da_extraction_method "qwen" \ 
            # "qwen" or "convlab"
            -dataset "frames" \
            # "frames" or "mwoz"
            -use_da_turn "False" \
            -da_expl_k "3" \
            -utt_expl_k "3" \
            -save_dir "experiments/output/frames_distilgpt2" \
            -epochs 15 10 5 \
            -online "True" 

```
3. Offline computing of DA-based semi-global aggregation: [framework_da_utt_score.py](experiments/framework_da_utt_score.py)
```python
python3 experiments/framework_da_utt_score.py \
    -da_extraction_method "qwen" \
    -dataset "frames" \
    -feature_match_model "experiments/output/feature_match_model/frames" \
    -save_dir "experiments/output/frames_distilgpt2"\
    -max_n_gram "3"

```
4. Detect and generate explanation for a given dialogue sample: [demo.py](experiments/demo.py)
```python
python3 experiments/demo.py \
    -da_extraction_method "qwen" \
    -dataset "frames" \
    -da_expl_k "3" \
    -utt_expl_k "3" \
    -save_dir "experiments/output/frames_distilgpt2"\
    -epochs 0 0 0 \
    -online "True" \
    -evaluate "False"\
    -da_utt_scores "fs_order1_fwlor_3grams.pkl"\
    -demo_id "0"\
    -utterances "user: Hi, I'm looking for a vacation package from Calgary to St. Louis from August 17 to August 31 for one person. Any options?" "user: Can you tell me more about the locations of these hotels in relation to local attractions?" "user: The 5-star Lunar Hotel sounds nice. Can I get the package with economy flights then?"\
    -sys_utterances "system: Hi! How can I help you?" "system: I found two options for you. A 3-star hotel with business class flights for \$1858 or a 5-star hotel with economy flights for \$1558, both include breakfast and wifi. Which do you prefer?" "system: The 3-star Glorious Cloak Inn is near downtown, close to museums and theaters. The 5-star Lunar Hotel is also centrally located, near parks and shopping areas."
```

## EMMM Framework - Further Evaluation from Paper
- Evaluate AOPC of DA-based semi-global aggregation: [framework_aopc.py](experiments/framework_aopc.py)
- Evaluate time complexity of the framework: [demo_timeComplexity.py](experiments/demo_timeComplexity.py)

## Citation
If you use the code, datasets or pre-trained models in your work, please cite the accompanying paper:
```bibtext
@misc{yuan2025emmm,
      title={EMMM, Explain Me My Model! Explainable Machine Generated Text Detection in Dialogues}, 
      author={Angela Yifei Yuan and Haoyi Li and Soyeon Caren Han and Christopher Leckie},
      year={2025}}
```

## Acknowledgements
This work was supported in part by the Australian Research Council Centre of Excellence for Automated Decision-Making and Society, and by the Australian Internet Observatory, which is co-funded by the Australian Research Data Commons (ARDC) through the HASS and Indigenous Research Data Commons.

## License
This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.
