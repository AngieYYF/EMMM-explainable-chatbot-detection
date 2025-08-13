from LLM.E2E_Frame import e2e_convo_dataset, e2e_user_prompt_domain_template, e2e_system_prompt_domain_template
from LLM.LLM_chatbot import LLM
from LLM.utils import load_json
import json
import time
import os
import argparse
import pandas as pd

def main():
    # arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("-system_llm", help="name of llm model serving the system role.",
                        type=str, default="huggingface/Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("-user_llm", help="name of llm model serving the user role.",
                        type=str, default="huggingface/Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("-admin_llm", help="name of llm model serving the admin role.",
                        type=str, default="huggingface/Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("-bona_fide", help="location of the bona fide dataset.", type=str)
    parser.add_argument("-goal_col", help="column in the dataset containing user goal.", type=str)
    parser.add_argument("-ontology", help="location of the ontology file.", type=str)
    parser.add_argument("-domains", nargs="+", help="ontology domains used in prompt.", type=str)
    parser.add_argument("-example_dialogues", nargs="+", help="dialogue ids serving as demonstrations.", type=int)
    parser.add_argument("-n_generate_dialogues", help="number of dialogues to generate.", type=int)
    parser.add_argument("-save_path", help="location to save the generated dataset.", type=str)
    parser.add_argument("-max_turns", type=int, default=15 , help="maximum number of conversation turns." )
    parser.add_argument("-utterance_regenerate", type=int, default=5 , help="maximum number of repeats for each utterance generation." )
    parser.add_argument("-dialogue_regenerate", type=int, default=3, help="maximum number of runs to regenerate unsuccessful dialogues.")
    
    args = parser.parse_args()

    # Load bona fide file (containing dia_no and goal)
    if not os.path.exists(args.bona_fide):
        print(f'bona fide file does not exist: {os.path.abspath(args.bona_fide)}')
        return
    original_dataset = pd.read_csv(args.bona_fide)
    assert 'dia_no' in original_dataset.columns and args.goal_col in original_dataset.columns, f"Bona fide dataset missing either dia_no or {args.goal_col}."
    original_dataset = original_dataset.sort_values(by='dia_no', ignore_index=True)

    # set up prompts
    # ontology
    if not os.path.exists(args.ontology):
        print(f'ontology file does not exist: {args.ontology}')
        return
    ontology = load_json(args.ontology)
    # example dialogues (goals and dialogues)
    example_dialogues = [f'EXAMPLE {i}\nUser Goal:\n'+ 
                            json.dumps(goal_outcome.replace('$','$$').split(' <outcome> ')[0].split(' <sep> '),indent=2) + 
                            '\nDialogue:\n' + 
                            dia.replace('$','$$') 
                        for i, (dia, goal_outcome) 
                        in enumerate(
                            zip(
                                list(original_dataset['dia'][args.example_dialogues]), 
                                list(original_dataset[args.goal_col][args.example_dialogues])
                                )
                            )
                        ]


    user_prompt = e2e_user_prompt_domain_template(example_dialogues, ontology, args.domains, [])
    system_prompt = e2e_system_prompt_domain_template(example_dialogues, ontology, args.domains, [])
    
    print("********** User Prompt **********")
    print(user_prompt.safe_substitute(), flush=True)

    print("********** System Prompt **********")
    print(system_prompt.safe_substitute(), flush=True)

    # set up LLMs
    try: 
        LLMS = {}
        for llm_name in [args.user_llm, args.system_llm, args.admin_llm]:
            if llm_name not in LLMS: 
                 LLMS[llm_name] = LLM({'name': llm_name}, '')

        user = LLMS[args.user_llm]
        system = LLMS[args.system_llm]
        admin = LLMS[args.admin_llm]

    except Exception as e:
        print(f"Error setting up LLM: {e}")

    # generate dataset
    try: 
        # generate e2e dataset
        e2e_convo_dataset(user, system, admin, 
                        original_dataset, args.goal_col, 
                        user_prompt, system_prompt, 
                        args.n_generate_dialogues, 
                        args.save_path, 
                        args.max_turns, args.utterance_regenerate, args.dialogue_regenerate)
    except Exception as e:
        print(f"Error generating dataset: {e}")
        

if __name__ == "__main__":
    main()




