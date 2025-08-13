from string import Template
from LLM.utils import *
from LLM.LLM_chatbot import LLM
import os
import pandas as pd
from tqdm import tqdm
import json

# ======================================================== LLM User Prompt ========================================================
USER_PROMPT_TEMPLATE = Template(
'''Task: Simulate as an user with a particular goal and generate one response to a task oriented dialogue system. Response must start with "user: ". After you achieved all your goals, end the conversation and generate "[END]" token. If you think the system cannot help you or the conversation falls into an infinite loop, generate a "[STOP]" token. The response must be one line only!

The information you can ask for or provide (include everything is not mandatory): 
$ontology_slot_value
Information with “mask_token” specified must be replaced by corresponding token in your response. Do not ask for or provide other information. You do not need to confirm details with the system unless it is ambiguous.

Here are demonstration dialogues unrelated to your own goal:
$demonstrations
Do not copy anything from the demonstration!

Here is your goal: $goal
Move through the goals in sequential order when preceding goals cannot be completed. 
You should end conversation only once a booking is successfully made by the system, or that none of the goals can be satisfied.
Do not generate "END" when requesting a booking.
Do not directly copy from the goal, be creative in generating the user response like a human.
The user response must be within 20 words using natural and fluent English.

Chat history between you and the system:
system: Hello! How can I assist you today?$history
''') 
# the history during e2e convo should have a \n at the start


def e2e_user_prompt_domain_template(demonstrations, ontology, domains, mask_slots=dict()):
    ontology_slot_value = retrieve_slots(ontology, domains, mask_slots)
    demonstrations = "\n\n".join(demonstrations)
    return Template(USER_PROMPT_TEMPLATE.safe_substitute(ontology_slot_value=ontology_slot_value, 
                                                demonstrations=demonstrations))

def e2e_user_prompt_dialogue(prompt_template, goal, history): 
    return prompt_template.safe_substitute(goal=goal, history=history)



# ======================================================== LLM System Prompt ========================================================
SYSTEM_PROMPT_TEMPLATE = Template(
'''Task: Simulate as a task oriented dialogue system and generate one response to a user. Response must start with "system: ". If and only if the user has no more queries or generated "[END]", end the conversation and generate "[END]" token. If you think the conversation falls into an infinite loop, generate a "[STOP]" token.

The information you can ask for or provide (include everything is not mandatory):
$ontology_slot_value
Information with “mask_token” specified must be replaced by corresponding token in your response. Not all information is mandatory, and you do not need to provide information not asked by the user, nor to confirm if they need it. Do not ask for or provide other information. Do not repeat yourself unless asked by the user. You do not need to confirm details with user unless it is ambiguous.

Here are demonstration dialogues:
$demonstrations
Do not copy anything from the demonstration!

Here are the user goals and the outcomes of searching for relevant vacation packages: $goal
Before making suggestions or bookings, check whether the user has specified preference or flexibility on all critical information: location (dst_city, or_city), time (str_date, end_date, duration), number of people (n_adults, n_children), and budget.
Identify any missing critical information based on the chat history: "I need to confirm: <at most 2 missing items, or None if all are provided>"
Then, generate your booking assistant response starting with: "system: "
In the booking assistant response, do not directly copy from the goal or outcome, do not say "I need to confirm", be creative and respond like a human.
The booking assistant response must be within 20 words using natural and fluent English.

Chat history between you and the user:
system: Hello! How can I assist you today?$history
'''
)
# the history during e2e convo should have a \n at the start


def e2e_system_prompt_domain_template(demonstrations, ontology, domains, mask_slots=dict()):
    ontology_slot_value = retrieve_slots(ontology, domains, mask_slots)
    demonstrations = "\n\n".join(demonstrations)
    return Template(SYSTEM_PROMPT_TEMPLATE.safe_substitute(ontology_slot_value=ontology_slot_value, 
                                                demonstrations=demonstrations))

def e2e_system_prompt_dialogue(prompt_template, history): 
    return prompt_template.safe_substitute(history=history)



# ======================================================== LLM Admin Prompt ========================================================
ADMIN_PROMPT_TEMPLATE = Template('''Based on the conversation so far, which goals is the user currently expressing? 
Conversation: $history

Goals: 
$goal

Your final response should be in format of <goal> x </goal>, where x is the index of the goal which the user is currently working on.''')

def parse_admin_response(response):
    if response is None: 
        return None
    start_token, end_token = "<goal>", "</goal>"
    start_idx = response.find(start_token)
    end_idx = response.find(end_token)
    if start_idx == -1 or end_idx == -1:
        return None
    goal = response[start_idx+len(start_token):end_idx].strip()
    if goal == "" or not goal.isdigit(): 
        return None
    return int(goal)


# ======================================================== E2E Conversation Framework ========================================================
def e2e_convo(user:LLM, system:LLM, admin:LLM, goal:str, user_prompt:Template, system_prompt:Template, max_turns:int = 15, utterance_regenerate=3, log=False): 
    history = ""
    finished_conversation = False
    if isinstance(goal, list): 
        goals, outcomes = goal[0], goal[1]
        usr_goal = json.dumps(goals, indent=4)
        sys_goal = [f"goal: {[goals[i]]}\noutcome: {outcomes[i]}" for i in range(len(goals))]
        admin_goal = '\n'.join([f'{i+1}. {g}' for i, g in enumerate(goals)])
    for turn_idx in range(max_turns): 
        # user utterance
        user_response = repeat_until(lambda: parse_utterance_response(user.chat(user_prompt.substitute(goal=usr_goal, history=history)), 'user'), 
                                     lambda x: x is not None, 
                                     max_iterations=utterance_regenerate, 
                                     default_result=None, 
                                     operation_btw_repeat=lambda:print("\n\nRegenerate User Response\n\n", flush=True))
        if user_response is None: 
            if log: 
                print("Cannot generate user response.", flush=True)
            break
        
        user_response, conversation_ended = remove_end_stop(user_response)
        if len(user_response) <= 6: # empty user utterance "user: "
            if log: 
                print(f"Empty user response: {user_response}", flush=True)
            break
        
        if log: 
            print(user_response, flush=True)
        history += '\n' + user_response
        if conversation_ended: 
            finished_conversation = True
            break

        # admin to check goals
        n_goal_expressed = repeat_until(lambda: parse_admin_response(admin.chat(ADMIN_PROMPT_TEMPLATE.substitute(history=history, goal=admin_goal))), 
                                      lambda x: x is not None, 
                                      max_iterations=5, 
                                      default_result=None, 
                                      operation_btw_repeat=lambda:print("\n\nRegenerate Admin Response\n\n", flush=True))
        if n_goal_expressed is None: 
            if log: 
                print("Cannot identify the goal progress.", flush=True)
            break
        if log: 
            print(f"{n_goal_expressed} goals expressed.", flush=True)
        cur_sys_goal = json.dumps(sys_goal[:n_goal_expressed], indent=4)
        
        # system utterance
        system_response = repeat_until(lambda: parse_utterance_response(system.chat(system_prompt.substitute(goal=cur_sys_goal, history=history)), 'system'), 
                                     lambda x: x is not None, 
                                     max_iterations=utterance_regenerate, 
                                     default_result=None, 
                                     operation_btw_repeat=lambda:print("\n\nRegenerate System Response\n\n", flush=True))
        if system_response is None: 
            if log: 
                print("Cannot generate system response.", flush=True)
            break
        
        system_response, conversation_ended = remove_end_stop(system_response)
        if len(system_response) <= 8: # empty system utterance "system: "
            if log: 
                print(f"Empty system response: {system_response}", flush=True)
            break
        
        if log: 
            print(system_response, flush=True)
        history += '\n' + system_response
        if conversation_ended: 
            finished_conversation = True
            break
    
    return history.strip(), finished_conversation

def e2e_convo_dataset(user:LLM, system:LLM, admin:LLM, original_dataset, goal_col, user_prompt:Template, system_prompt:Template, n_generate_dialogues, save_path, max_turns:int = 15, utterance_regenerate=5, dialogue_regenerate=3): 
    # load any existing file
    if os.path.exists(save_path):
        generated_e2e_convo = pd.read_csv(save_path, index_col=0)
        existing_dia_nos = set(generated_e2e_convo.loc[generated_e2e_convo['is_complete']]['dia_no'])
    else:
        generated_e2e_convo = pd.DataFrame(columns=['dia_no', 'goal', 'dia', 'is_complete'])
        existing_dia_nos = set()

    # track number of dialogues to generate
    n_generate_dialogues = min(n_generate_dialogues, len(original_dataset)-len(existing_dia_nos))
    print(f"Generating {n_generate_dialogues} dialogues.", flush=True)
    n_generated_dialogues = 0
    progress_bar = tqdm(total=n_generate_dialogues, desc="Generating Dialogues")

    # iterate through dataset to generate new dialogues
    for i, bona_fide in original_dataset.iterrows():
        dia_no = bona_fide['dia_no']
        if dia_no in existing_dia_nos:
            continue

        goal = [goal_outcome.split(' <sep> ') for goal_outcome in bona_fide[goal_col].split(' <outcome> ')]
        
        synthetic_dialogue, is_complete = repeat_until(
            lambda: e2e_convo(user, system, admin, goal, user_prompt, system_prompt, max_turns, utterance_regenerate, log=True), 
            lambda x: x[1], 
            max_iterations=dialogue_regenerate, 
            default_result="latest result", 
            operation_btw_repeat=lambda:print("\n\nRegenerate Dialogue\n\n", flush=True))
        
        generated_e2e_convo.loc[len(generated_e2e_convo)] = [dia_no, goal, synthetic_dialogue, is_complete]
        generated_e2e_convo.to_csv(save_path)
        
        # track number of dialogues generated
        n_generated_dialogues += 1
        progress_bar.update(1)
        if n_generated_dialogues >= n_generate_dialogues: 
            break