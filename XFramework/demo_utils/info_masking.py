from string import Template
from LLM.LLM_chatbot import LLM
from LLM.utils import repeat_until

frames_info_masking_prompt = Template("""Mask the following information in the dialogue: hotel name as [hotel name]. You should generate the entire original dialogue with the specified masks applied. Wrap the final dialogue output with <DIA> and </DIA> tags.

## Demonstration
Dialogue: 
user: Hello there i am looking to go on a vacation with my family to Gotham City, can you help me?
system: when  would you like to travel and how many people will you be?
user: Not sure when we want to leave, but we are 12 kids and 5 adults
system: do you have a budget?
user: yes i do, it is around $$2200
system: where will you be travelling from?
user: We are from Neverland
system: We have nothing available leaving from Neverland, are you able to depart from another city?
user: we can depart from Toronto
system: Gotham City is not a destination we travel to. Are you interested in any other destinations?
user: hmm what options would i have out of Toronto?
system: Would you be interested in Calgary?
user: that would be nice
system: I have a departure from Toronto to Calgary  on August 17 returning on August 24. Does that work for you?
user: No sorry i was planning on getting a tan this vacation. Thanks, is there anywhere else? Thank you is there anywhere else?
system: How about a trip to Dominican Republic? it is quite cheap at this time of year.
user: What options do you have available for me?
system: I have the el Mar hotel available in Punta Cana this week.
user: That sounds interesting. how many stars does the hotel have?
system: 2.5 stars. You would have to fly out of Melbourne though
user: Ahh...maybe another time

Output: 
<DIA>user: Hello there i am looking to go on a vacation with my family to Gotham City, can you help me?
system: when  would you like to travel and how many people will you be?
user: Not sure when we want to leave, but we are 12 kids and 5 adults
system: do you have a budget?
user: yes i do, it is around $$2200
system: where will you be travelling from?
user: We are from Neverland
system: We have nothing available leaving from Neverland, are you able to depart from another city?
user: we can depart from Toronto
system: Gotham City is not a destination we travel to. Are you interested in any other destinations?
user: hmm what options would i have out of Toronto?
system: Would you be interested in Calgary?
user: that would be nice
system: I have a departure from Toronto to Calgary  on August 17 returning on August 24. Does that work for you?
user: No sorry i was planning on getting a tan this vacation. Thanks, is there anywhere else? Thank you is there anywhere else?
system: How about a trip to Dominican Republic? it is quite cheap at this time of year.
user: What options do you have available for me?
system: I have the [hotel name] available in Punta Cana this week.
user: That sounds interesting. how many stars does the hotel have?
system: 2.5 stars. You would have to fly out of Melbourne though
user: Ahh...maybe another time</DIA>


## Here is the dialogue you need to mask:
$dialogue

Output: 
""")


mwoz_info_masking_prompt = Template("""Mask the following information in the dialogue: hotel name as [hotel name], phone number as [phone number], postcode as [postcode], address of hotel as [address], reference number as [ref]. You should generate the entire original dialogue with the specified masks applied. Wrap the final dialogue output with <DIA> and </DIA> tags.

## Demonstration
Dialogue: 
user: Hello, I am trying to find a hotel located in the west part of town. Any ideas?
system: We have two options available. Would you prefer expensive or cheap?
user: Either would be fine. I would like it to have a five-star rating.
system: I apologize neither is five stars. Would you like to try something else?
user: Is there a 4-star hotel available in the west part of town?
system: There are 2 hotels available. The Huntingdon Marriott Hotel in the expensive price range and the Cambridge Belfry in the cheap price range. Do you have a preference?
user: Sure, let's go with the Cambridge Belfry. Please give me its postcode and address.
system: The post code is cb236bw. The address is Back Lane, Cambourne. Do you want me to book it for you?
user: No, that's fine for now.
system: Their phone is 01954714600 if you choose to make a reservation yourself, or you may contact me again. Thank you!

Output: 
<DIA>user: Hello, I am trying to find a hotel located in the west part of town. Any ideas?
system: We have two options available. Would you prefer expensive or cheap?
user: Either would be fine. I would like it to have a five-star rating.
system: I apologize neither is five stars. Would you like to try something else?
user: Is there a 4-star hotel available in the west part of town?
system: There are 2 hotels available. The [hotel name] in the expensive price range and the [hotel name] in the cheap price range. Do you have a preference?
user: Sure, let's go with the [hotel name]. Please give me its postcode and address.
system: The post code is [postcode]. The address is [address]. Do you want me to book it for you?
user: No, that's fine for now.
system: Their phone is [phone number] if you choose to make a reservation yourself, or you may contact me again. Thank you!</DIA>


## Here is the dialogue you need to mask:
$dialogue

Output: 
""")

class Info_Masker:
    def __init__(self, prompt): 
        llm_model_info = {'name': "huggingface/Qwen/Qwen2.5-7B-Instruct"}
        self.llm_masker = LLM(llm_model_info, '')
        self.prompt = prompt

    def parse_masked_dia(response): 
        if response is None: 
            return None
        start_token, end_token = "<DIA>", "</DIA>"
        start_idx = response.find(start_token)
        end_idx = response.find(end_token)
        if start_idx == -1 or end_idx == -1:
            return None
        response = response[start_idx+len(start_token):end_idx].strip()
        if response == "": 
            return None
        return response

    def __call__(self, utt): 
        return repeat_until(
                        # LLM masking information
                        lambda: Info_Masker.parse_masked_dia(
                            self.llm_masker.chat(
                                self.prompt.substitute(dialogue=utt)
                                )
                            ),

                        # check that result is not None and the number of lines returned is the same
                        lambda result: result is not None and len(result.split('\n'))== len(utt.split('\n')),

                        # repeat until previous condition is satisfied, or max_iteration is reached
                        max_iterations=1,

                        # return original utterance if condition is not satisfied
                        default_result=utt,

                        log=False)
