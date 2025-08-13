import json

# masking slots supporting retrieve_slots function, where keys are slots
MASK_SLOTS = {'name': '[name]', 
              'ref': '[ref]', 
              'postcode': '[postcode]', 
              'phone': '[phone]', 
              'address': '[address]'}

# masking information to be used in prompt directly
HOTEL_MASK_INFO = {'hotel name': '[name]', 
              'booking reference number': '[ref]', 
              'hotel postcode': '[postcode]', 
              'phone number': '[phone]', 
              'address': '[address]'}


def load_json(json_path): 
    with open(json_path) as f:
        json_data = json.loads(f.read())
    return json_data

def retrieve_slots(ontology, domains=[], mask_slots=dict()): 
    slots = dict()
    domains = domains if domains else ontology['domains'].keys() # empty domains default to all domains
    for domain, domain_val in ontology['domains'].items(): 
        if domain in domains: 
            domain_slots = dict()
            for slot, slot_val in domain_val['slots'].items(): 
                domain_slots[slot] = dict()
                domain_slots[slot]['description'] = slot_val['description']
                if slot_val.get('possible_values',[]): 
                    domain_slots[slot]['possible_values'] = str([v for v in slot_val['possible_values'] if v!='dontcare'])
                if slot in mask_slots: 
                    domain_slots[slot]['mask_token'] = mask_slots[slot] 
            slots[domain] = domain_slots
    return json.dumps(slots, indent=4) # convert to string format


def repeat_until(operation, stop_condition, max_iterations=5, default_result='reached max iterations', log=False, operation_btw_repeat = None): 
    for i in range(max_iterations): 
        result = operation()
        if stop_condition(result): 
            if log: print(f'Succeed on the iteration {i+1}.')
            return result
        if operation_btw_repeat is not None: 
            operation_btw_repeat() # an operation between repeats of performing the main operation, e.g. time.sleep(4)
    if log: print(f"Failed after {max_iterations} iterations.")
    if default_result == 'latest result': 
        return result
    return default_result

import string
def english_char_only(text): 
    base_chars = string.ascii_letters + string.digits + string.punctuation
    extra_chars = 'áéíóúàèüöäñçÁÉÍÓÚÀÈÜÖÄÑÇ—€£ãõâêîôûÂÊÎÔÛ '
    allowed_chars = set(base_chars + extra_chars)
    return all(char in allowed_chars for char in text)

ROLE_OPPONENTS = {'user':'system', 'system':'user'}
def parse_utterance_response(response, role):
    assert role in ROLE_OPPONENTS, f"Invalid role: {role}."

    # does not contain utterance
    if response is None or f"{role}:" not in response: 
        return None 
    
    # adjust format
    if f"{role}: " not in response: 
        response = response.replace(f"{role}:", f"{role}: ")
    
    # check if it only contains one utterance of the role
    extracted_response = response.split(f"{role}: ")[1:]
    if len(extracted_response) > 1: 
        return None
    if f"{ROLE_OPPONENTS[role]}:" in extracted_response[0]: 
        return None
    formatted_response = extracted_response[0].replace('\n', ' ').strip()
    formatted_response = formatted_response.replace('‘', "'").replace('’', "'")
    if not english_char_only(formatted_response): 
        return None
    return f"{role}: {formatted_response}"


import re
def remove_end_stop(text):
    # Removes any END or STOP token from the text. Returns processed text, and whether it hass been modified
    # pattern = r"[^\w\s]*\b(END|STOP)\b[^\w\s]*"
    pattern = r"[^\w\s]+\b(END|STOP)\b[^\w\s]+"
    new_text, num_subs = re.subn(pattern, "", text, flags=re.IGNORECASE)
    return new_text.strip(), num_subs > 0


def collapse_dialogue_lines(text):
    # any utterance that spans over multiple lines would be collapsed into a single line
    lines = text.splitlines()
    merged_lines = []
    current_line = ""

    for line in lines:
        if line.startswith("user:") or line.startswith("system:"):
            if current_line:
                merged_lines.append(current_line.strip())
            current_line = line
        else:
            current_line += " " + line.strip()

    if current_line:
        merged_lines.append(current_line.strip())

    return "\n".join(merged_lines)
