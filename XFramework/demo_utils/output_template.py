from string import Template
import re
import ast

####### HTML TEMPLATE ########

def utt_extractor(role, dia):
    """
    input:
    role: extract user or system utterance
    dia: the input dialogue, should be a string with user and system utt in each line
    output:
    a list of utts
    """
    utt = []
    utterance_lines = dia.splitlines()
    for line in utterance_lines:
        if line.startswith(role + ":"):
            utt.append(line)
    return utt

def score_extractor(explain_df, dia_no, label):
    score = []
    rows = explain_df[explain_df["dia_turn_label"].apply(lambda x: x[0] == dia_no and x[2] == label)]
    for explanation in rows["explanation"]:
        score.append(explanation)
    return score

def dia_lv_score_extractor(explain_df, dia_no, label):
    score = []
    rows = explain_df[explain_df["dia_turn_label"].apply(lambda x: x[0] == dia_no and x[2] == label)]
    for i in range(len(rows["top_explanation"])):
        score.append([rows["top_explanation"][i], rows["dia_turn_label"][i][1]])
    return score

def percentage_cal(explain_df, role, threshold):
    count = 0
    sorted_explain = sorted(explain_df,key=lambda x: x[1], reverse=True)
    for j in sorted_explain:
        if role == "AI":
            if j[1] > threshold:
                count += 1
        else:
            if j[1] <= threshold:
                count += 1
    return round((count/len(sorted_explain))*100, 2)
    
def important_token_extractor(explain_df, topk, label):
    """
    output: token, position name (beginning, middle, end), important score
    """
    if label == 1:
        sorted_explain = sorted(explain_df,key=lambda x: x[1], reverse=True)
    elif label == 0:
        sorted_explain = sorted(explain_df,key=lambda x: x[1], reverse=False)
    else:
        return "you entered an invalid label"
    num_of_token_threshold = len(sorted_explain)//3
    j = sorted_explain[topk - 1]
    token_pos, important_score = j
    token = token_pos[0].split(" ")[0]
    position = int(token_pos[0].split(" ")[1])

    # describe the position of the token
    if position <= num_of_token_threshold:
        position_name = "beginning"
    elif position > num_of_token_threshold*2:
        position_name = "end"
    else:
        position_name = "middle"

    # find previous token
    if position != 0:
        for i in sorted_explain:
            if int(i[0][0].split(" ")[1]) == position - 1:
                prev_token = i[0][0].split(" ")[0]
    else:
        prev_token = "" # remeber to fix this

    return token, position, position_name, important_score, prev_token

def split_sentences_by_tokens(tokens):
    sentences = []
    current = []

    abbreviations = {"St.", "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Mt."} 

    for i, token in enumerate(tokens):
        current.append(token)

        if token in {".", "?", "!"}:
            # Check previous token to see if it's an abbreviation
            prev_token = tokens[i - 1] if i > 0 else ""
            possible_abbr = f"{prev_token}{token}"

            # If it's an abbreviation or followed by lowercase, do not split
            if possible_abbr in abbreviations:
                continue
            elif i + 1 < len(tokens) and tokens[i + 1][0].islower():
                continue
            else:
                sentences.append(current)
                current = []

    if current:
        sentences.append(current)

    return sentences


def get_sentence_index_from_token_pos(token_pos, sentence_list):
    index_counter = 0
    for i, sent in enumerate(sentence_list):
        if token_pos < index_counter + len(sent):
            return i + 1, sent  # 1-based index
        index_counter += len(sent)
    return None, None

def analyze_token_sentences(score_list, utterance):
    tokens = utterance.replace(",", " ,").replace(".", " .").replace("?", " ?").replace("'", " '").split()
    sentence_list = split_sentences_by_tokens(tokens)

    results = []
    for (token_str,), score in score_list:
        token, idx_str = token_str.rsplit(" ", 1)
        token_pos = int(idx_str)
        sent_num, sent_tokens = get_sentence_index_from_token_pos(token_pos, sentence_list)
        results.append({
            "token": token,
            "position": token_pos,
            "score": score,
            "sentence_number": sent_num,
            "sentence_tokens": sent_tokens
        })
    
    return results

def get_sentence_number_for_token(results, target_token, target_position):
    """
    Finds the sentence number of the given target token from the result list.
    If multiple instances exist, returns the first one.
    """
    for entry in results:
        if entry["token"] == target_token and entry["position"] == target_position:
            return entry["sentence_number"]
    return None

def get_sentence_from_token(results, target_token, target_position):
    for entry in results:
        if entry["token"] == target_token and entry["position"] == target_position:
            return " ".join(entry["sentence_tokens"])
    return None

def get_sentence_pos(results, num_sent):
    sentence_numbers = [entry["sentence_number"] for entry in results]
    sent_num_threshold = max(sentence_numbers) // 3
    if num_sent <= sent_num_threshold:
        sentence_pos = "beginning"
    elif num_sent > sent_num_threshold*2:
        sentence_pos = "end"
    else:
        sentence_pos = "middle"
    return sentence_pos

def get_act_ai_rate(explain_df, role, threshold): # assume the value larger than the threshold is ai
    """
    output: how many dia act in this utterance, percentage of ai like dialogue act
    """
    count = 0
    sorted_explain = sorted(explain_df,key=lambda x: x[1], reverse=True)
    n_acts = len(sorted_explain)
    for j in sorted_explain:
        if role == "AI":
            if j[1] > threshold:
                count += 1
        else:
            if j[1] <= threshold:
                count += 1
    return count, n_acts, round(count/n_acts*100, 2) if n_acts>0 else 0
    

def get_act_info(explain_df, topk):
    sorted_explain = sorted(explain_df,key=lambda x: x[1], reverse=True)
    input_str = sorted_explain[topk - 1][0][0]
    match = re.search(r"\[(.*?)\]", input_str)
    if match:
        items = [item.strip().strip("'") for item in match.group(1).split(",")]
        act_into_list = [i for i in items[:3]]
        return act_into_list[0], act_into_list[1], act_into_list[2]
    return []


def format_dialogue_acts(da_list):
    if not da_list:
        return "<p>No AI-like dialogue acts detected.</p>"

    bullet_sections = []

    for entry in da_list:
        if len(entry) != 2:
            continue 

        act_list, utter_index = entry
        act_lines = []

        for (raw_str,), score in act_list:
            try:
                da_part, _ = raw_str.rsplit(" ", 1)
                da_components = ast.literal_eval(da_part)
                act_type = da_components[0]
                domain = da_components[1]
                slot = da_components[2]
                value = da_components[3] if da_components[3] else "[unspecified]"
                readable = f"{act_type.title()} - {domain.title()} - {slot.replace('_', ' ').title()} = {value}"
                act_lines.append(f"<li>{readable} (score: {score:.2f})</li>")
            except Exception:
                act_lines.append(f"<li>Unparsed: {raw_str} (score: {score:.2f})</li>")

        if act_lines:
            section = f"<strong>Utterance {utter_index + 1}:</strong>\n<ul>\n{''.join(act_lines)}</ul>"
            bullet_sections.append(section)

    return "<br>".join(bullet_sections) if bullet_sections else "<p>No AI-like dialogue acts detected.</p>"


BEGINNING = """During detection, the model gives each token and action a score that shows how much it influenced the prediction.
If the utterance is detected as human-generated, the scores are usually at or below the threshold (shown in <span style="color:blue; font-weight:bold;">blue</span> or no color). If it's detected as AI-generated, the scores are generally above the threshold (shown in <span style="color:red; font-weight:bold;">red</span>)."""

# normal format
OUTPUT_TEMPLATE_TOKEN = Template( """
<h2>Token-Based Explanation</h2> The model considers "$utterance" likely to be $output_label-generated.
This is because $percentage% of its tokens exhibit patterns typically found in $output_label responses.
For instance, in sentence $num_sent, the token "$token" appears in the $position, immediately following "$prev_token" — a phrase structure commonly observed in $output_label communication.
Such usage patterns contribute to the model's decision, as they reflect stylistic or syntactic choices characteristic of $output_label utterances.
""")

# normal format but the ai/human like word is the first word
OUTPUT_TEMPLATE_TOKEN_BEG = Template( """  
<h2>Token-Based Explanation</h2> The model considers "$utterance" likely to be $output_label-generated.
This is because $percentage% of its tokens exhibit patterns typically found in $output_label responses.
For instance, in sentence $num_sent, the token "$token" appears in the $position — a sentence-starting word commonly observed in $output_label communication.
""")

# when the percentage is not over 50%
OUTPUT_TEMPLATE_TOKEN_50 = Template( """   
<h2>Token-Based Explanation</h2> The model classified this utterance as $output_label based on the patterns it has learned from previous examples. Specifically, it noticed that $number of the words or phrases in this sentence closely resemble those typically used in $output_label responses.
For instance, in sentence $num_sent, the phrase '$prev_token $token' appears — this is a phrasing pattern often found in $output_label utterances due to its tone and structured syntax.
These subtle linguistic choices help the model determine the likely origin of the sentence.
""")


# act normal template
OUTPUT_TEMPLATE_ACT = Template( """
<h2>DA-Based Explanation</h2> In addition, the utterance involves $act_num dialogue acts, with $act_percentage% of them classified as $output_label-like.
This includes patterns such as how the utterance $intent $slot of $domain, which aligns with common behaviors observed in $output_label responses.
Given that a majority of the dialogue acts reflect $output_label characteristics, the model considers this utterance to be $output_label-generated.                       
""")

# when the percentage is not over 50%
OUTPUT_TEMPLATE_ACT_50 = Template( """
<h2>DA-Based Explanation</h2> Although the utterance involves $act_num dialogue acts, and only $act_num_like of them resemble those typically found in $output_label responses, the model still finds these instances notable.
For example, the utterance $intent $slot of $domain, which reflects a phrasing pattern or structure commonly seen in $output_label communication.
These distinctive patterns, even if limited in number, are strong indicators of $output_label generation. Therefore, the model considers this utterance likely to be $output_label-generated.                       
""")

explained_line_num = 5


LABEL2NAME = ["human", "AI"]
TOPK = 1 # how many examples we want
THRESHOLD = 0



def generate_html_demo(user_utterances, system_utterances, output_texts, token_scores, pic_path, threshold, detections, da_x, diaLv_detection):
    html = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Interactive Dialogue Page</title>
  <style>
    body {
      margin: 0;
      font-family: Arial, sans-serif;
      background-color: white;
      display: flex;
      height: 100vh;
    }
    .dialogue, .explanation {
      padding: 20px;
      width: 50%;
      overflow-y: auto;
    }
    .dialogue {
      border-right: 2px dashed black;
    }
    .utterance {
      padding: 10px;
      margin: 5px 0;
      cursor: pointer;
      text-align: left;
      transition: transform 0.2s ease-in-out;
    }
    .utterance:hover {
    transform: scale(1.03);
    z-index: 10;
    position: relative;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1); /* optional soft shadow */
    }
    .user {
      background-color: #b1ff9c;
    }
    .system {
      background-color: #e0ffff;
    }
    .utterance.highlight {
      outline: 4px solid #ff4500;
      font-weight: bold;
    }
    .explanation {
      font-size: 16px;
    }
    .explanation .explanation-block {
    opacity: 0;
    transform: translateY(20px);
    max-height: 0;
    overflow: hidden;
    transition: opacity 0.4s ease-out, transform 0.4s ease-out, max-height 0.4s ease-out;
    pointer-events: none;
    }
    .explanation .explanation-block.active {
    opacity: 1;
    transform: translateY(0);
    max-height: 2000px; /* big enough to fit your content */
    pointer-events: auto;
    }
    .token {
      padding: 2px 4px;
      border-radius: 4px;
    }
    .back-button {
    background-color: #f8f8f8;
    border: 1px solid #ccc;
    padding: 5px 10px;
    margin-bottom: 12px;
    font-size: 14px;
    cursor: pointer;
    border-radius: 4px;
    }

    .back-button:hover {
    background-color: #8b8c8b;
    }
  </style>
</head>
<body>

<div class="dialogue">
  <h1>Chat History</h1>
"""
    
    max_len = max(len(user_utterances), len(system_utterances))

    for i in range(max_len):
        if i < len(user_utterances):
            prediction = 1 if detections[i] > 0 else 0
            tokens = user_utterances[i].split()
            score_entries = token_scores[i] if i < len(token_scores) else []
            
            # Build a dictionary of {position: (token, score)} for score > 0
            score_entries = token_scores[i] if i < len(token_scores) else []

            # Parse token list into (index, token_text, score)
            parsed_tokens = []
            for (tok_str,), score in score_entries:
                parts = tok_str.rsplit(" ", 1)
                if len(parts) == 2:
                    token_text, idx_str = parts
                    try:
                        idx = int(idx_str)
                        parsed_tokens.append((idx, token_text, score))
                    except ValueError:
                        continue

            # Sort tokens by index
            parsed_tokens.sort(key=lambda x: x[0])

            # Normalize scores for tokens with score > 0
            if prediction == 1:
                valid_scores = [s for _, _, s in parsed_tokens if s > 0]
                max_score = max(valid_scores) if valid_scores else 1.0
            # Normalize scores for tokens with score <= 0
            if prediction == 0:
                valid_scores = [s for _, _, s in parsed_tokens if s <= 0]
                max_score = min(valid_scores) if valid_scores else 1.0

            # box top3 tokens
            boxed_tokens = set()
            if prediction == 1:
                # Top 3 highest scoring tokens > threshold
                top_tokens = sorted(
                    [t for t in parsed_tokens if t[2] > threshold],
                    key=lambda x: -x[2]
                )[:3]
            elif prediction == 0:
                # Top 3 lowest scoring tokens <= threshold
                top_tokens = sorted(
                    [t for t in parsed_tokens if t[2] <= threshold],
                    key=lambda x: x[2]
                )[:3]
            boxed_tokens = {(t[0], t[1]) for t in top_tokens} 

            # Build token HTML without re-splitting anything
            plain_token_html = ""
            highlighted_token_html = ""

            for idx, token_text, score in parsed_tokens:
                alpha = score / max_score if max_score != 0 else 0  # normalize

                # Only apply background color to highlighted version
                if score > threshold:
                    color = f"rgba(255, 0, 0, {abs(alpha):.2f})"  # Red
                else:
                    color = f"rgba(0, 0, 255, {abs(alpha):.2f})"  # Blue

                # Black border for top tokens (both versions)
                border_style = 'border:3px solid red; padding:0.5px 0.5px;' if (idx, token_text) in boxed_tokens else ''

                # Highlighted (with background color)
                highlighted_token_html += (
                    f'<span class="token" data-color="{color}" '
                    f'style="background-color:{color}; {border_style}">{token_text}</span> '
                )

                # Plain (no background color)
                plain_token_html += (
                    f'<span class="token" style="{border_style}">{token_text}</span> '
                )


            # original_text = user_utterances[i]

            html += f'''  <div class="utterance user" data-id="{i + 1}">
                <strong></strong> 
                <span class="plain-text">user: {plain_token_html.strip()}</span>
                <span class="highlighted-text" style="display:none;">user: {highlighted_token_html.strip()}</span>
            </div>\n'''


        if i < len(system_utterances):
            html += f'  <div class="utterance system">{system_utterances[i]}</div>\n'

    html += '<div id="legend" style="display:none; margin-top: 20px; font-style: italic; color: #333;"> red means AI, blue means human</div>'
    
    html += '</div>\n<div class="explanation">\n  <h1>Explanation</h1>\n'

    # dialogue-level explanation
    if diaLv_detection:
        encoded_diaLv = diaLv_detection
        diaLv_img_html = f'''
            <div style="text-align: center; margin-top: 10px;">
                <img src="data:image/png;base64,{encoded_diaLv}" alt="Dialogue-level AI Detection Graph"
                    style="width: 100%; max-width: 600px; height: auto; border: 2px solid black; box-sizing: border-box;">
            </div>
        '''
    else:
        diaLv_img_html = "<p><em>Dialogue-level detection image not found.</em></p>"

    da_x_html = format_dialogue_acts(da_x)
    html += f'''<div id="dialogue-level-explanation" class="explanation-block active">
        <p>Select an utterance from the left panel to see utterance level explanation.</p>
        <h2>Dialogue-Level AI Detection Results</h2>
        {diaLv_img_html}
        <p>The line graph illustrates the AI detection results as the dialogue advances with each new user utterance. The detection model bases its predictions solely on the boxed tokens and the actions listed below.</p>
        <h4> Dialogue Acts Feature Used</h4>
        {da_x_html}
        </div>\n'''

    for i, explanation in enumerate(output_texts, 1):
        explanation_html = explanation.replace("\n", "<br>")
        html += f'<div class="explanation-block" data-id="{i}">\n'
        html += '<button class="back-button" onclick="showDialogueLevel()">⬅️ Dialogue-level Explanation</button><br><br>'
        html += f'{explanation_html}\n'

        image_path = pic_path[i - 1]
        if image_path and image_path['ai'] is not None and image_path['human'] is not None:
            encoded_ai = image_path['ai']
            encoded_human = image_path['human']

            html += '<h2>Aggregating Language Use</h2>\n'
            html += '<p style="text-align: center; font-style: italic; color: #555;">The utterance performs actions.<br>The following word clouds represent common language patterns contrubuting toward AI and Human classification respectively, when expressing these actions.</p>\n'
            
            html += '<div style="display: flex; justify-content: space-between; gap: 10px; margin-top: 20px;">\n'

            # AI image
            html += '<div style="width: 48%; text-align: center;">\n'
            html += '<h3>AI</h3>\n'
            html += f'<img src="data:image/png;base64,{encoded_ai}" alt="AI word cloud" style="width: 100%; height: auto; border: 2px solid black; box-sizing: border-box;">\n'
            html += '</div>\n'

            # Human image
            html += '<div style="width: 48%; text-align: center;">\n'
            html += '<h3>Human</h3>\n'
            html += f'<img src="data:image/png;base64,{encoded_human}" alt="Human word cloud" style="width: 100%; height: auto; border: 2px solid black; box-sizing: border-box;">\n'
            html += '</div>\n'

            html += '</div>\n'

            
        html += '</div>\n'

    
    html += """
</div>

<script>
  const userUtterances = document.querySelectorAll('.user');
  const explanations = document.querySelectorAll('.explanation .explanation-block');

  userUtterances.forEach(utterance => {
  utterance.addEventListener('click', () => {
    userUtterances.forEach(u => {
      u.classList.remove('highlight');
      u.querySelector('.plain-text').style.display = 'inline';
      u.querySelector('.highlighted-text').style.display = 'none';
    });
    explanations.forEach(p => p.classList.remove('active'));

    utterance.classList.add('highlight');

    const plain = utterance.querySelector('.plain-text');
    const highlighted = utterance.querySelector('.highlighted-text');
    if (plain && highlighted) {
      plain.style.display = 'none';
      highlighted.style.display = 'inline';

      // Apply background colors to tokens
      const tokens = highlighted.querySelectorAll('.token');
      tokens.forEach(token => {
        const color = token.getAttribute('data-color');
        token.style.backgroundColor = color;
      });
    }

    const id = utterance.getAttribute('data-id');
    const explanation = document.querySelector(`.explanation .explanation-block[data-id="${id}"]`);
    if (explanation) explanation.classList.add('active');
  });
});

function showDialogueLevel() {
  // Un-highlight all utterances
  document.querySelectorAll('.utterance.user').forEach(u => {
    u.classList.remove('highlight');
    const plain = u.querySelector('.plain-text');
    const highlighted = u.querySelector('.highlighted-text');
    if (plain && highlighted) {
      plain.style.display = 'inline';
      highlighted.style.display = 'none';
    }
  });

  // Hide all utterance-level explanations
  document.querySelectorAll('.explanation .explanation-block').forEach(e => {
    e.classList.remove('active');
  });

  // Show dialogue-level explanation
  const dialogueExplanation = document.getElementById('dialogue-level-explanation');
  if (dialogueExplanation) {
    dialogueExplanation.classList.add('active');
  }
}
</script>

</body>
</html>
"""
    return html

def output_template_demo(slot_info, utt_x, da_x, dia, pic_path, utt_preds, da_preds, diaLv_detection, threshold = THRESHOLD, file_path=None):
    explanation_output_text = []
    explain_df_token = utt_x
    
    target_dia, _, ground_truth_label = explain_df_token.iloc[0]['dia_turn_label']
    user_utt = utt_extractor("user", dia)
    for index, row in explain_df_token.iterrows():
        utt_pred_label = 1 if utt_preds[index] >= 0.5 else 0
        da_pred_label  = 1 if da_preds[index]  >= 0.5 else 0
        dia_num, utt_num, ground_truth_label = row["dia_turn_label"]
        token, position, position_name, important_score, prev_token = important_token_extractor(row["explanation"], TOPK, utt_pred_label)
        sent_token_match = analyze_token_sentences(row["explanation"], user_utt[utt_num]) # a dictionary
        sent_num = get_sentence_number_for_token(sent_token_match, token, position) # the matched sentence for a token
        percentage = percentage_cal(row["explanation"], LABEL2NAME[utt_pred_label], threshold) # how many words are classified as ai

        if percentage >= 50:
            if prev_token != "":
                output_text_token = OUTPUT_TEMPLATE_TOKEN.safe_substitute(
                    utterance = user_utt[utt_num], output_label = LABEL2NAME[utt_pred_label], percentage = percentage,
                    num_sent = sent_num, token = token, 
                    position = position_name, prev_token = prev_token, 
                )
            else:
                output_text_token = OUTPUT_TEMPLATE_TOKEN_BEG.safe_substitute(
                    utterance = user_utt[utt_num], output_label = LABEL2NAME[utt_pred_label], percentage = percentage,
                    num_sent = sent_num, token = token, 
                    position = position_name, 
                )
        else:
            output_text_token = OUTPUT_TEMPLATE_TOKEN_50.safe_substitute(
                utterance = user_utt[utt_num], output_label = LABEL2NAME[utt_pred_label], percentage = percentage,
                num_sent = sent_num, token = token, 
                position = position_name, prev_token = prev_token, 
            )

        # only consider the matched act of the current line 
        explain_df_act = da_x[da_x["dia_turn_label"].apply(lambda x: x[0] == dia_num and x[1] == utt_num and x[2] == ground_truth_label)]
        for index, row in explain_df_act.iterrows():
            act_num, total_acts, act_percentage = get_act_ai_rate(row["explanation"], LABEL2NAME[da_pred_label], threshold)
            if act_num > 0:
                intent, domain, slot = get_act_info(row["explanation"], TOPK)
                if len(slot_info[slot_info["index"] == slot])>0:
                    slot = slot_info[slot_info["index"] == slot]["slots"].values[0]["description"]
                if act_percentage >= 50:
                    output_text_act = OUTPUT_TEMPLATE_ACT.safe_substitute(
                        act_num = total_acts,
                        act_percentage = act_percentage,
                        output_label = LABEL2NAME[da_pred_label],
                        intent = intent,
                        domain = domain,
                        slot = slot
                        )
                else:
                    output_text_act = OUTPUT_TEMPLATE_ACT_50.safe_substitute(
                        act_num = total_acts,
                        act_num_like = act_num,
                        output_label = LABEL2NAME[da_pred_label],
                        intent = intent,
                        domain = domain,
                        slot = slot
                        )
            elif act_num == 0 and total_acts > 0:
                output_text_act = f"<h2>DA-Based Explanation</h2>In addition, it is classified as {LABEL2NAME[da_pred_label]} because there is no strongly {LABEL2NAME[1-da_pred_label]}-like act, and all the scores of the acts in this utterance are very close to the threshold. Therefore, the model considers this utterance to be {LABEL2NAME[da_pred_label]}-generated." # if there is no ai/human acts, we do not show act explanation
                break
            elif total_acts == 0: 
                output_text_act = f"<h2>DA-Based Explanation</h2>In addition, no actions are extracted from the utterance, and the act-based detection model classifies such empty actions as {LABEL2NAME[da_pred_label]}." # if there is acts, we do not show act explanation - inform baseline model behavior
                break



        output_common_sense = ""
        output_text = BEGINNING + output_text_token + output_common_sense + output_text_act
        explanation_output_text.append(output_text)

    user_utterances = utt_extractor("user", dia)
    system_utterances = utt_extractor("system", dia)
    user_token_scores = score_extractor(utt_x, target_dia, ground_truth_label)
    da_scores = dia_lv_score_extractor(da_x, target_dia, ground_truth_label)

    html_content = generate_html_demo(user_utterances, system_utterances, explanation_output_text, user_token_scores, pic_path, 0, utt_preds, da_scores, diaLv_detection) # the second last input is threshold for importance score

    # Save to file
    if file_path is None: 
        file_path = f"output_template/html/dialogue_demo_page.html"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)