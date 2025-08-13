from string import Template

frame_goal_prompt = Template('''Generate the progression of goals and outcomes for the user in the dialogue. 

**GOAL**: Defines user's requirements for vacation packages, including origin, destination(s), dates, number of travelers, budget, flexibility, and preferences. Each goal should reflect either:
1. The initial request from the user, or
2. An alternative suggested by the user after the system fails to meet the previous goal.

If a goal was unsuccessful, the user either ended the dialogue or continued with an **alternative goal**, which must begin with:  
“If nothing matches your constraints, ...”

Please differentiate between multiple options within the same goal and alternative goals. They are characterised as follows: 
1. Options within the same goal: The user modifies previously specified constraints voluntarily to explore and compare different options, even when the system has already returned packages that match their earlier constraints.
2. Alternative goal: The user modifies constraints as a fallback because the system was unable to find any matching packages with the original constraints. This must start with "If nothing matches your constraints, ...". Alternative goals can also include multiple options within the same goal.

Goal Templates: 
For the initial goal:
<GOAL> Find a vacation between [START DATE] and [END DATE] for [NUM ADULTS] adults and [NUM CHILDREN] kids. You leave from [ORIGIN CITY]. You want to go to [DESTINATION CITY]. You are travelling on a budget and would like to spend at most $$[BUDGET]. </GOAL>
For any subsequent goal:
<ALT_GOAL> If nothing matches your constraints, [describe alternative criteria change like changing dates, destinations, budget, etc.] </ALT_GOAL>

**OUTCOME**: Defines the vacation packages or suggestions the system returned in response to each goal. Include specific package details mentioned in the dialogue (e.g., hotel names, dates, locations, cost, star ratings, amenities, etc.).


### Examples

#### Example 1:
Dialogue:
user: I'd like to book a trip to Atlantis from Caprica on Saturday, August 13, 2016 for 8 adults. I have a tight budget of 1700.
system: Hi...I checked a few options for you, and unfortunately, we do not currently have any trips that meet this criteria.  Would you like to book an alternate travel option?
user: Yes, how about going to Neverland from Caprica on August 13, 2016 for 5 adults. For this trip, my budget would be 1900.
system: I checked the availability for this date and there were no trips available.  Would you like to select some alternate dates?
user: I have no flexibility for dates... but I can leave from Atlantis rather than Caprica. How about that?
system: I checked the availability for that date and there were no trips available.  Would you like to select some alternate dates?
user: I suppose I'll speak with my husband to see if we can choose other dates, and then I'll come back to you.Thanks for your help

Goal Outcome:
<THINK> The user modifies their criterias after the system fails to find any matching package. Thus there are alternative goals. </THINK>

<GOAL> Find a vacation from Saturday, Auguest 13, 2016 for 8 adults. You leave from Caprica. Your destination is Atlantis. You are travelling on a budget and you would like to spend at most $$1700. </GOAL> 
<OUTCOME> No package available. </OUTCOME>

<ALT_GOAL> If nothing matches your constraints, try changing destination to Neverland, reducing to 5 adults, and increasing your budget by $$200. Dates are not flexible. </ALT_GOAL> 
<OUTCOME> No package available. </OUTCOME>

<ALT_GOAL> If nothing matches your constraints, try leaving from Atlantis. </ALT_GOAL> 
<OUTCOME> No package available. </OUTCOME>

<ALT_GOAL> If nothing matches your constraints, then end the conversation. </ALT_GOAL>
<OUTCOME> N/A </OUTCOME>


#### Example 2:
Dialogue:
user: Hi i wanna look into packages to san antonio
system: Hi and where will you be travelling from?
user: belem. hoping to leaving august 17 and go for two weeks
system: do you have a budget in mind? are those dates flexible as well?
user: no budget. dates are preferably the ones i have sent you but could be flexible if we find a great package i suppose
system: And one last thing, how many will be travelling?
user: just me
system: There is the Winter Hill Inn a 2 star hotel in San Antonio from August 18th to August 28, it has free breakfast, wifi and parking. it goes for $$3167, how does that sound
user: ok.. i'll consider that. anything available in san francisco?
system: Unfortunately there is nothing in san francisco
user: ok. what about in Kyoto?
system: the Glorious Bazaar Resort is available it has a 3 star rating, with free breakfast and wifi
user: Is that from the 17-31?
system: I have 17th to the 22nd or the 25th until the 30th
user: ok. one more destination I am wondering about is Cordoba. What can you offer me there?
system: while there are possibilities to cordoba for a 10 day trip but only leaving on August 24th
user: hmm ok.. well, how much would the Cordoba trip cost?
system: it would be at a 5 star hotel for only $$2018
user: wow! I like that option. free breakfast?
system: and free wifi!
user: sweet! ok, i think I want to book that. what is the return date?
system: it leaves on August 24th and returns on September 2nd
user: lets book it!
system: perfect! its all done
user: thanks :slightly_smiling_face:

Goal Outcome:
<THINK> The user explores packages for different destinations while the system has provided one package that meets the previous constraints. Thus there are multiple options within the same goal. </THINK>

<GOAL> Find a vacation from August 17 to August 31 for 1 adult. The dates are flexible. You leave from Belem. You want to go to either San Antonio, San Francisco, Kyoto, or Cordoba. Ask for information about each package. Compare the packages and pick the one you like best. </GOAL>
<OUTCOME> 
In San Antonio, there is the Winter Hill Inn, a 2 star hotel, from August 18th to August 28, it has free breakfast, wifi and parking. it goes for $$3167. 
In san francisco, thre is no package available .
In Kyoto, the Glorious Bazaar Resort is available it has a 3 star rating, with free breakfast and wifi. The dates are from the 17th to the 22nd or the 25th until the 30th.
In Cordoba, there are possibilities for a 10-day trip but only leaving on August 24th. The cost is $$2018 with free breakfast and wifi. The dates are from August 24th to September 2nd.
</OUTCOME>


#### Example 3:
Dialogue:
user: I wanna find a vacation between yesterday and the 24th for under 3200 and leaving from Tijuana
system: Ok! Do you have a destination in mind?
user: no
system: Ok, well I'll let you know the top options I've got. I have a 16 day package to Porto, 14 days in Los Angeles, Rio de Janeiro, or Tampa, 13 days in Vancouver, or Goiania.
user: Tell me about Porto
system: The hotel Amistad is 3.5 stars with 9.16/10 from past guests. It is near a mall, shopping district, and univeristy. The total with economy flights would be 3064.92 dollars, including breakfast and wifi.
user: Tell me about the Goiania package
system: With economy class flights, you could stay at Scarlet Palms Resort for 2903.54 USD. This 3.5 star hotel is near a park and shopping district.
user: whats the guest rating
system: 7.15/10
user: I like the one to Porto can you book it?
system: Sure! Will that be everything for you today?
user: ya but are there business class flights for porto?
system: Not within your budget, sorry.
user: ok thats fine, economy it is.
system: Great - I'll book economy class. Enjoy your trip!
user: thanks

Goal Outcome:
<THINK> The user does not modify their criteria to explore other options after they find a matching package. Thus there is only one goal without multiple options. </THINK>

<GOAL> Find a vacation from yesterday to 24th. You leave from Tijuana. You have a budget of 3200. You are flexible on destinations. Ask for information about suggested packages. Compare the packages and pick the one you like best. </GOAL>
<OUTCOME> 
In Porto, there is a 16 day package. The hotel Amistad is 3.5 stars with 9.16/10 from past guests. It is near a mall, shopping district, and univeristy. The total with economy flights would be 3064.92 dollars, including breakfast and wifi. Business class flights are not within 3200.
In Los Angeles, Rio de Janeiro, or Tampa, there is a 14 day package.
In Vancouver, there is a 13 day package. 
In Goiania, there is a 13 day package. With economy class flights, there is a stay at Scarlet Palms Resort for 2903.54 USD. This 3.5 star hotel is near a park and shopping district. The guest rating is 7.15/10.
</OUTCOME>



### Your Task:
Now, extract the progression of <GOAL> and <OUTCOME> tags for the following dialogue.  
Think about the goal progression using <THINK> and </THINK>, focus on whether the user has multiple options within one goal.
For the one initial goal, use <GOAL> ... </GOAL>.  
For any subsequent alternative goal, use <ALT_GOAL> ... </ALT_GOAL>.
Every alternative goal must begin with “If nothing matches your constraints,”  
Include all relevant database information under <OUTCOME>.

Dialogue:  
$dialogue

Goal Outcome:
<THINK> The user 
''')

def parse_goal(response): 
    if response is None or not response.startswith('<THINK>'): 
        return None, None
    # print(response)
    # print()
    goals = []
    outcomes = []
    while True: 
        start_token, end_token = "<ALT_GOAL>" if goals else "<GOAL>", "</ALT_GOAL>" if goals else "</GOAL>"
        start_idx = response.find(start_token)
        end_idx = response.find(end_token)
        if start_idx == -1 or end_idx == -1:
            break
        goal = response[start_idx+len(start_token):end_idx].strip()
        response = response[end_idx+len(end_token):].strip()
        goals.append(goal)

        start_token, end_token = "<OUTCOME>", "</OUTCOME>"
        start_idx = response.find(start_token)
        end_idx = response.find(end_token)
        if start_idx == -1 or end_idx == -1:
            break
        outcome = response[start_idx+len(start_token):end_idx].strip()
        response = response[end_idx+len(end_token):].strip()
        outcomes.append(outcome)
    if not goals or not outcomes: 
        return None, None
    if len(goals) != len(outcomes): 
        print("mismatched number of goals and outcomes.")
    return goals, outcomes
