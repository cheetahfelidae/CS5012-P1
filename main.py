import json
from collections import defaultdict
from nltk.corpus import brown

root_dict = defaultdict(dict)  # this contains tag to word mapping with number of occurrences of the word
tags_dict = defaultdict(dict)  # this contains transition values
tags_counter = dict()
emmit_tag_counter = dict()

cur_tag = 'S0'
prev_tag = 'S0'

if 'S0' not in tags_counter:
    tags_counter['S0'] = 1
else:
    tags_counter['S0'] += 1

if 'S0' not in emmit_tag_counter:
    emmit_tag_counter['S0'] = 1
else:
    emmit_tag_counter['S0'] += 1

sents = brown.tagged_sents(tagset='universal')

for sent in sents:
    words = sent

    for word in words:
        token = word[0]  # this gives the word
        tag = word[1]  # this gives the POS tag

        # Logic for word to tag mapping
        if token not in root_dict:
            root_dict[token][tag] = 1
        elif tag not in root_dict[token]:
            root_dict[token][tag] = 1
        else:
            root_dict[token][tag] += 1

        # Update the emission tags counter
        if tag not in emmit_tag_counter:
            emmit_tag_counter[tag] = 1
        else:
            emmit_tag_counter[tag] += 1

        # Logic for transition probability dict
        prev_tag = cur_tag
        cur_tag = tag

        if prev_tag not in tags_dict:
            tags_dict[prev_tag][cur_tag] = 1
        elif cur_tag not in tags_dict[prev_tag]:
            tags_dict[prev_tag][cur_tag] = 1
        else:
            tags_dict[prev_tag][cur_tag] += 1

        # Update the tags counter
        if tag not in tags_counter:
            tags_counter[tag] = 1
        else:
            tags_counter[tag] += 1

# print(root_dict)
# print(tags_dict)
# print(emmit_tag_counter)

####### Transition Probability Calculation #######
# fill zero value in empty cells
for row in tags_dict:
    for col in tags_dict[row]:
        if col not in tags_dict[row] and col is not 'S0':
            tags_dict[row][col] = 0

for row in tags_dict:
    for col in tags_dict[row]:
        tags_dict[row][col] = tags_dict[row][col] * 1.0 / tags_counter[row]
######## End of Transition Probability Calculation = tagsDict (Fix FF/end of line cases) #######

######## Emission Probability Calculation #######
for row in root_dict:
    for innerTag in root_dict[row]:
        root_dict[row][innerTag] = root_dict[row][innerTag] * 1.0 / emmit_tag_counter[innerTag]

with open('model.txt', 'w') as outfile:
    # json.dump({"TransitionProb": tags_dict, "EmissionProb": root_dict}, outfile, indent=4)
    json.dump({"TransitionProb": tags_dict}, outfile, indent=4)
