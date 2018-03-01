from __future__ import print_function
import json
from collections import defaultdict
from nltk.corpus import brown


def map_token_to_tag(token, tag, root_dict):
    if token not in root_dict:
        root_dict[token][tag] = 1
    elif tag not in root_dict[token]:
        root_dict[token][tag] = 1
    else:
        root_dict[token][tag] += 1


def count(prev_tag, cur_tag, tags_dict):
    if prev_tag not in tags_dict:
        tags_dict[prev_tag][cur_tag] = 1
    elif cur_tag not in tags_dict[prev_tag]:
        tags_dict[prev_tag][cur_tag] = 1
    else:
        tags_dict[prev_tag][cur_tag] += 1


def transition_cal(tags_dict, tags_counter):
    for row in tags_dict:
        for col in tags_dict:
            if col != '<S>':
                if col not in tags_dict[row]:
                    tags_dict[row][col] = 0  # fill empty cells with zero value
                else:
                    tags_dict[row][col] = tags_dict[row][col] * 1.0 / tags_counter[row]


def emission_cal(root_dict, tags_counter):
    for row in root_dict:
        for col in root_dict[row]:
            root_dict[row][col] = root_dict[row][col] * 1.0 / tags_counter[col]


def main():
    obser_table = defaultdict(
            dict)  # containing word-tag mapping with number of occurrences of the word (observation likelihood)
    trans_table = defaultdict(dict)  # containing transition values (transition probabilities)
    tags_counter = dict()
    sents = brown.tagged_sents(tagset='universal')

    for sent in sents:
        if '<S>' not in tags_counter:
            tags_counter['<S>'] = 1
        else:
            tags_counter['<S>'] += 1

        cur_tag = '<S>'
        for word in sent:
            token = word[0]  # giving the token (word)
            tag = word[1]  # giving the POS tag

            map_token_to_tag(token, tag, obser_table)

            # logic for transition probability dict
            prev_tag = cur_tag
            cur_tag = tag

            count(prev_tag, cur_tag, trans_table)

            # update the number of occurrences of the (new) tag
            tags_counter[tag] = 1 if tag not in tags_counter else tags_counter[tag] + 1

    transition_cal(trans_table, tags_counter)

    emission_cal(obser_table, tags_counter)

    with open('model.txt', 'w') as outfile:
        # json.dump({"TransitionProb": tags_dict, "EmissionProb": root_dict}, outfile, indent=4)
        json.dump({"TransitionProb": trans_table}, outfile, indent=4)


if __name__ == "__main__": main()
