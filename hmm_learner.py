from __future__ import print_function
import json
from collections import defaultdict


def count_word_to_tag(word, tag, emiss_table):
    if word not in emiss_table:
        emiss_table[word][tag] = 1
    elif tag not in emiss_table[word]:
        emiss_table[word][tag] = 1
    else:
        emiss_table[word][tag] += 1


def count_prev_tag_to_cur_tag(prev_tag, cur_tag, transit_table):
    if prev_tag not in transit_table:
        transit_table[prev_tag][cur_tag] = 1
    elif cur_tag not in transit_table[prev_tag]:
        transit_table[prev_tag][cur_tag] = 1
    else:
        transit_table[prev_tag][cur_tag] += 1


def create_transit_table(transit_table, tags_counter):
    total_tags = len(transit_table) - 1  # one less because of start of sentence <S>

    for i in transit_table:
        for j in transit_table:
            if j != '<S>':
                if j not in transit_table[i]:
                    transit_table[i][j] = 0  # fill empty cells with zero value
                else:
                    transit_table[i][j] = (transit_table[i][j] + 1.0) / (
                            tags_counter[i] + total_tags)  # apply Laplace smoothing


def create_emiss_table(root_dict, tags_counter):
    for i in root_dict:
        for j in root_dict[i]:
            root_dict[i][j] = root_dict[i][j] * 1.0 / tags_counter[j]


def hmm_learner(sents, model_file):
    emiss_table = defaultdict(
            dict)  # containing word-tag mapping with number of occurrences of the word (observation likelihood)
    transit_table = defaultdict(dict)  # containing transition values (transition probabilities)
    tag_counters = dict()

    for sent in sents:
        if '<S>' not in tag_counters:
            tag_counters['<S>'] = 1
        else:
            tag_counters['<S>'] += 1

        cur_tag = '<S>'
        for token in sent:
            word = token[0]  # giving the word
            tag = token[1]  # giving the POS tag

            count_word_to_tag(word, tag, emiss_table)

            # logic for transition probability dict
            prev_tag = cur_tag
            cur_tag = tag

            count_prev_tag_to_cur_tag(prev_tag, cur_tag, transit_table)

            # update the number of occurrences of the (new) tag
            tag_counters[tag] = 1 if tag not in tag_counters else tag_counters[tag] + 1

    create_transit_table(transit_table, tag_counters)

    create_emiss_table(emiss_table, tag_counters)

    with open(model_file, 'w') as f_out:
        json.dump({"Transition": transit_table, "Emission": emiss_table}, f_out, indent=4)
