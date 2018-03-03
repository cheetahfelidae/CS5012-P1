from __future__ import print_function
import json
from collections import defaultdict
from nltk.corpus import brown


def map_word_to_tag(word, tag, root_dict):
    if word not in root_dict:
        root_dict[word][tag] = 1
    elif tag not in root_dict[word]:
        root_dict[word][tag] = 1
    else:
        root_dict[word][tag] += 1


def count(prev_tag, cur_tag, tags_dict):
    if prev_tag not in tags_dict:
        tags_dict[prev_tag][cur_tag] = 1
    elif cur_tag not in tags_dict[prev_tag]:
        tags_dict[prev_tag][cur_tag] = 1
    else:
        tags_dict[prev_tag][cur_tag] += 1


def transition_cal(tags_dict, tags_counter):
    total_tags = len(tags_dict) - 1  # one less because of start of sentence <S>

    for i in tags_dict:
        for j in tags_dict:
            if j != '<S>':
                if j not in tags_dict[i]:
                    tags_dict[i][j] = 0  # fill empty cells with zero value
                else:
                    tags_dict[i][j] = (tags_dict[i][j] + 1.0) / (
                            tags_counter[i] + total_tags)  # apply Laplace smoothing


def emission_cal(root_dict, tags_counter):
    for i in root_dict:
        for j in root_dict[i]:
            root_dict[i][j] = root_dict[i][j] * 1.0 / tags_counter[j]


def main():
    obser_table = defaultdict(
            dict)  # containing word-tag mapping with number of occurrences of the word (observation likelihood)
    trans_table = defaultdict(dict)  # containing transition values (transition probabilities)
    tags_counter = dict()

    sents = brown.tagged_sents(tagset='universal')
    sents = sents[
            :int(round(len(sents) * 0.95))]  # only 95% of sentences from the beginning being used as training data

    for sent in sents:
        if '<S>' not in tags_counter:
            tags_counter['<S>'] = 1
        else:
            tags_counter['<S>'] += 1

        cur_tag = '<S>'
        for token in sent:
            word = token[0]  # giving the word
            tag = token[1]  # giving the POS tag

            map_word_to_tag(word, tag, obser_table)

            # logic for transition probability dict
            prev_tag = cur_tag
            cur_tag = tag

            count(prev_tag, cur_tag, trans_table)

            # update the number of occurrences of the (new) tag
            tags_counter[tag] = 1 if tag not in tags_counter else tags_counter[tag] + 1

    transition_cal(trans_table, tags_counter)

    emission_cal(obser_table, tags_counter)

    with open('hmm-model.txt', 'w') as outfile:
        json.dump({"Transition": trans_table, "Emission": obser_table}, outfile, indent=4)


if __name__ == "__main__": main()
