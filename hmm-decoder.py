import json
from collections import defaultdict

import sys
from nltk.corpus import brown


def viterbi_algo(sents, emission_table, transit_table):
    correct_counter = 0  # the number of the correct guess
    words_counter = 0  # the number of testing words

    trained_tags = transit_table.keys()

    for sent in sents:
        viterbi = defaultdict(dict)  # a path probability matrix viterbi
        back_pointer = defaultdict(dict)

        words = [w for (w, _) in sent]
        tags = [t for (_, t) in sent]

        words_counter += len(words)

        word = words[0]

        ''' initialisation step '''
        if word in emission_table:  # known words
            for s in emission_table[word]:
                viterbi[s][0] = transit_table['<S>'][s] * emission_table[word][s]
                back_pointer[s][0] = '<S>'

        else:  # unknown words
            for s in trained_tags:
                if s != '<S>':
                    viterbi[s][0] = transit_table['<S>'][s]
                    back_pointer[s][0] = '<S>'

        ''' recursion step '''
        t = 1
        while t < len(words):
            word = words[t]

            ### get the tags of previous word - Remember tags may have <S> so cover that case in conditions ###
            if words[t - 1] in emission_table:
                prev_tags = emission_table[words[t - 1]].keys()
            else:
                prev_tags = trained_tags

            if word in emission_table:  # known words
                for s in emission_table[word]:
                    max_val = -sys.maxint - 1
                    cur_back_pointer = ''
                    for k in prev_tags:
                        if k != '<S>':
                            prob_val = viterbi[k][t - 1] * transit_table[k][s] * emission_table[word][s]
                            if prob_val > max_val:
                                max_val = prob_val
                                cur_back_pointer = k

                    viterbi[s][t] = max_val
                    back_pointer[s][t] = cur_back_pointer

            else:  # unknown words
                for s in trained_tags:
                    if s != '<S>':
                        max_val = -sys.maxint - 1
                        cur_back_pointer = ''
                        for k in prev_tags:
                            if k != '<S>':
                                prob_val = viterbi[k][t - 1] * transit_table[k][s]
                                if prob_val > max_val:
                                    max_val = prob_val
                                    cur_back_pointer = k

                        viterbi[s][t] = max_val
                        back_pointer[s][t] = cur_back_pointer
            t += 1

        ''' termination step '''
        post_tags = list()

        max_val = -sys.maxint - 1
        most_state = ''
        for s in trained_tags:
            if t - 1 in viterbi[s] and viterbi[s][t - 1] > max_val:
                max_val = viterbi[s][t - 1]
                most_state = s

        post_tags.append(most_state)
        prev_state = most_state

        counter = t - 1
        while counter > 0:
            prev_state = back_pointer[prev_state][counter]
            post_tags.append(prev_state)
            counter -= 1

        tagged_line = ''
        tags_len = len(post_tags)
        for i, word in enumerate(words):
            tagged_line += word + '/' + post_tags[tags_len - 1] + ' '
            if post_tags[tags_len - 1] == tags[i]: correct_counter += 1
            tags_len -= 1

        # fout = open("hmm-output.txt", 'w')
        # fout.write(taggedLine.strip() + '\n')
        # fout.close()

    print("Accuracy: " + str(correct_counter * 100.0 / words_counter) + "%")


def main():
    # Load the HMM model
    with open('hmm-model.txt') as model_file:
        model = json.load(model_file)
        transit_table = model["Transition"]
        emission_table = model["Emission"]

    sents = brown.tagged_sents(tagset='universal')
    sents = sents[int(round(len(sents) * 0.95)):]  # only 5% of sentences from the end being used as testing data

    viterbi_algo(sents, emission_table, transit_table)


if __name__ == "__main__": main()
