import json
from collections import defaultdict

import sys
from nltk.corpus import brown


def start_viterbi(sents, emiss_table, transit_table):
    correct_count = 0  # the number of the correct guess
    word_count = 0  # the number of testing words

    trained_tags = transit_table.keys()

    for sent in sents:
        viterbi = defaultdict(dict)  # a path probability matrix viterbi
        back_pointer = defaultdict(dict)

        words = [w for (w, _) in sent]
        tags = [t for (_, t) in sent]

        word_count += len(words)

        ''' 
            ------------------------------------------- Initialisation Step ------------------------------------------
            - Set each state in the first column to the product of the transition probability (into it from the start state)
            and the observation probability (of the first word (t = 0)) 
        '''
        if words[0] in emiss_table:  # known words
            for s in emiss_table[words[0]]:
                viterbi[s][0] = transit_table['<S>'][s] * emiss_table[words[0]][s]
                back_pointer[s][0] = '<S>'

        else:  # unknown words
            for s in trained_tags:
                if s != '<S>':
                    viterbi[s][0] = transit_table['<S>'][s]
                    back_pointer[s][0] = '<S>'

        ''' 
            ------------------------------------------- Recursion Step ---------------------------------------------
            - For every state in column 1, compute the probability of moving into each state in column 2, and so on.
            - For each state qj at time t, compute the value viterbi[s, t] 
            by taking the maximum over the extensions of all the paths that lead to the current cell 
         '''
        t = 1
        while t < len(words):
            '''
                - get the previous-word tags
                    - known words condition
                    - unknown words condition
                        - include <S> but it will be covered in the conditions later
            '''
            if words[t - 1] in emiss_table:
                prev_tags = emiss_table[words[t - 1]].keys()
            else:
                prev_tags = trained_tags

            '''
                - find the maximum of the product of three factors as follow:
                    # the previous viterbi path probability from the previous time step
                    # the transition probability from the previous state qi to current state qj
                    # the state observation likelihood of the observation symbol ot given the current state j
                    
                    - known words condition
                    - unknown words condition
            '''
            if words[t] in emiss_table:
                for s in emiss_table[words[t]]:
                    max_val = -sys.maxint - 1
                    cur_back_pointer = ''
                    for k in prev_tags:
                        if k != '<S>':
                            val = viterbi[k][t - 1] * transit_table[k][s] * emiss_table[words[t]][s]
                            if val > max_val:
                                max_val = val
                                cur_back_pointer = k

                    viterbi[s][t] = max_val
                    back_pointer[s][t] = cur_back_pointer

            else:
                for s in trained_tags:
                    if s != '<S>':
                        max_val = -sys.maxint - 1
                        cur_back_pointer = ''
                        for k in prev_tags:
                            if k != '<S>':
                                val = viterbi[k][t - 1] * transit_table[k][s]
                                if val > max_val:
                                    max_val = val
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
            if post_tags[tags_len - 1] == tags[i]: correct_count += 1
            tags_len -= 1

        # fout = open("hmm-output.txt", 'w')
        # fout.write(taggedLine.strip() + '\n')
        # fout.close()

    print("Accuracy: " + str(correct_count * 100.0 / word_count) + "%")


def main():
    # Load the HMM model
    with open('hmm-model.txt') as model_file:
        model = json.load(model_file)
        transit_table = model["Transition"]
        emission_table = model["Emission"]

    sents = brown.tagged_sents(tagset='universal')
    sents = sents[int(round(len(sents) * 0.95)):]  # only 5% of sentences from the end being used as testing data

    start_viterbi(sents, emission_table, transit_table)


if __name__ == "__main__": main()
