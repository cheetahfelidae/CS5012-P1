import io
import json
from collections import defaultdict
import sys


def start_viterbi(sents, emiss_table, transit_table, output_file):
    correct_count = 0  # the number of the correct guess
    word_count = 0  # the number of testing words

    trained_tags = transit_table.keys()

    f_out = io.open(output_file, mode='w', encoding='utf-8')

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
            
                - known words condition
                - unknown words condition
        '''
        if words[0] in emiss_table:
            for s in emiss_table[words[0]]:
                viterbi[s][0] = transit_table['<S>'][s] * emiss_table[words[0]][s]
                back_pointer[s][0] = '<S>'

        else:
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

        ''' 
           ------------------------------------------- termination step -------------------------------------------
           - get the final state (tag)
           - backtrace from the final state
           - write the backtrace path to the file
        '''
        t = t - 1
        post_tags = list()  # containing the backtrace path to states back in time from backpointer

        max_val = -sys.maxint - 1
        most_state = ''
        for s in trained_tags:
            if t in viterbi[s] and viterbi[s][t] > max_val:
                max_val = viterbi[s][t]
                most_state = s

        post_tags.append(most_state)

        prev_state = most_state
        while t > 0:
            prev_state = back_pointer[prev_state][t]
            post_tags.append(prev_state)
            t -= 1

        tagged_line = ''
        tags_len = len(post_tags)
        for i, word in enumerate(words):
            tagged_line += word + '/' + post_tags[tags_len - 1] + ' '
            if post_tags[tags_len - 1] == tags[i]: correct_count += 1
            tags_len -= 1
        f_out.write(tagged_line.strip() + '\n')

    f_out.close()

    print("(4/4) The output file showing words with their assigned tags is written to " + output_file)
    print("")
    print("Accuracy Rate: " + str(
            round(correct_count * 100.0 / word_count, 2)) + "%")  # calculate and show the accuracy rate


def hmm_decoder(sents, model_file, output_file):
    print("(3/4) Applying the trained HMM on each testing sentence and assigning a tag to each word")

    # Load the HMM model
    with open(model_file) as f_model:
        model = json.load(f_model)
        transit_table = model["Transition"]
        emission_table = model["Emission"]

    start_viterbi(sents, emission_table, transit_table, output_file)
