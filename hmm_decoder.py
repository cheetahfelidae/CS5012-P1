from __future__ import print_function
import io
import json
from collections import defaultdict
import sys

import operator


def get_unique_tags(tags):
    unique_tags = list()
    for tag in tags:
        if tag not in unique_tags:
            unique_tags.append(tag)

    unique_tags.sort()
    return unique_tags


def file_confusion_matrix(unique_ans_tags, out_tags, confusion_file):
    with open(confusion_file, "wb") as file:
        # print all output tags for the first row
        file.write(",")
        for j in unique_ans_tags:
            file.write(j + ",")
        file.write("\n")

        for i in unique_ans_tags:
            file.write(i + ",")
            for j in unique_ans_tags:
                file.write(str(out_tags[i][j]) + ",")
            file.write("\n")


# outputs the percentage of tags in the test corpus which has been guessed correctly
# and overall accuracy rate to the terminal
def print_accuracy(unique_ans_tags, out_tags):
    print("Accuracy Rate for Each Tag")

    stats = {}
    max(stats.iteritems(), key=operator.itemgetter(1))[0]
    num_tags = 0
    num_correct_tags = 0
    for i in unique_ans_tags:
        sum_row = 0
        for j in unique_ans_tags:
            sum_row += out_tags[i][j]
            num_tags += out_tags[i][j]
        accuracy = round(out_tags[i][i] * 100.0 / sum_row, 2)
        num_correct_tags += out_tags[i][i]
        print("%s:\t\t%.2f%%" % (i, accuracy))

    print("")
    print("Overall Accuracy Rate: %d correct tags / %d tags = %.2f%%" %
          (num_correct_tags, num_tags, num_correct_tags * 100.0 / num_tags))


# The algorithm applies the trained HMM on each untagged sentence form the testing part of a corpus
# to determine the sequence of tags with the highest probability
def start_viterbi(sents, emiss_table, transit_table, output_file, confusion_file):
    confusion_matrix = defaultdict(dict)

    trained_tags = transit_table.keys()

    f_out = io.open(output_file, mode='w', encoding='utf-8')

    for sent in sents:
        viterbi = defaultdict(dict)  # a path probability matrix viterbi
        back_pointer = defaultdict(dict)

        ans_words = [w for (w, _) in sent]
        ans_tags = [t for (_, t) in sent]

        ''' 
            ------------------------------------------- Initialisation Step ------------------------------------------
            - Set each state in the first column to the product of the transition probability (into it from the start state)
            and the observation probability (of the first word (t = 0)) 
            
                - known words condition
                - unknown words condition
        '''
        if ans_words[0] in emiss_table:
            for s in emiss_table[ans_words[0]]:
                viterbi[s][0] = transit_table['<S>'][s] * emiss_table[ans_words[0]][s]
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
        while t < len(ans_words):
            '''
                - get the previous-word tags
                    - known words condition
                    - unknown words condition
                        - include <S> but it will be covered in the conditions later
            '''
            if ans_words[t - 1] in emiss_table:
                prev_tags = emiss_table[ans_words[t - 1]].keys()
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
            if ans_words[t] in emiss_table:
                for s in emiss_table[ans_words[t]]:
                    max_val = -sys.maxint - 1
                    cur_back_pointer = ''
                    for k in prev_tags:
                        if k != '<S>':
                            val = viterbi[k][t - 1] * transit_table[k][s] * emiss_table[ans_words[t]][s]
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
        for i, word in enumerate(ans_words):
            post_tag = post_tags[tags_len - 1]
            tagged_line += word + '/' + post_tag + ' '

            if ans_tags[i] not in confusion_matrix:
                confusion_matrix[ans_tags[i]][post_tag] = 1
            elif post_tag not in confusion_matrix[ans_tags[i]]:
                confusion_matrix[ans_tags[i]][post_tag] = 1
            else:
                confusion_matrix[ans_tags[i]][post_tag] += 1

            tags_len -= 1
        f_out.write(tagged_line.strip() + '\n')

    f_out.close()

    print("(4/4) The output file showing words with their assigned tags is written to " + output_file)

    ''' 
          ------------------------------------- Confusion Matrix & Accuracy Rate -------------------------------------------
    '''
    # get all expected tags (answers) from testing data
    all_ans_tags = list()
    for sent in sents:
        all_ans_tags.extend([t for (_, t) in sent])
    unique_ans_tags = get_unique_tags(all_ans_tags)

    # fill empty cells of the confusion matrix with zero
    for i in unique_ans_tags:
        for j in unique_ans_tags:
            if i not in confusion_matrix:
                confusion_matrix[i][j] = 0
            elif j not in confusion_matrix[i]:
                confusion_matrix[i][j] = 0

    file_confusion_matrix(unique_ans_tags, confusion_matrix, confusion_file)

    print("")
    print_accuracy(unique_ans_tags, confusion_matrix)


def hmm_decoder(sents, model_file, output_file, confusion_file):
    print("(3/4) Applying the trained HMM on each testing sentence and assigning a tag to each word")

    # Load the HMM model
    with open(model_file) as f_model:
        model = json.load(f_model)
        transit_table = model["Transition"]
        emission_table = model["Emission"]

    start_viterbi(sents, emission_table, transit_table, output_file, confusion_file)
