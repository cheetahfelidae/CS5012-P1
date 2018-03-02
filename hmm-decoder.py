import json
from collections import defaultdict
from nltk.corpus import brown


def main():
    # Load the HMM model
    with open('hmm-model.txt') as model_file:
        modelData = json.load(model_file)
        trans_table = modelData["Transition"]
        emission_table = modelData["Emission"]
        allPossibleTags = trans_table.keys()  # all possible tags from the model

    fout = open("hmm-output.txt", 'w')

    sents = brown.tagged_sents(tagset='universal')

    correct_counter = 0
    words_counter = 0

    num_training_sents = int(round(len(sents) * 0.95))  # used as start index of the first sentence of the testing part

    for sent in sents[num_training_sents:]:
        probability = defaultdict(dict)
        backpointer = defaultdict(dict)

        words = [w for (w, _) in sent]
        tags = [t for (_, t) in sent]

        words_counter += len(words)

        T = len(words)
        ##### Start of Viterbi #####

        ## Initialization at t=1 ##
        word = words[0]

        ##### If the word is seen #####
        if word in emission_table:
            for eachTag in emission_table[word]:
                probability[eachTag][0] = trans_table['<S>'][eachTag] * emission_table[word][eachTag]
                backpointer[eachTag][0] = '<S>'
        ##### If the word is unseen #####
        else:
            for eachTag in allPossibleTags:
                if eachTag != '<S>':
                    probability[eachTag][0] = trans_table['<S>'][eachTag]
                    backpointer[eachTag][0] = '<S>'

        ## Recursion Step T=2 onwards ##
        itr = 1
        while (itr < T):
            word = words[itr]

            ### get the tags of previous word - Remember tags may have S0 so cover that case in conditions ###
            if words[itr - 1] in emission_table:
                previousTagsList = emission_table[words[itr - 1]].keys()
            else:
                previousTagsList = allPossibleTags

            ##### If the word is seen #####
            if word in emission_table:
                for eachTag in emission_table[word]:
                    maxVal = -1000
                    currentBackPtr = ''
                    for eachPrevTag in previousTagsList:
                        if eachPrevTag != '<S>':
                            probabilityVal = probability[eachPrevTag][itr - 1] * trans_table[eachPrevTag][
                                eachTag] * emission_table[word][eachTag]
                            if probabilityVal > maxVal:
                                maxVal = probabilityVal
                                currentBackPtr = eachPrevTag

                    probability[eachTag][itr] = maxVal
                    backpointer[eachTag][itr] = currentBackPtr

            ##### If the word is seen #####
            else:
                for eachTag in allPossibleTags:
                    if eachTag != '<S>':
                        maxVal = -1000
                        currentBackPtr = ''
                        for eachPrevTag in previousTagsList:
                            if eachPrevTag != '<S>':
                                probabilityVal = probability[eachPrevTag][itr - 1] * trans_table[eachPrevTag][
                                    eachTag]
                                if probabilityVal > maxVal:
                                    maxVal = probabilityVal
                                    currentBackPtr = eachPrevTag

                        probability[eachTag][itr] = maxVal
                        backpointer[eachTag][itr] = currentBackPtr

            itr += 1

        ##### Termination Step #####
        posTags = list()

        maxProbableVal = -10000
        mostProbableState = ''
        for state in allPossibleTags:
            if (itr - 1) in probability[state] and probability[state][itr - 1] > maxProbableVal:
                maxProbableVal = probability[state][itr - 1]
                mostProbableState = state

        posTags.append(mostProbableState)
        counter = itr - 1
        prevState = mostProbableState

        while (counter > 0):
            prevState = backpointer[prevState][counter]
            counter -= 1
            posTags.append(prevState)
        taggedLine = ''
        tagsLen = len(posTags)

        for i, word in enumerate(words):
            taggedLine += word + '/' + posTags[tagsLen - 1] + ' '
            if posTags[tagsLen - 1] == tags[i]: correct_counter += 1
            tagsLen -= 1

        # fout.write(taggedLine.strip() + '\n')
    fout.close()

    print("Accuracy: " + str(correct_counter * 100.0 / words_counter) + "%")


if __name__ == "__main__": main()
