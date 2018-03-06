import sys
from nltk.corpus import brown, conll2000, alpino, floresta, dependency_treebank, treebank, conll2002
from hmm_learner import hmm_learner
from hmm_decoder import hmm_decoder


def select_sents(x):
    return {
        'brown'              : brown.tagged_sents(tagset='universal'),  # Accuracy: 95.12%
        'conll2000'          : conll2000.tagged_sents(tagset='universal'),  # Accuracy: 95.63%
        'conll2002'          : conll2002.tagged_sents(),  # Accuracy: 89.45%
        'alpino'             : alpino.tagged_sents(),  # Accuracy: 88.79%
        'dependency_treebank': dependency_treebank.tagged_sents(),  # Accuracy: 90.79%
        'treebank'           : treebank.tagged_sents(),  # Accuracy: 91.44%
        'floresta'           : floresta.tagged_sents(),  # Accuracy: 83.63%
        'else'               : []  # in case of an unavailable corpus
    }.get(x, 'else')


def main():
    if len(sys.argv) == 4:
        sents = select_sents(sys.argv[1])

        if sents:
            training_sents = sents[:int(
                    round(len(sents) * 0.95))]  # only 95% of sentences from the beginning being used as training data
            testing_sents = sents[int(
                round(len(sents) * 0.95)):]  # only 5% of sentences from the end being used as testing data

            hmm_learner(training_sents, sys.argv[2])
            hmm_decoder(testing_sents, sys.argv[2], sys.argv[3])
        else:
            print("INPUT CORPUS NOT AVAILABLE")

    else:
        print("USAGE: <hmm_model_file> <hmm_output_file>")


if __name__ == "__main__": main()
