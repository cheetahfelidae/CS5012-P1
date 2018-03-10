import sys
from nltk.corpus import brown, conll2000, alpino, floresta, dependency_treebank, treebank, conll2002
from hmm_learner import hmm_learner
from hmm_decoder import hmm_decoder


def select_sents(x):
    return {
        'brown_tag'          : brown.tagged_sents(tagset='universal'),  # Accuracy: 95.12% -> 91.54
        'brown'              : brown.tagged_sents(),  # Accuracy: 64.59%
        'conll2000_tag'      : conll2000.tagged_sents(tagset='universal'),  # Accuracy: 95.63% -> 96.75
        'conll2000'          : conll2000.tagged_sents(),  # Accuracy: -> 95.28
        'conll2002'          : conll2002.tagged_sents(),  # Accuracy: 89.45% -> 90.16%
        'alpino'             : alpino.tagged_sents(),  # Accuracy: 88.79% -> 76.99%
        'dependency_treebank': dependency_treebank.tagged_sents(),  # Accuracy: 90.79% -> 89.68%
        'treebank'           : treebank.tagged_sents(),  # Accuracy: 91.44% -> 90.0%
        # 'floresta'           : floresta.tagged_sents(),  # Accuracy: 83.63% ->
        'else'               : []  # in case of an unavailable corpus
    }.get(x, 'else')


def main():
    if len(sys.argv) == 5:
        # sents = select_sents(sys.argv[1])
        sents = select_sents('brown')

        if sents:
            training_sents = sents[:int(
                    round(len(sents) * 0.95))]  # only 95% of sentences from the beginning being used as training data
            testing_sents = sents[int(
                    round(len(sents) * 0.95)):]  # only 5% of sentences from the end being used as testing data

            hmm_learner(training_sents, sys.argv[2])
            hmm_decoder(testing_sents, sys.argv[2], sys.argv[3], sys.argv[4])
        else:
            print("INPUT CORPUS NOT AVAILABLE")

    else:
        print("USAGE:main.py <hmm_model_txt> <hmm_output_txt> <hmm_confusion_csv>")


if __name__ == "__main__": main()
