import sys
from nltk.corpus import brown, conll2000, alpino, floresta, dependency_treebank, treebank, conll2002
from hmm_learner import hmm_learner
from hmm_decoder import hmm_decoder


def main():
    if len(sys.argv) == 3:
        # sents = brown.tagged_sents(tagset='universal')
        sents = conll2000.tagged_sents(tagset='universal')
        # sents = conll2002.tagged_sents()
        # sents = alpino.tagged_sents()
        # sents = dependency_treebank.tagged_sents()
        # sents = treebank.tagged_sents()
        # sents = floresta.tagged_sents()

        training_sents = sents[:int(round(len(
                sents) * 0.95))]  # only 95% of sentences from the beginning being used as training data
        testing_sents = sents[int(round(
                len(sents) * 0.95)):]  # only 5% of sentences from the end being used as testing data

        hmm_learner(training_sents, sys.argv[1])
        hmm_decoder(testing_sents, sys.argv[1], sys.argv[2])

    else:
        print("usage: <hmm_model_file> <hmm_output_file>")


if __name__ == "__main__": main()
