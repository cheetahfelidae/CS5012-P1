##### README file for hmm-pos-tagger #####

This is a Hidden Markov Model part-of-speech tagger for Catalan. Program uses HMM for modeling/learning the tagged corpus and the Viterbi Algorithm for decoding. Laplace smoothing is used in learning to cover unseen words and tags transitions.

The accuracy of this utility is 94.23%

Source Code : There are 2 main python files.

Usage:
	1. hmmlearn3.py --> This is the python code to learn existing tagged corpus and generate the model for decoding.

	Input: Give the input path of tagged corpus file (corpus -> catalan_corpus_train_tagged.txt)

	eg. > python hmmlearn3.py /path/to/input

	Output: This will output "hmmmodel.txt" file (sample model fileavailable in Sample Outputs folder).

	2. hmmdecode3.py --> This is the python code to decode untagged corpus and assign tags to them.

	Input: Give the input path of raw untagged corpus file (corpus -> catalan_corpus_dev_raw.txt). The code picks up the hmmmodel.txt internally.

	> python hmmdecode3.py /path/to/input

	Output: This will output "hmmoutput.txt" file (sample model fileavailable in Sample Outputs folder). You can compare this output with standard output file available in (corpus -> catalan_corpus_dev_tagged.txt) for accuracy calculation.

Note: Corpus folder also includes a small english corpus for small scale testing.