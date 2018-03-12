##### README File for Hidden Markov Model part-of-speech tagger #####

- Program uses HMM for modeling and learning the tagged corpus and the Viterbi Algorithm for decoding.
- Laplace smoothing is used in learning to cover unseen words and tags transitions.

Source Code : There are 3 python files as follow.
    main.py -> This is main python file which will run hmm_learn.py and hmm_decode.py consecutively.

	    - hmm_learn.py --> This is used for learning existing tagged corpus and generating the model for decoding.

	    - hmm_decode.py --> This is used for decoding untagged corpus and assigning tags to them.

Usage :
    main.py --> there are 3 required commandline arguments as below
        - the desired corpus which is used as training and testing data (95% for training and the remaining for testing)
            there are available nine corpora for this programme
                1. 'brown_universal' - Brown Corpus
                2. 'brown' - Brown Corpus with the universal tagset
                3. 'conll2000_universal' - CONLL 2000 Chucking Corpus with the universal tagset
                4. 'conll2000' - CONLL 2000 Chucking Corpus
                5. 'conll2002' - CONLL 2002 Named Entity Recognition Corpus
                6. 'alpino' - Alpino Dutch Treebank
                7. 'dependency_treebank' - Dependency Parsed Treebank
                8. 'treebank' - Penn Treebank Sample
                9. 'indian' - Indian Language POS-tagged Corpus

        - the output path of the generated model
        - the output path of the words with their assigned tags
        - the output path of the confusion matrix

        e.g. python main.py brown_universal hmm-model.txt hmm_output.txt confusion.txt