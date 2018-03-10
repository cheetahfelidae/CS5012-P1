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
            there are available seven corpora for this programme
                1. 'brown_universal'
                2. 'brown'
                3. 'conll2000_universal'
                4. 'conll2000'
                5. 'conll2002'
                6. 'alpino'
                7. 'dependency_treebank'
                8. 'treebank'

        - the output path of the generated model
        - the output path of the words with their assigned tags
        - the output path of the confusion matrix

        e.g. python main.py brown hmm-model.txt hmm_output.txt confusion.txt