from collections import Counter
from nltk.corpus import brown

sents = brown.tagged_sents(tagset='universal')

first = sents[0]

words = [w for (w, _) in first]

tags = [t for (_, t) in first]

# for sent in tags[0:10]:
#     print(sent)

wordcounts = Counter(words)
print(wordcounts['The'])