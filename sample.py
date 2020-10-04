from nltk.corpus import brown
brown_news_tagged = brown.tagged_sents()

brown_train = brown_news_tagged

from nltk.tag import untag
for i in brown_train:
    for j in i:
        print(j[0]+'_'+j[1], end=' ')
    print()