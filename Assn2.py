'''
	Implementing HMM using Viterbi Algo.
	Steps:
		Split the text data into sentences
		add word boundary '###_###' at front and end of every sentence
		split the data into 3 parts for cross fold validation
		split word_tag pair
'''

import sys
import random
from math import log

word_tag_separator = '_'
BOS = "#"  # Beggining of sentence
EOS = "##"  # End of Sentence
OOV = "###"  # Out of vocabulary
SEED = 23
emission_matrix = {}  # 2-D dictionary
transmission_matrix = {}  # 2-D dictionary
frequency_of_tags = {}
vocabulary = {}


def get_sentences_from_text(dataset):
    '''
    '''
    # each newline is considered a sentence
    sentences_in_text = list(dataset.split('\n'))
    sentences_in_text = [
        sentence for sentence in sentences_in_text if len(sentence) > 0]

    # append BOS and EOS at both ends of each sentence
    number_of_sentences = len(sentences_in_text)

    for i in range(number_of_sentences):
        sentences_in_text[i] = BOS + word_tag_separator + BOS + " " + \
            sentences_in_text[i] + " " + EOS + word_tag_separator + EOS

    return sentences_in_text


def split_dataset_into_3_parts(sentences):
    '''
    '''
    # random.seed(SEED)
    # random.shuffle(sentences)
    number_of_sentences = len(sentences)
    split = number_of_sentences // 3
    dataset_1 = sentences[: split]
    dataset_2 = sentences[split: 2 * split]
    dataset_3 = sentences[2 * split:]
    return [dataset_1, dataset_2, dataset_3]


def make_train_and_test_data(dataset, i, j, k):
    return dataset[i] + dataset[j], dataset[k]


def separate_tag_from_word(sentences):

    number_of_sentences = len(sentences)
    for i in range(number_of_sentences):
        sentence = []
        for word_tag in sentences[i].split():
            try:
                idx = word_tag.rindex(word_tag_separator)
                word = word_tag[: idx]
                tag = word_tag[idx + 1:][: 2]
            except BaseException:
                word = word_tag
                tag = OOV

            # Normalization on word
            word = word.lower()
            sentence.append([word, tag])

        sentences[i] = sentence
    return sentences


def preprocessing(train_data):
    global frequency_of_tags, vocabulary
    frequency_of_tags = {OOV: 0}
    vocabulary = {}

    for sentence in train_data:
        for word_tag in sentence:
            tag = word_tag[1]
            word = word_tag[0]
            if word not in vocabulary:
                vocabulary[word] = 0
            vocabulary[word] += 1
            if tag not in frequency_of_tags:
                frequency_of_tags[tag] = 0
            frequency_of_tags[tag] += 1


def calculate_emmission_matrix(train_data):

    for tag in frequency_of_tags:
        emission_matrix[tag] = {}

    for sentence in train_data:
        for word_tag in sentence:
            word = word_tag[0]
            tag = word_tag[1]

            if word not in emission_matrix[tag]:
                emission_matrix[tag][word] = 1

            emission_matrix[tag][word] += 1
    for tag in emission_matrix:
        for word in emission_matrix[tag]:
            emission_matrix[tag][word] = log(
                emission_matrix[tag][word] / (frequency_of_tags[tag] + len(vocabulary)))  # laplace smoothing


def calculate_transmission_matrix(train_data):
    for tag_1 in frequency_of_tags:
        transmission_matrix[tag_1] = {}
        for tag_2 in frequency_of_tags:
            transmission_matrix[tag_1][tag_2] = 1

    for sentence in train_data:
        for i in range(1, len(sentence)):
            tag_1 = sentence[i - 1][1]
            tag_2 = sentence[i][1]
            transmission_matrix[tag_1][tag_2] += 1

    for tag_1 in transmission_matrix:
        total = sum(transmission_matrix[tag_1].values())
        for tag_2 in transmission_matrix[tag_1]:
            transmission_matrix[tag_1][tag_2] = log(
                transmission_matrix[tag_1][tag_2] / total)


def viterbi(sentence):
    number_of_tags = len(frequency_of_tags)
    number_of_words = len(sentence)
    viterbi = [[[] for j in range(number_of_words)]
               for i in range(number_of_tags)]

    # Base Case for 1st word
    for i, tag in enumerate(frequency_of_tags):
        word = sentence[1][0]
        if word in emission_matrix[tag]:
            viterbi[i][1] = [emission_matrix[tag][word], (i, BOS)]
        else:
            viterbi[i][1] = [
                1 / (len(vocabulary) + frequency_of_tags[tag]), (i, BOS)]

        viterbi[i][1][0] += transmission_matrix[BOS][tag]

    for k in range(2, number_of_words):
        word = sentence[k][0]
        for i, tag_2 in enumerate(frequency_of_tags):
            c = -1 * 10**15
            t = ""
            for j, tag_1 in enumerate(frequency_of_tags):
                z = transmission_matrix[tag_1][tag_2] + viterbi[j][k - 1][0]
                if z > c:
                    c = z
                    t = (j, tag_1)
            if word in emission_matrix[tag_2]:
                c = c + emission_matrix[tag_2][word]
            else:
                c = c + (1 / (len(vocabulary) + frequency_of_tags[tag_2]))

            viterbi[i][k] = [c, t]

    m = 0
    t = EOS
    ans = []
    for i in range(1, number_of_tags):
        if viterbi[m][-1][0] < viterbi[i][-1][0]:
            m = i
            t = viterbi[i][-1][1]

    idx = number_of_words - 1
    while t[1] != BOS:
        ans.append(t[1])
        print(t[0])
        idx -= 1
        t = viterbi[t[0]][idx][1]
    ans = ans[::-1]
    print(ans, '\n\n', sentence)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage : python3 Assn2.py <file_name>")
        sys.exit()

    f = open(sys.argv[1], 'r')
    dataset = f.read()
    f.close()

    sentences = get_sentences_from_text(dataset)
    sentences = separate_tag_from_word(sentences)
    dataset = split_dataset_into_3_parts(sentences)
    train_data, test_data = make_train_and_test_data(dataset, 0, 1, 2)
    preprocessing(train_data)
    calculate_emmission_matrix(train_data)
    calculate_transmission_matrix(train_data)
    viterbi(train_data[0])

    # c = 0
    # for tag in transmission_matrix:
    #     print(tag, transmission_matrix[tag], '\n\n\n')
    #     c += 1
    #     if c > 2:
    #         break
