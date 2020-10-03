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
import multiprocessing

word_tag_separator = '_'
BOS = "#"  # Beggining of sentence
EOS = "@"  # End of Sentence
OOV = "^"  # Out of vocabulary
SEED = 9322238373737383837
emission_matrix = {}  # 2-D dictionary
transmission_matrix = {}  # 2-D dictionary
frequency_of_tags = {}
size_of_vocabulary = 0
K1 = 40
K2 = 50
want_parallelism = 0
number_of_tags = 0
assumed_max_length_sent = 100

def get_dataset(file_name):
    f = open(file_name, 'r')
    dataset = f.read()
    f.close()
    return dataset


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
    random.seed(SEED)
    random.shuffle(sentences)
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
                tag = word_tag[idx + 1:][:2]
            except BaseException:
                word = word_tag
                tag = OOV
            word = word.lower()

            sentence.append((word, tag))

        sentences[i] = sentence

    return sentences


def preprocessing(train_data):
    global frequency_of_tags, viterbi, tags, number_of_tags, size_of_vocabulary
    frequency_of_tags = {}
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

    number_of_tags = len(frequency_of_tags)
    size_of_vocabulary = len(vocabulary)
    viterbi = [[0] * number_of_tags for _ in range(assumed_max_length_sent)]
    tags = list(frequency_of_tags.keys())


def calculate_emmission_matrix(train_data):

    for tag in frequency_of_tags:
        emission_matrix[tag] = {}

    for sentence in train_data:
        for word_tag in sentence:
            word = word_tag[0]
            tag = word_tag[1]

            if word not in emission_matrix[tag]:
                emission_matrix[tag][word] = K1

            emission_matrix[tag][word] += 1
    for tag in emission_matrix:
        for word in emission_matrix[tag]:
            emission_matrix[tag][word] = log(emission_matrix[tag][word] / (
                frequency_of_tags[tag] + K1 * size_of_vocabulary))  # laplace smoothing


def calculate_transmission_matrix(train_data):
    for tag_1 in frequency_of_tags:
        transmission_matrix[tag_1] = {}
        for tag_2 in frequency_of_tags:
            transmission_matrix[tag_1][tag_2] = K2

    for sentence in train_data:
        for i in range(1, len(sentence)):
            tag_1 = sentence[i - 1][1]
            tag_2 = sentence[i][1]
            transmission_matrix[tag_1][tag_2] += 1

    for tag_1 in transmission_matrix:
        total = frequency_of_tags[tag_1] + K2 * len(frequency_of_tags)
        for tag_2 in transmission_matrix[tag_1]:
            transmission_matrix[tag_1][tag_2] = log(
                transmission_matrix[tag_1][tag_2] / total)


def viterbi_algo(sentence):
    global assumed_max_length_sent
    number_of_words = len(sentence)

    if number_of_words > assumed_max_length_sent:

        for i in range(assumed_max_length_sent, number_of_words):
            viterbi.append([0] * number_of_tags)

        assumed_max_length_sent = number_of_words

    # Base Case for 1st word
    for i in range(number_of_tags):
        tag = tags[i]
        word = sentence[1][0]

        try:
            emi_prob = emission_matrix[tag][word]
        except:
            emi_prob = -log(K1 * size_of_vocabulary + frequency_of_tags[tag])

        viterbi[1][i] = (emi_prob + transmission_matrix[BOS][tag], i)

    for k in range(2, number_of_words):
        word = sentence[k][0]
        for i in range(number_of_tags):

            tag_2 = tags[i]
            max_val = -1000000
            col_idx = 0

            for j in range(number_of_tags):
                tag_1 = tags[j]
                val = transmission_matrix[tag_1][tag_2] + viterbi[k - 1][j][0]
                if val > max_val:
                    max_val = val
                    col_idx = j

            try:
                max_val += emission_matrix[tag_2][word]
            except:
                max_val -= log(K1 * size_of_vocabulary + frequency_of_tags[tag])

            viterbi[k][i] = (max_val, col_idx)

    col_idx = 0
    ans = []
    for i in range(number_of_tags):
        if viterbi[number_of_words - 1][col_idx][0] < viterbi[number_of_words - 1][i][0]:
            col_idx = i

    for i in range(number_of_words - 1, 1, -1):
        ans.append(tags[viterbi[i][col_idx][1]])
        col_idx = viterbi[i][col_idx][1]

    return ans[::-1]


def predict_parallely(test_data):
    correct = wrong = 0
    for sentence in test_data:
        predicted = viterbi_algo(sentence)
        for j in range(1, len(sentence) - 1):
            if sentence[j][1] == predicted[j - 1]:
                correct += 1
            else:
                wrong += 1
    return (correct, wrong)


def predict(sentence):
    correct = wrong = 0
    predicted = viterbi_algo(sentence)
    for j in range(1, len(sentence) - 1):
        if sentence[j][1] == predicted[j - 1]:
            correct += 1
        else:
            wrong += 1
    return (correct, wrong)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage : python3 Assn2.py <file_name>")
        sys.exit()

    dataset = get_dataset(sys.argv[1])
    sentences = get_sentences_from_text(dataset)
    sentences = separate_tag_from_word(sentences)
    dataset = split_dataset_into_3_parts(sentences)

    for i in range(3):
        print(i)
        p = [0, 1, 2]
        p.remove(i)
        train_data, test_data = make_train_and_test_data(
            dataset, p[0], p[1], i)
        preprocessing(train_data)
        calculate_emmission_matrix(train_data)
        calculate_transmission_matrix(train_data)
        correct = wrong = 0
        if want_parallelism:
            pool = multiprocessing.Pool(processes=8)
            start = 0
            n = len(test_data)
            split = n // 8
            x = []
            for i in range(split, n, split):
                x.append(test_data[start:i])
                start = i

            outputs = pool.map(predict_parallely, x)
            pool.terminate()
            for i in outputs:
                correct += i[0]
                wrong += i[1]
            print(correct, wrong, correct / (correct + wrong), K1, K2)
        else:
            for sentence in test_data:
                x = predict(sentence)
                correct += x[0]
                wrong += x[1]

            print(correct, wrong, correct / (correct + wrong), K1, K2)
