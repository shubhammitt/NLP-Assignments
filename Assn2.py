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
import concurrent.futures
from nltk.tokenize import word_tokenize
import threading 
word_tag_separator = '_'
BOS = "##"  # Beggining of sentence
EOS = "###"  # End of Sentence
OOV = "####"  # Out of vocabulary
SEED = 9322238373737383837
emission_matrix = {}  # 2-D dictionary
transmission_matrix = {}  # 2-D dictionary
frequency_of_tags = {}
vocabulary = {}
K1 = 40
K2 = 50
want_parallelism = 1

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
                emission_matrix[tag][word] = K1

            emission_matrix[tag][word] += 1
    for tag in emission_matrix:
        for word in emission_matrix[tag]:
            emission_matrix[tag][word] = log(emission_matrix[tag][word] / (
                frequency_of_tags[tag] + K1 * len(vocabulary)))  # laplace smoothing


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


def viterbi(sentence):
    number_of_tags = len(frequency_of_tags)
    number_of_words = len(sentence)
    viterbi = [[0] * number_of_words for i in range(number_of_tags)]
    tags = list(frequency_of_tags.keys())

    # Base Case for 1st word
    for i, tag in enumerate(tags):
        word = sentence[1][0]
        if word in emission_matrix[tag]:
            viterbi[i][1] = [emission_matrix[tag][word], i]
        else:
            viterbi[i][1] = [
                log(1 / (K1 * len(vocabulary) + frequency_of_tags[tag])), i]

        viterbi[i][1][0] += transmission_matrix[BOS][tag]

    for k in range(2, number_of_words):
        word = sentence[k][0]
        for i, tag_2 in enumerate(tags):
            c = -10**15
            t = ""
            for j, tag_1 in enumerate(tags):
                z = transmission_matrix[tag_1][tag_2] + viterbi[j][k - 1][0]
                if z > c:
                    c = z
                    t = j
            if word in emission_matrix[tag_2]:
                c = c + emission_matrix[tag_2][word]
            else:
                c = c + \
                    log((1 / (K1 * len(vocabulary) + frequency_of_tags[tag_2])))

            viterbi[i][k] = [c, t]

    m = 0
    ans = []
    for i in range(number_of_tags):
        if viterbi[m][-1][0] < viterbi[i][-1][0]:
            m = i

    for i in range(number_of_words - 1, 1, -1):
        ans.append(tags[viterbi[m][i][1]])
        m = viterbi[m][i][1]

    ans = ans[::-1]
    return ans

def predict_parallely(test_data):
	correct = wrong = 0
	for sentence in test_data:
	    predicted = viterbi(sentence)
	    for j in range(1, len(sentence) - 1):
	        if sentence[j][1] == predicted[j - 1]:
	            correct += 1
	        else:
	            wrong += 1
	return [correct, wrong]

def predict(sentence):
    correct = wrong = 0
    predicted = viterbi(sentence)
    for j in range(1, len(sentence) - 1):
        if sentence[j][1] == predicted[j - 1]:
            correct += 1
        else:
            wrong += 1
    return [correct, wrong]

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
        else:
        	for sentence in test_data:
        		x = predict(sentence)
        		correct += x[0]
        		wrong += x[1]

        print(correct, wrong, correct / (correct + wrong), K1, K2)
