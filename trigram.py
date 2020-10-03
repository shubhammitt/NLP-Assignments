
import sys
import random
from math import log
from nltk.tokenize import word_tokenize

word_tag_separator = '_'
BOS = "##"  # Beggining of sentence
EOS = "###"  # End of Sentence
OOV = "####"  # Out of vocabulary
SEED = 9322238373737383837
emission_matrix = {}  # 2-D dictionary
transmission_matrix = {}  # 2-D dictionary
frequency_of_tags = {}
vocabulary = {}

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


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage : python3 trigram.py <file_name>")
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
        
