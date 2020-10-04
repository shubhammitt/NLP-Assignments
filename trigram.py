
import sys
import random
import multiprocessing
from math import log

word_tag_separator = '_'
BOS = "##"  # Beggining of sentence
EOS = "###"  # End of Sentence
OOV = "####"  # Out of vocabulary
SEED = 9322238373737383837
emission_matrix = {}  # 2-D dictionary
transmission_matrix = {}  # 2-D dictionary
frequency_of_tags = {}
vocabulary = {}
want_parallelism = 1
num_of_process = 4
K1 = 50
K2 = 40


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
        sentences_in_text[i] = BOS + word_tag_separator + BOS + " " + BOS + word_tag_separator + BOS + " " + \
            sentences_in_text[i] + " " + EOS + word_tag_separator + EOS + " " + EOS + word_tag_separator + EOS

    return sentences_in_text


def split_dataset_into_3_parts(sentences):
    '''
    '''
    random.seed(SEED)
    random.shuffle(sentences)
    number_of_sentences = len(sentences)
    split = int(number_of_sentences * 0.9)
    return sentences[:split], sentences[split:]


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
            transmission_matrix[tag_1][tag_2] = {}
            for tag_3 in frequency_of_tags:
                transmission_matrix[tag_1][tag_2][tag_3] = K2

    for sentence in train_data:
        for i in range(2, len(sentence)):
            tag_1 = sentence[i - 2][1]
            tag_2 = sentence[i - 1][1]
            tag_3 = sentence[i][1]
            transmission_matrix[tag_1][tag_2][tag_3] += 1

    for tag_1 in transmission_matrix:
        for tag_2 in transmission_matrix[tag_1]:
            total = sum(
                transmission_matrix[tag_1][tag_2].values()) + K2 * len(frequency_of_tags)
            for tag_3 in transmission_matrix[tag_1][tag_2]:
                transmission_matrix[tag_1][tag_2][tag_3] = log(
                    transmission_matrix[tag_1][tag_2][tag_3] / total)


def viterbi(sentence):
    number_of_tags = len(frequency_of_tags)
    number_of_tags1_tag2 = number_of_tags**2
    number_of_words = len(sentence)
    viterbi = [[0] * number_of_words for i in range(number_of_tags1_tag2)]
    tags = []
    tag1_tag2 = {}
    idx_to_tag1_tag2 = {}
    c = 0
    for tag_1 in frequency_of_tags:
        tag1_tag2[tag_1] = {}
        tags.append(tag_1)
        for tag_2 in frequency_of_tags:
            tag1_tag2[tag_1][tag_2] = c
            idx_to_tag1_tag2[c] = (tag_1, tag_2)
            c += 1

    # Base Case for 1st word
    for tag_3 in tags:
        word = sentence[2][0]
        i = tag1_tag2[BOS][tag_3]
        j = tag1_tag2[BOS][BOS]
        if word in emission_matrix[tag_3]:
            viterbi[i][2] = [emission_matrix[tag_3][word], j]
        else:
            viterbi[i][2] = [
                log(1 / (K1 * len(vocabulary) + frequency_of_tags[tag_3])), j]

        viterbi[i][2][0] += transmission_matrix[BOS][BOS][tag_3]

    for i in range(number_of_tags1_tag2):
        if viterbi[i][2] == 0:
            viterbi[i][2] = [-10**15, i]

    for k in range(3, number_of_words):
        word = sentence[k][0]
        for tag_2 in tags:
            for tag_3 in tags:
                i = tag1_tag2[tag_2][tag_3]
                c = -10**15
                t = ""
                for j, tag_1 in enumerate(tags):
                    z = transmission_matrix[tag_1][tag_2][tag_3] + \
                        viterbi[tag1_tag2[tag_1][tag_2]][k - 1][0]
                    if z > c:
                        c = z
                        t = tag1_tag2[tag_1][tag_2]
                if word in emission_matrix[tag_3]:
                    c = c + emission_matrix[tag_3][word]
                else:
                    c = c + \
                        log((1 / (K1 * len(vocabulary) + frequency_of_tags[tag_3])))
                viterbi[i][k] = [c, t]

    m = 0
    ans = []
    for i in range(number_of_tags1_tag2):
        if viterbi[m][-2][0] < viterbi[i][-2][0]:
            m = i

    for i in range(number_of_words - 2, 2, -1):
        ans.append(idx_to_tag1_tag2[viterbi[m][i][1]][1])
        m = viterbi[m][i][1]
    ans = ans[::-1]
    return ans


def predict_parallely(test_data):
    correct = wrong = 0
    for sentence in test_data:
        predicted = viterbi(sentence)
        for j in range(2, len(sentence) - 4):
            if sentence[j][1] == predicted[j - 2]:
                correct += 1
            else:
                wrong += 1
    return [correct, wrong]


def predict(sentence):
    correct = wrong = 0
    predicted = viterbi(sentence)
    for j in range(2, len(sentence) - 4):
        if sentence[j][1] == predicted[j - 2]:
            correct += 1
        else:
            wrong += 1
    return [correct, wrong]


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage : python3 trigram.py <file_name>")
        sys.exit()

    dataset = get_dataset(sys.argv[1])
    sentences = get_sentences_from_text(dataset)
    sentences = separate_tag_from_word(sentences)
    train_data, test_data = split_dataset_into_3_parts(sentences)

    preprocessing(train_data)
    calculate_transmission_matrix(train_data)
    calculate_emmission_matrix(train_data)

    correct = wrong = 0
    
    if want_parallelism:
        pool = multiprocessing.Pool(processes=num_of_process)
        start = 0
        n = len(test_data[:300])
        split = n // num_of_process
        x = []
        for i in range(split, n + 1, split):
            x.append(test_data[start:i])
            start = i

        outputs = pool.map(predict_parallely, x)
        pool.terminate()
        for i in outputs:
            correct += i[0]
            wrong += i[1]
    else:
        for sentence in test_data[:50]:
            x = predict(sentence)
            correct += x[0]
            wrong += x[1]
    print(correct, wrong, correct / (correct + wrong))
