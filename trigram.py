
import sys
import random
import multiprocessing
from math import log

word_tag_separator = '_'
BOS = "#"  # Beggining of sentence
EOS = "@"  # End of Sentence
OOV = "^"  # Out of vocabulary
SEED = 12332258
emission_matrix = {}  # 2-D dictionary
transmission_matrix = {}  # 2-D dictionary
frequency_of_tags = {}
vocabulary = {}
want_parallelism = 1
num_of_process = 4
K1 = 50
K2 = 40
viterbi = []
assumed_max_length_sent = 100
number_of_tags = 0
number_of_tags1_tag2 = 0
tag1_tag2 = {}
idx_to_tag1_tag2 = {}
tags = []


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
            sentences_in_text[i] + " " + EOS + word_tag_separator + \
            EOS + " " + EOS + word_tag_separator + EOS

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
    global frequency_of_tags, vocabulary, number_of_tags, number_of_tags1_tag2, viterbi, tag1_tag2, idx_to_tag1_tag2, tags
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

    number_of_tags = len(frequency_of_tags)
    number_of_tags1_tag2 = number_of_tags**2

    viterbi = [
        [0] *
        assumed_max_length_sent for i in range(number_of_tags1_tag2)]
    c = 0
    for tag_1 in frequency_of_tags:
        tag1_tag2[tag_1] = {}
        tags.append(tag_1)
        for tag_2 in frequency_of_tags:
            tag1_tag2[tag_1][tag_2] = c
            idx_to_tag1_tag2[c] = (tag_1, tag_2)
            c += 1


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


def viterbi_algo(sentence):
    global assumed_max_length_sent
    number_of_words = len(sentence)

    if number_of_words > assumed_max_length_sent:
        for i in range(number_of_tags1_tag2):
            for j in range(assumed_max_length_sent, number_of_words):
                viterbi[i].append(0)
        assumed_max_length_sent = number_of_words

    # Base Case for 1st word
    for tag_3 in tags:
        word = sentence[2][0]
        i = tag1_tag2[BOS][tag_3]
        j = tag1_tag2[BOS][BOS]
        c = 0
        if word in emission_matrix[tag_3]:
            c = emission_matrix[tag_3][word]
        else:
            c = -log(K1 * len(vocabulary) + frequency_of_tags[tag_3])

        viterbi[i][2] = (c + transmission_matrix[BOS][BOS][tag_3], j)

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
                    c = c - \
                        log(((K1 * len(vocabulary) +
                              frequency_of_tags[tag_3])))
                viterbi[i][k] = (c, t)

    m = 0
    ans = []
    for i in range(number_of_tags1_tag2):
        if viterbi[m][number_of_words -
                      2][0] < viterbi[i][number_of_words - 2][0]:
            m = i

    for i in range(number_of_words - 2, 2, -1):
        ans.append(idx_to_tag1_tag2[viterbi[m][i][1]][1])
        m = viterbi[m][i][1]
    ans = ans[::-1]
    return ans


def predict_parallely(test_data):
    ans_to_return = [0, 0]
    for sentence in test_data:
        predicted = viterbi_algo(sentence)
        ans_to_return.append((sentence[2:-2], predicted))
        for j in range(2, len(sentence) - 4):
            if sentence[j][1] == predicted[j - 2]:
                ans_to_return[0] += 1
            else:
                ans_to_return[1] += 1
    return ans_to_return


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
    confusion_matrix = {}
    for tag_1 in tags:
        confusion_matrix[tag_1] = {}
        for tag_2 in tags:
            confusion_matrix[tag_1][tag_2] = 0

    correct = wrong = 0

    pool = multiprocessing.Pool(processes=num_of_process)
    start = 0
    n = len(test_data[:100])
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
        for j in range(2, len(i)):
            for l in range(len(i[j][0])):
                # print(i[j][0])
                confusion_matrix[i[j][0][l][1]][i[j][1][l]] += 1
    print(correct, wrong, correct / (correct + wrong))

    print("-------------------------------------------------------------Confusion Matrix-------------------------------------------------------------\n")
    for tag_1 in tags:
        for tag_2 in tags:
            if tag_1 != BOS and tag_1 != EOS and tag_2 != BOS and tag_2 != EOS:
                # print(tag_1, tag_2)
                print(tag_1, tag_2, confusion_matrix[tag_1][tag_2])

    # print("-------------------------------------------------------------True Positives----------------------------------------------------------------\n")
    true_positives = {}
    for tag in tags:
        if tag != BOS and tag != EOS:
            true_positives[tag] = confusion_matrix[tag][tag]
            # print(tag, true_positives[tag])

    # print("-------------------------------------------------------------False Positives----------------------------------------------------------------\n")
    false_positives = {}
    for tag in tags:
        if tag != BOS and tag != EOS:
            false_positives[tag] = -true_positives[tag]
            for tag_1 in tags:
                if tag_1 != BOS and tag_1 != EOS:
                    false_positives[tag] += confusion_matrix[tag_1][tag]
            # print(tag, false_positives[tag])

    # print("-------------------------------------------------------------False Negatives----------------------------------------------------------------\n")
    false_negatives = {}
    for tag in tags:
        if tag != BOS and tag != EOS:
            false_negatives[tag] = -true_positives[tag]
            for tag_1 in tags:
                if tag_1 != BOS and tag_1 != EOS:
                    false_negatives[tag] += confusion_matrix[tag][tag_1]
            # print(tag, false_negatives[tag])
    accuracy = {}
    precision = {}
    recall = {}
    f1_score = {}
    print("\n\n----------------------------------------------------------------------TAGWISE----------------------------------------------------------------------")
    for tag in tags:
        if tag != BOS and tag != EOS:
            if true_positives[tag] + false_negatives[tag] + \
                    false_positives[tag] != 0:
                accuracy[tag] = true_positives[tag] / \
                    (true_positives[tag] +
                     false_negatives[tag] +
                     false_positives[tag])
            else:
                accuracy[tag] = 0

            if true_positives[tag] + false_positives[tag] != 0:
                precision[tag] = true_positives[tag] / \
                    (true_positives[tag] + false_positives[tag])
            else:
                precision[tag] = 0
            if true_positives[tag] + false_negatives[tag] != 0:
                recall[tag] = true_positives[tag] / \
                    (true_positives[tag] + false_negatives[tag])
            else:
                recall[tag] = 0
            if precision[tag] + recall[tag] != 0:

                f1_score[tag] = 2 * precision[tag] * \
                    recall[tag] / (precision[tag] + recall[tag])
            else:
                f1_score[tag] = 0

            print(
                "Tag = ",
                tag,
                "\t\tAccuracy = ",
                accuracy[tag],
                "\t\tPrecision = ",
                precision[tag],
                "\t\tRecall = ",
                recall[tag],
                "\t\tF1- score = ",
                f1_score[tag])

    print("\n\n------------------------------------------------------------- OverAll ------------------------------------------------------------------")

    s = 0
    for tag in true_positives:
        s += frequency_of_tags[tag]

    full_accuracy = 0
    full_precision = 0
    full_recall = 0
    full_f1_macro = 0
    for tag in accuracy:
        full_accuracy += accuracy[tag] * frequency_of_tags[tag] / s
        full_recall += recall[tag] * frequency_of_tags[tag] / s
        full_precision += precision[tag] * frequency_of_tags[tag] / s
        full_f1_macro += f1_score[tag] * frequency_of_tags[tag] / s

    print(
        "Accuracy = ",
        full_accuracy,
        "\t\tPrecision = ",
        full_precision,
        "\t\tRecall = ",
        full_recall,
        "\t\tF1- score = ",
        full_f1_macro)
