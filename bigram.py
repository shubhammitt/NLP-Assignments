import sys
import random
from math import log
import multiprocessing

word_tag_separator = '_'
BOS = "#"  # Beggining of sentence
EOS = "@"  # End of Sentence
OOV = "^"  # Out of vocabulary
SEED = 12332258
emission_matrix = {}  # 2-D dictionary
transmission_matrix = {}  # 2-D dictionary
frequency_of_tags = {}
size_of_vocabulary = 0
K1 = 40
K2 = 50
want_parallelism = 1
num_process = 5
number_of_tags = 0
test_ID = 0
assumed_max_length_sent = 100
model_accuracy = []
model_precision = []
model_recall = []
model_f1 = []
all_fold_tags = {}

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

    for tag in frequency_of_tags:
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
        except BaseException:
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
            except BaseException:
                max_val -= log(K1 * size_of_vocabulary +
                               frequency_of_tags[tag])

            viterbi[k][i] = (max_val, col_idx)

    col_idx = 0
    ans = []
    for i in range(number_of_tags):
        if viterbi[number_of_words -
                   1][col_idx][0] < viterbi[number_of_words - 1][i][0]:
            col_idx = i

    for i in range(number_of_words - 1, 1, -1):
        ans.append(tags[viterbi[i][col_idx][1]])
        col_idx = viterbi[i][col_idx][1]

    return ans[::-1]


def predict_parallely(test_data):
    ans_to_return = [0, 0]
    for sentence in test_data:
        predicted = viterbi_algo(sentence)
        ans_to_return.append((sentence[1:-1], predicted))
        for j in range(1, len(sentence) - 1):
            if sentence[j][1] == predicted[j - 1]:
                ans_to_return[0] += 1
            else:
                ans_to_return[1] += 1
    return ans_to_return


def predict(sentence):
    correct = wrong = 0
    predicted = viterbi_algo(sentence)
    for j in range(1, len(sentence) - 1):
        if sentence[j][1] == predicted[j - 1]:
            correct += 1
        else:
            wrong += 1
    return [correct, wrong, (sentence[1:-1], predicted)]


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage : python3 bigram.py <file_name>")
        sys.exit()

    dataset = get_dataset(sys.argv[1])
    sentences = get_sentences_from_text(dataset)
    sentences = separate_tag_from_word(sentences)
    dataset = split_dataset_into_3_parts(sentences)

    for i in range(3):
        test_ID = i
        print("Test ID = ", test_ID)
        p = [0, 1, 2]
        p.remove(i)
        train_data, test_data = make_train_and_test_data(
            dataset, p[0], p[1], i)
        preprocessing(train_data)
        calculate_emmission_matrix(train_data)
        calculate_transmission_matrix(train_data)
        confusion_matrix = {}
        for tag_1 in tags:
            confusion_matrix[tag_1] = {}
            for tag_2 in tags:
                confusion_matrix[tag_1][tag_2] = 0

        correct = wrong = 0
        if want_parallelism:
            pool = multiprocessing.Pool(processes=num_process)
            start = 0
            n = len(test_data)
            split = n // num_process
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
                        confusion_matrix[i[j][0][l][1]][i[j][1][l]] += 1
            print(correct, wrong, correct / (correct + wrong), K1, K2)
        else:
            for sentence in test_data[:1000]:
                x = predict(sentence)
                correct += x[0]
                wrong += x[1]
                for l in range(len(x[2][0])):
                    confusion_matrix[x[2][0][l][1]][x[2][1][l]] += 1

            print(correct, wrong, correct / (correct + wrong), K1, K2)

        print("-------------------------------------------------------------Confusion Matrix-------------------------------------------------------------\n")
        for tag_1 in tags:
            for tag_2 in tags:
                if tag_1 != BOS and tag_1 != EOS and tag_2 != BOS and tag_2 != EOS:
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
        print("Test ID = ", test_ID)
        print("\n\n----------------------------------------------------------------------TAGWISE----------------------------------------------------------------------")
        
        all_fold_tags[test_ID]  = {}
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

                all_fold_tags[test_ID][tag] = [accuracy[tag], precision[tag], recall[tag], f1_score[tag]]
                print(
                    "Tag = ",
                    tag,
                    "\tAccuracy = ",
                    accuracy[tag],
                    "\tPrecision = ",
                    precision[tag],
                    "\tRecall = ",
                    recall[tag],
                    "\tF1- score = ",
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

        model_f1.append(full_f1_macro)
        model_recall.append(full_recall)
        model_precision.append(full_precision)
        model_accuracy.append(full_accuracy)

        print(
            "Accuracy = ",
            full_accuracy,
            "\tPrecision = ",
            full_precision,
            "\tRecall = ",
            full_recall,
            "\tF1- score = ",
            full_f1_macro)

    print("\n\n----------------------------------------------------------------- Model Stats -----------------------------------------------------------------")
    
    for tag in all_fold_tags[2]:
    	c = 1
    	idx = 0
    	x = all_fold_tags[2][tag][idx]
    	if tag in all_fold_tags[0]:
    		x += all_fold_tags[0][tag][idx]
    		c += 1
    	if tag in all_fold_tags[1]:
    		x += all_fold_tags[1][tag][idx]
    		c += 1
    	print("Tag = ", tag, "Accuracy = ", x / c, end ='')
    	c = 1
    	idx = 1
    	x = all_fold_tags[2][tag][idx]
    	if tag in all_fold_tags[0]:
    		x += all_fold_tags[0][tag][idx]
    		c += 1
    	if tag in all_fold_tags[1]:
    		x += all_fold_tags[1][tag][idx]
    		c += 1
    	print("\t\tPrecision = ", x / c, end ='')
    	c = 1
    	idx = 2
    	x = all_fold_tags[2][tag][idx]
    	if tag in all_fold_tags[0]:
    		x += all_fold_tags[0][tag][idx]
    		c += 1
    	if tag in all_fold_tags[1]:
    		x += all_fold_tags[1][tag][idx]
    		c += 1
    	print("\t\tRecall = ", x / c, end ='')
    	c = 1
    	idx = 3
    	x = all_fold_tags[2][tag][idx]
    	if tag in all_fold_tags[0]:
    		x += all_fold_tags[0][tag][idx]
    		c += 1
    	if tag in all_fold_tags[1]:
    		x += all_fold_tags[1][tag][idx]
    		c += 1
    	print("\t\tF1-score = ", x / c)

    print("Average Model Accuracy = ", sum(model_accuracy) / 3)
    print("Average Model Precision = ", sum(model_precision) / 3)
    print("Average Model Recall = ", sum(model_recall) / 3)
    print("Average Model F1-score = ", sum(model_f1) / 3)
