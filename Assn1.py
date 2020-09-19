from  more_itertools import unique_everseen
import re
import sys
import nltk
from word2number import w2n
from nltk.tokenize import word_tokenize


def break_text_into_sentences(text):
    '''
    Args:
        text - the text which needs to be broken into sentences.
    Function:
        Break the 'text' into sentences after replacing every newline character by a whitespace.
    returns:
        a list of sentences.
    '''
    text = text.replace('\n', ' ')
    sentences = nltk.sent_tokenize(text)
    return sentences


def break_sentence_into_words(sentence):
    '''
    Args:
            sentence - a string.
    Function:
            Break the 'sentence' into words using word_tokenize
            and then checking if the word contains atleast an alphanum
            for it to be a valid word.
    return:
            a list of words.
    '''
    words = []
    probable_words = word_tokenize(sentence)
    for word in probable_words:
        if len(re.findall(r'[a-z0-9]', word, re.I)) > 0:
            words.append(word)
    return words


def Q_1(text):
    '''
    Question:
        Print the number of words and sentences contained in text.
    '''
    print("\n\nQ1 : \n")

    sentences = break_text_into_sentences(text)
    num_of_sentences = len(sentences)
    words = []
    num_of_words = 0

    for sentence in sentences:
        words.append(break_sentence_into_words(sentence))
        num_of_words += len(words[-1])

    print("Number of Sentences \t:-\t ", num_of_sentences)
    print("Number of Words \t:-\t ", num_of_words, '\n')


def Q_2(text):
    '''
    Question:
        Print the number of words starting with consonants and the number of words starting
        with vowels in the file given as input.
    '''
    print("\n\nQ2 : \n")

    sentences = break_text_into_sentences(text)
    words = []
    num_of_vowels = num_of_consonants = 0

    for sentence in sentences:
        words.extend(break_sentence_into_words(sentence))

    for word in words:
        if re.match('[a-z]', word, re.I) != None:
            if re.match('[aeiou]', word, re.I) != None:
                num_of_vowels += 1
            else:
                num_of_consonants += 1

    print("Number of Words starting with a vowel \t\t:-\t ", num_of_vowels)
    print("Number of Words starting with a consonant \t:-\t ", num_of_consonants, '\n')


def Q_3(text):
    '''
    Question:
    	List all the email ids in the file given as input.
    Assumptions :
        start with a alphanum
        only contain alphanum, !#$%&’*+-/=?^_{|}~ in prefix before @
        prefix length is atleast 1
        @ can be followed by only alphanum
        only alphanum is expected before @
        only contain alphanum in domain and dots(in suffix) and hyphen
        atleast a dot in domain
        does not end with dot or hyphen
        max len of ID is 64
        no consecutive dots in mail ID
        no dot before @
        Exaclty one @
        Atleast a dot after @
    '''
    print("\n\nQ3 : \n")

    sentences = break_text_into_sentences(text)
    mail_id = []
    for sentence in sentences:
        mail_id.extend(
            [
                id for id in re.findall(
                    r'\b[a-zA-Z0-9][a-zA-Z0-9!#$%&’\*\+\-/=?^_{\|}~]*[a-zA-Z0-9]*@[a-zA-Z0-9][a-zA-Z0-9-]*\.[a-zA-Z0-9\.\-]*[a-zA-Z0-9]\b',
                    sentence,
                    re.I) if len(id) < 65 and ".." not in id])

    mail_id = list(unique_everseen(mail_id)) # remove duplicates
    print("Number of valid mail-IDs = ", len(mail_id), '\n')

    for i, id in enumerate(mail_id):
        print(str(i + 1) + '.',id)

def Q_4(text):
    '''
    Print the sentences and number of sentences starting with a given word in an input file.
    Case insensitive
    '''
    sentences = break_text_into_sentences(text)
    count = 0
    word = input("Enter a word : ")
    word = word.lower()
    word = re.findall(r'\w+', word)[0]
    for sentence in sentences:
        words = re.findall(r'\w+', sentence)
        if len(words) > 0 and words[0].lower() == word:
            print(sentence)
            count += 1
    print("Number of sentences which starts with entered word \t\t=\t\t ", count)


def Q_5(text):
    '''
    Print the sentences and number of sentences ending with a given word in an input file.
    Case insensitive
    '''
    sentences = break_text_into_sentences(text)
    count = 0
    word = input("Enter a word : ")
    word = word.lower()
    word = re.findall(r'\w+', word)[0]

    for sentence in sentences:
        words = re.findall(r'\w+', sentence)
        if len(words) > 0 and words[-1].lower() == word:
            print(sentence)
            count += 1
    print("Number of sentences which starts with entered word \t\t=\t\t ", count)


def word_to_num(word):
    '''
    '''
    if word.startswith('0'):
        return word

    try:
        word = str(w2n.word_to_num(word))
    except BaseException:
        word = word
    return word


def Q_6(text):
    '''
            Given a word and a file as input, print the count of that word and sentences containing
            that word in the input file.
    '''
    sentences = break_text_into_sentences(text)
    num_of_words = num_of_sentences = 0

    word = input("Enter a word : ")
    word = word.lower()
    word = re.findall(r'\w+', word)[0]
    word = word_to_num(word)

    for sentence in sentences:
        words = re.findall(r'\w+', sentence.lower())
        for i in range(len(words)):
            words[i] = word_to_num(words[i])

        w = words.count(word)
        if w > 0:
            num_of_sentences += 1
            num_of_words += w
            print(w, "times in the sentence --> \n" + sentence, "\n")

    print("Total count of words = ", num_of_words)
    print("Total count of sentences containing enetered word = ", num_of_sentences)


def Q_7(text):
    '''
            Given an input file, print the questions present, if any, in that file.
    '''
    sentences = break_text_into_sentences(text)
    for sentence in sentences:
        if sentence.endswith('?'):
            print(sentence)


def Q_8(text):
    '''
            List the minutes and seconds mentioned in the date present in the file given as input.
            (For instance, for the date - Tue, 20 Apr 1993 17:51:16 GMT, the output should be 51
            min, 16 sec)
    '''
    sentences = break_text_into_sentences(text)
    for sentence in sentences:
        times = re.findall(r'\b[01]?[0-9]:[0-5][0-9]:[0-5][0-9]\b', sentence)
        times.extend(re.findall(r'\b2[0-3]:[0-5][0-9]:[0-5][0-9]\b', sentence))
        for time in times:
            hr, mi, se = time.split(':')
            print(time, " --> ", mi, "min", se, "sec")


def Q_9(text):
    '''
            List the abbreviations present in a file given as input.
    '''
    sentences = break_text_into_sentences(text)
    abbreviations = []
    for sentence in sentences:
        # abbreviations.extend(re.findall(r'\b([A-Z]\.){2,}\b', sentence))
        abbreviations.extend(re.findall(r'\b[A-Z]{2,}\b', sentence))

    for i in abbreviations:
        print(i)


if __name__ == "__main__":

    if(len(sys.argv) != 2):
        print("Usage : python3 Assn1.py <File_Path>")
        sys.exit()

    try:
        file = open(sys.argv[1], "r")
        text = file.read()  # type(text) = str)
        # print(text)
        file.close()
    except BaseException:
        print("File not found!!")
        sys.exit()

    Q_1(text)
    Q_2(text)
    Q_3(text)
    # Q_4(text)
    # Q_5(text)
    # Q_6(text)
    # Q_7(text)
    # Q_8(text)
    # Q_9(text)
