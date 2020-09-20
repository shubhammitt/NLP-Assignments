import re
import sys
from more_itertools import unique_everseen
import nltk
from word2number import w2n
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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
    lemmatizer = WordNetLemmatizer() 

    for word in probable_words:
        w = lemmatizer.lemmatize(word)
        if len(re.findall(r'[a-z0-9]', w, re.I)) > 0:
            words.append(w)

    return words


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
        if re.match('[a-z]', word, re.I) is not None:
            if re.match('[aeiou]', word, re.I) is not None:
                num_of_vowels += 1
            else:
                num_of_consonants += 1

    print("Number of Words starting with a vowel \t\t:-\t ", num_of_vowels)
    print(
        "Number of Words starting with a consonant \t:-\t ",
        num_of_consonants,
        '\n')


def Q_3(text):
    '''
    Question:
        List all the email ids in the file given as input.
    Assumptions :
        Broke the text into sentences as done in Q1.
            ID starts with alphanum.
            Only contain alphanum, !#$%&’*+-/=?^_{|}~ in prefix (part before @)
            Prefix is of length atleast 1.
            @ can be followed by only alphanum(no special characters).
            only alphanum is expected before @.
            Exaclty one @.
            Atleast a dot after @.
            no dot before @.
            No consecutive dots in ID
            Max length of ID is 64
            Ends with alphanum.
            Duplicate IDs are not printed and counted.
            If a ID is invalid but if its substring is a valid ID then it is chosen.
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

    mail_id = list(unique_everseen(mail_id))  # remove duplicates
    print("Number of valid mail-IDs = ", len(mail_id), '\n')

    for i, id in enumerate(mail_id):
        print(str(i + 1) + '.\t', id)


def Q_4(text):
    '''
    Question:
        Print the sentences and number of sentences starting with a given word in an input file.
    Assumption:
        Case insensitive
        if entered word is a sentence then we take the first word of it
        numerical search allowed
    '''
    print("\n\nQ4 : \n")

    sentences = break_text_into_sentences(text)
    count = 0

    word = input("Enter a word : ")
    word = break_sentence_into_words(word)[0]
    word = word.lower()
    word = word_to_num(word)

    print("\nWord which will be searched in file (case-insensitive) \t=\t", word, '\n')

    for sentence in sentences:
        words = break_sentence_into_words(sentence)
        if len(words) > 0 and word_to_num(words[0].lower()) == word:
            count += 1
            print(str(count) + '.\t', sentence)

    print("\nNumber of sentences which starts with entered word \"" +
          str(word) + "\"\t\t=\t\t ", count, '\n')


def Q_5(text):
    '''
    Question:
        Print the sentences and number of sentences ending with a given word in an input file.
    Assumption:
        Case insensitive
        if entered word is a sentence then we take the first word of it
        numerical search allowed
    '''
    print("\n\nQ5 : \n")

    sentences = break_text_into_sentences(text)
    count = 0

    word = input("Enter a word : ")
    word = break_sentence_into_words(word)[0]
    word = word.lower()
    word = word_to_num(word)

    print(
        "\nWord which will be searched in file (case-insensitive) \t=\t",
        word,
        '\n')

    for sentence in sentences:
        words = break_sentence_into_words(sentence)
        if len(words) > 0 and word_to_num(words[-1].lower()) == word:
            count += 1
            print(str(count) + '.\t', sentence)

    print("\nNumber of sentences which ends with entered word \"" +
          str(word) + "\"\t\t=\t\t ", count, '\n')


def Q_6(text):
    '''
    Question:
        Given a word and a file as input, print the count of that word and sentences containing
        that word in the input file.
    Assumptions:
        Numerical search allowed
        Case-insensitive
    '''
    print("\n\nQ6 : \n")

    sentences = break_text_into_sentences(text)
    num_of_words = num_of_sentences = 0

    word = input("Enter a word : ")
    word = word.lower()
    word = re.findall(r'\w+', word)[0]
    word = word_to_num(word)

    print(
        "\nWord which will be searched in file (case-insensitive) \t=\t",
        word,
        '\n')

    for sentence in sentences:
        words = break_sentence_into_words(sentence.lower())

        for i in range(len(words)):
            words[i] = word_to_num(words[i])

        w = words.count(word)
        if w > 0:
            num_of_sentences += 1
            num_of_words += w
            print(w, "times in the sentence --> \n" + sentence, "\n\n")

    print("\n\nTotal count of words \t\t\t\t =\t ", num_of_words)
    print(
        "Number of sentences containing entered word \t =\t\t",
        num_of_sentences,
        '\n\n')


def Q_7(text):
    '''
    Question:
        Given an input file, print the questions present, if any, in that file.
    '''
    print("\n\nQ7 : \n")

    count = 0
    sentences = break_text_into_sentences(text)
    for sentence in sentences:
        if sentence.endswith('?'):
            count += 1
            print(str(count) + '.\t', sentence)

    print("\nNumber of questions in given file \t=\t", count, '\n\n')


def Q_8(text):
    '''
    Question:
        List the minutes and seconds mentioned in the date present in the file given as input.
        For instance, for the date - Tue, 20 Apr 1993 17:51:16 GMT, the output should be 51
        min, 16 sec)
    '''
    print("\n\nQ8 : \n")

    sentences = break_text_into_sentences(text)

    for sentence in sentences:

        times = re.findall(r'\b[01][0-9]:[0-5][0-9]:[0-5][0-9]\b', sentence)
        times.extend(re.findall(r'\b2[0-3]:[0-5][0-9]:[0-5][0-9]\b', sentence))

        for time in times:

            hr, mi, se = time.split(':')
            print(time, " --> ", mi, "min", se, "sec")

    print('\n\n')


def Q_9(text):
    '''
    Question:
        List the abbreviations present in a file given as input.
    Assumption:
        All uppercase.
        Atleast length 2
    '''
    print("\n\nQ9 : \n")

    sentences = break_text_into_sentences(text)
    abbreviations = []

    for sentence in sentences:
        abbreviations.extend(re.findall(r'\b[A-Z]{2,}\b', sentence))

    for i in abbreviations:
        print(i)

    print('\n\n')


if __name__ == "__main__":

    if(len(sys.argv) != 2):
        print("Usage : python3 Assn1.py <File_Path>")
        sys.exit()

    try:
        file = open(sys.argv[1], "r")
        text = file.read()  # type(text) = str)
        file.close()
    except BaseException:
        print("File not found!!")
        sys.exit()

    Q_1(text)
    Q_2(text)
    Q_3(text)
    Q_4(text)
    Q_5(text)
    Q_6(text)
    Q_7(text)
    Q_8(text)
    Q_9(text)
