import nltk

vocabulary = {}
all_tags = {}
list_of_len_sentences = []

def get_sentences_from_text(file_name):
	file = open(file_name, "r")
	text = file.read()  # type(text) = str)
	file.close()
	sentences_in_text = list(text.split('\n'))
	return sentences_in_text


def split_sentence_into_word_tag(sentence):
	words_in_sentence = list(sentence.split())
	for i in words_in_sentence:
		separate_tag_from_word(i)


def separate_tag_from_word(word_tag):
	try:
		idx = word_tag.rindex('_')
		word = word_tag[ : idx]
		tag = word_tag[idx + 1 :]
	except:
		word = word_tag
		tag = "OOV"

	if word not in vocabulary:
		vocabulary[word] = {}

	if tag not in all_tags:
		all_tags[tag] = 0

	if tag not in vocabulary[word]:
		vocabulary[word][tag] = 0
	vocabulary[word][tag] += 1
	all_tags[tag] += 1

	return (word, tag)

def statistics():
	print("Number of words in vocabulary \t\t:", len(vocabulary))
	print("Number of tags \t\t\t:", len(all_tags))
	print("Min of sentence \t\t:", min(list_of_len_sentences))
	print("Max of sentence \t\t:", max(list_of_len_sentences))
	print("Avg of sentence \t\t:", sum(list_of_len_sentences) / len(list_of_len_sentences))
	print("Number of tags \t\t:", len(all_tags))
	print("Total words \t\t:", sum(list_of_len_sentences))
	words_with_more_tags = 0
	print("-"*60, '\n\n')
	print("Frequency of tags in dataset:")
	for i in all_tags:
		print(i, all_tags[i])

	print("-"*60, '\n\n')
	print("Word with more than 1 tag:")
	for i in vocabulary:
		if len(vocabulary[i]) > 1:
			words_with_more_tags += 1
			print(i, vocabulary[i])
	print("Words with more than 1 tag \t\t:", words_with_more_tags)

sentences_in_text = get_sentences_from_text("Brown_train.txt")
for sentence in sentences_in_text:
	if len(sentence) == 0:
		continue
	list_of_len_sentences.append(len(sentence))
	split_sentence_into_word_tag(sentence)
statistics()
