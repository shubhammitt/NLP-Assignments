import re
import sys 


def Q_1(text):
	'''
		Print the number of words and sentences contained in the file given as input.
	'''

def Q_2(text):
	'''
		Print the number of words starting with consonants and the number of words starting
		with vowels in the file given as input.
	'''

def Q_3(text):
	'''
		List all the email ids in the file given as input.
	'''

def Q_4(text):
	'''
		Print the sentences and number of sentences starting with a given word in an input file.
	'''

def Q_5(text):
	'''
		Print the sentences and number of sentences ending with a given word in an input file.
	'''

def Q_6(text):
	'''
		Given a word and a file as input, print the count of that word and sentences containing
		that word in the input file.
	'''

def Q_7(text):
	'''
		Given an input file, print the questions present, if any, in that file.
	'''

def Q_8(text):
	'''
		List the minutes and seconds mentioned in the date present in the file given as input.
		(For instance, for the date - Tue, 20 Apr 1993 17:51:16 GMT, the output should be 51
		min, 16 sec)
	'''

def Q_9(text):
	'''
		List the abbreviations present in a file given as input.
	'''

if __name__ == "__main__" :

	if(len(sys.argv) != 2):
		print("Usage : python3 Assn1.py <File_Path>")
		exit(1)

	try:
		file = open(sys.argv[1], "r")
		text = file.read() # type(text) = str
		Q_1(text)
		# Q_2(text)
		# Q_3(text)
		# Q_4(text)
		# Q_5(text)
		# Q_6(text)
		# Q_7(text)
		# Q_8(text)
		# Q_9(text)
	except:
		print("File not found!!")

