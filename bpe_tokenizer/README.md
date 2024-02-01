## Documentation
``` python
class BPE_Tokenizer:
	''' tokenizes the file based on byte pair encoding '''
	
	Attributes:
		filename (str): path to the corpus
		word_freq_pairs (list[list[list[str], int]]): pairs of tokenized word and its frequency in the corpus
		vocabulary (list[str]): set of tokenized words in the corpus
		n_merges (int): number of merges for tokenization loop sequence
		merge_rules(list[type[str, str]]): stores the merge rules in order
	
	def __init__(self):
        ''' constructor for the BPE_Tokenizer class '''
	
	def learn_vocabulary(self, filename, n_merges):
		''' learns the split rules and frequencies based on the corpus and n_merges '''
		''' IMPORTANT: does not override previous learn_vocabulary calls '''
		Attributes:
			filename (str): path to the corpus
			n_merges (int): number of merges for tokenization loop sequence
	
	def save_merge_rules(self, filename):
		''' saves the learnt merge_rules to a file in order '''
		Attributes:
			filename (str): path at which the merge_rules will be stored
```
