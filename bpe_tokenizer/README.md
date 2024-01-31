## Documentation
``` python
class BPE_Tokenizer:
	''' tokenizes the file based on byte pair encoding '''
	
	Attributes:
		filename (str): path to the corpus
		word_freq_pairs (list[list[list[str], int]]): pairs of tokenized word and its frequency in the corpus
		vocabulary (list[str]): set of tokenized words in the corpus
		iterations (int): number of iterations for tokenization loop sequence
	
	def __init__(self, filename, iterations):
        ''' constructor for the BPE_Tokenizer class '''
```
