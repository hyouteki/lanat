class Bigram_LM:
    def __init__(self):
        self.tokenized_data = []
        self.unigram_counts = dict()
        self.bigram_counts = dict()
        self.bigrams = dict()
        
    def learn(self, tokenized_data):
        self.tokenized_data.extend(tokenized_data)
        for line in tokenized_data:
            for i in range(len(line)):
                if line[i] in self.unigram_counts:
                    self.unigram_counts[line[i]] += 1
                    if i < len(line)-1:
                        self.bigrams[line[i]].add(line[i+1])
                        bigram = (line[i], line[i+1])
                        if bigram in self.bigram_counts:
                            self.bigram_counts[bigram] += 1
                        else:
                            self.bigram_counts[bigram] = 1
                else:
                    self.unigram_counts[line[i]] = 1
                    if i < len(line)-1:
                        self.bigrams[line[i]] = set([line[i+1]])
                        self.bigram_counts[(line[i], line[i+1])] = 1

    def unigram_probability(self, unigram):
        if unigram in self.unigram_counts:
            prob = self.unigram_counts[unigram]
            words = sum([count for _, count in self.unigram_counts.items()])
            return prob/words
        else:
            return 0

    def bigram_probability(self, bigram):
        return self.bigram_counts[bigram]/self.unigram_counts[bigram[0]] if bigram in self.bigram_counts else 0
    
    def sentence_probability(self, sentence):
        prob = self.unigram_probability(sentence[0])
        for i in range(len(sentence)-1):
            bigram = (sentence[i], sentence[i+1])
            prob *= self.bigram_probability(bigram)
        return prob
                        
if __name__ == "__main__":
    data = ["this is a  dog",
            "this is a cat",
            "i love my cat",
            "this is my name "]
    tokenized_data = [line.split() for line in data]
    print(tokenized_data)
    bigram_lm = Bigram_LM()
    bigram_lm.learn(tokenized_data)
    print(bigram_lm.unigram_counts)
    print(bigram_lm.bigram_counts)
    print(bigram_lm.bigrams)
    test_sentence = "i love my cat".split()
    print(test_sentence)
    print(bigram_lm.sentence_probability(test_sentence))
