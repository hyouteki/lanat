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
            # TODO: need to add laplace or some other kind of softening \
            #    algorithm to deal with zero probability
            return 0

    def bigram_probability_given_first_word(self, bigram):
        if bigram in self.bigram_counts:
            prob = self.bigram_counts[bigram]
            words = sum([self.bigram_counts[(bigram[0], word2)]
                         for word2 in self.bigrams[bigram[0]]])
            return prob/words
        else:
            # TODO: need to add laplace or some other kind of softening \
            #    algorithm to deal with zero probability
            return 0

    def laplace_unigram_probability(self, unigram):
        prob = 1 + (self.unigram_counts[unigram] if unigram in self.unigram_counts else 0)
        words = sum([count+1 for _, count in self.unigram_counts.items()])
        return prob/words
    
    def laplace_bigram_probability_given_first_word(self, bigram):
        prob = 1 + (if bigram in self.bigram_counts self.bigram_counts[bigram] else 0)
        words = sum([self.bigram_counts[(bigram[0], word2)]+1 for word2 in self.bigrams[bigram[0]]])
        return prob/words
    
    def kneser_ney_unigram_probability(self, unigram):
        total_unigrams = sum(self.unigram_counts.values())
        prob = max(self.unigram_counts[unigram] - 0.5, 0) / total_unigrams
        return prob

    def kneser_ney_bigram_probability_given_first_word(self, bigram):
        continuation_count = sum(1 for w2 in self.bigram_counts if w2[0] == bigram[0])

        discount = 0.75
        alpha_factor = discount / self.unigram_counts[bigram[0]]

        prob = max(self.bigram_counts[bigram] - discount, 0) / self.unigram_counts[bigram[0]]
        prob += (alpha_factor * continuation_count)
        return prob
    
        
    def laplace_sentence_probability(self, sentence):
        prob = self.laplace_unigram_probability(sentence[0])
        for i in range(len(sentence)-1):
            bigram = (sentence[i], sentence[i+1])
            prob *= self.laplace_bigram_probability_given_first_word(bigram)
        return prob
    
    def kneser_ney_sentence_probability(self, sentence):
        prob = self.kneser_ney_unigram_probability(sentence[0])
        for i in range(len(sentence)-1):
            bigram = (sentence[i], sentence[i+1])
            prob *= self.kneser_ney_bigram_probability_given_first_word(bigram)
        return prob

    def sentence_probability(self, sentence):
        prob = self.unigram_probability(sentence[0])
        for i in range(len(sentence)-1):
            bigram = (sentence[i], sentence[i+1])
            prob *= self.bigram_probability_given_first_word(bigram)
        return prob
        
                        
if __name__ == "__main__":
    data = ["This is a  dog",
            "This is a cat",
            "I love my cat",
            "This is my name "]
    tokenized_data = [line.split() for line in data]
    bigram_lm = Bigram_LM()
    bigram_lm.learn(tokenized_data)
    print(bigram_lm.unigram_counts)
    print(bigram_lm.bigram_counts)
    print(bigram_lm.bigrams)
    print(bigram_lm.sentence_probability("This is a  dog".split()))
    print(bigram_lm.laplace_sentence_probability("This is a  dog".split()))
    print(bigram_lm.kneser_ney_sentence_probability("This is a  dog".split()))
