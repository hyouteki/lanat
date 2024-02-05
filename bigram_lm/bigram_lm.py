class Bigram_LM:
    def __init__(self):
        self.tokenized_data = []
        self.unigram_counts = dict()
        self.bigram_counts = dict()
        self.bigrams = dict()
        self.vocabulary = set()
        self.bigram_prob = dict()
        
    def learn(self, tokenized_data):
        self.vocabulary=set(token for line in tokenized_data for token in line)
        self.tokenized_data.extend(tokenized_data)
        for line in tokenized_data:
            for i in range(len(line)):
                if line[i] in self.unigram_counts:
                    self.unigram_counts[line[i]] += 1
                    if i < len(line) - 1:
                        if line[i] in self.bigrams:
                            self.bigrams[line[i]].add(line[i + 1])
                        else:
                            self.bigrams[line[i]] = set([line[i + 1]])
                        bigram = (line[i], line[i + 1])
                        if bigram in self.bigram_counts:
                            self.bigram_counts[bigram] += 1
                        else:
                            self.bigram_counts[bigram] = 1
                else:
                    self.unigram_counts[line[i]] = 1
                    if i < len(line) - 1:
                        self.bigrams[line[i]] = set([line[i + 1]])
                        self.bigram_counts[(line[i], line[i + 1])] = 1


    def get_unigram_probability(self, unigram):
        if unigram in self.unigram_counts:
            prob = self.unigram_counts[unigram]
            words = sum([count for _, count in self.unigram_counts.items()])
            return prob/words
        else:
            return 0

    def get_bigram_probability(self, bigram,laplace,kneser_ney):
        if laplace:
            result = self.laplace_bigram_probability_given_first_word(bigram)
        elif kneser_ney:
            result = self.kneser_ney_bigram_probability_given_first_word(bigram)
        else:
            result = (self.bigram_counts[bigram] / self.unigram_counts[bigram[0]]) if bigram in self.bigram_counts else 0
        
        self.bigram_prob[bigram] = result
        
        return result

            
    def laplace_unigram_probability(self, unigram):
        prob = 1 + (self.unigram_counts[unigram] if unigram in self.unigram_counts else 0)
        words = sum([count+1 for _, count in self.unigram_counts.items()])
        return prob/words
    
    def laplace_bigram_probability_given_first_word(self, bigram):
        prob = 1 + (self.bigram_counts[bigram] if bigram in self.bigram_counts else 0)
        words = (self.unigram_counts[bigram[0]]+len(self.vocabulary))
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
    
    # def sentence_probability(self, sentence):
    #     prob = self.unigram_probability(sentence[0])
    #     for i in range(len(sentence)-1):
    #         bigram = (sentence[i], sentence[i+1])
    #         prob *= self.bigram_probability(bigram)
    #     return prob
                        
if __name__ == "__main__":

    with open('lanat\dataset\corpus.txt', 'r') as file:
        data = file.readlines()
    # data = ["this is a  dog",
    #         "this is a cat",
    #         "i love my cat",
    #         "this is my name "]
    tokenized_data = [line.split() for line in data]
    # print(tokenized_data)
    bigram_lm = Bigram_LM()
    
    # print(len(bigram_lm.vocabulary))
    # print(bigram_lm.vocabulary)
    bigram_lm.learn(tokenized_data)
    bigram_lm.bigram_prob = {bigram: bigram_lm.get_bigram_probability(bigram,False,False) for bigram in bigram_lm.bigram_counts}
    print(bigram_lm.bigram_prob)
    print(bigram_lm.unigram_counts)
    print(bigram_lm.bigram_counts)
    print(bigram_lm.bigrams)
    # test_sentence = "i love my cat".split()
    # print(test_sentence)
    # print(bigram_lm.sentence_probability(test_sentence))
