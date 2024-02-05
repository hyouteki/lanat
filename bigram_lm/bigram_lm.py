class Bigram_LM:
    def __init__(self):
        self.tokenized_data = []
        self.starting_token = "@"
        self.unigram_counts = dict()
        self.bigram_counts = dict()
        self.bigrams = dict()
        self.vocabulary = set()
        self.bigram_probs = dict()

    def preprocess_data(self, data):
        return [[self.starting_token]+line.split() for line in data]
        
    def learn(self, tokenized_data):
        self.vocabulary = set(token for line in tokenized_data for token in line)
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

    def calc_bigram_probability(self, bigram, laplace, kneser_ney):
        if laplace:
            return self.laplace_bigram_probability(bigram)
        elif kneser_ney:
            return self.kneser_ney_bigram_probability(bigram)
        elif bigram in self.bigram_counts:
            return self.bigram_counts[bigram] / self.unigram_counts[bigram[0]]
        return 0

    def set_bigram_probability(self, bigram, probability):
        self.bigram_probs[bigram] = probability

    def laplace_bigram_probability(self, bigram):
        prob = 1 + (self.bigram_counts[bigram] if bigram in self.bigram_counts else 0)
        words = (self.unigram_counts[bigram[0]]+len(self.vocabulary))
        return prob/words

    def kneser_ney_bigram_probability(self, bigram):
        continuation_count = len(self.bigrams[bigram[0]])
        discount = 0.75
        alpha = (discount / self.unigram_counts[bigram[0]]) * continuation_count
        probability_continuation = sum([1 for bigram_t in self.bigram_counts
                                        if bigram_t[1] == bigram[1]]) / len(self.bigram_counts)
        probability = max(self.bigram_counts[bigram] - discount, 0) / self.unigram_counts[bigram[0]]
        probability += alpha * probability_continuation
        return probability
                        
if __name__ == "__main__":
    # with open("../dataset/corpus.txt", "r") as file:
    #     data = file.readlines()

    data = ["this is a  dog",
            "this is a cat",
            "i love my cat",
            "this is my name "]
    
    tokenized_data = [line.split() for line in data]
    print(tokenized_data)
    bigram_lm = Bigram_LM()
    bigram_lm.learn(bigram_lm.preprocess_data(data))
    bigram_lm.bigram_prob = {
        bigram: bigram_lm.get_bigram_probability(bigram, False, False) for bigram in bigram_lm.bigram_counts
    }
    print(bigram_lm.bigram_prob)
    print(bigram_lm.unigram_counts)
    print(bigram_lm.bigram_counts)
    print(bigram_lm.bigrams)
    print(bigram_lm.kneser_ney_bigram_probability(("this", "is")))
    bigram_lm.preprocess_data(data)
