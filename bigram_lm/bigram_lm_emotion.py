import sys
sys.path.append('../bpe_tokenizer')

from bpe_tokenizer import my_bpe_tokenizer
 
class Bigram_LM:
    def __init__(self):
        self.bigram_emotion_profile = dict()
        
    def learn(self, tokenized_corpus, labels):
        for i, line in enumerate(tokenized_corpus):
            for j in range(len(line)-1):
                bigram = (line[j], line[j+1])
                emotion = labels[i]
                if bigram in self.bigram_emotion_profile:
                    if emotion in self.bigram_emotion_profile[bigram]:
                        self.bigram_emotion_profile[bigram][emotion] += 1
                    else:
                        self.bigram_emotion_profile[bigram][emotion] = 1
                else:
                    self.bigram_emotion_profile[bigram] = {emotion: 1}
                        
if __name__ == "__main__":
    bigram_lm = Bigram_LM()
    with open("../dataset/corpus.txt") as file:
        tokenized_corpus = [line.split() for line in file.readlines()]
    with open("../dataset/labels.txt") as file:
        labels = [line.split()[0] for line in file.readlines()]
    bigram_lm.learn(tokenized_corpus, labels)
    print(bigram_lm.bigram_emotion_profile)
