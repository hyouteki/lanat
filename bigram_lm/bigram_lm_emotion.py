import sys
sys.path.append('../bpe_tokenizer')

from utils import emotion_scores
import pickle

class Bigram_LM:
    def __init__(self):
        self.bigram_emotion_profile = dict()
        self.emotion_profiles = {"joy": dict(), "sadness": dict(), "surprise": dict(),
                                 "fear": dict(), "anger": dict(), "love": dict()}
        
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

    def bigram_emotion_scores(self):
        for bigram in self.bigram_emotion_profile:
            emotion_profile = emotion_scores(" ".join(bigram))
            for emotion_score in emotion_profile:
                self.emotion_profiles[emotion_score["label"]][bigram] = emotion_score["score"]
        with open("emotion_profiles", "wb") as file: 
            pickle.dump(self.emotion_profiles, file) 

    def load_emotion_profiles(self, filename):
        with open(filename, "rb") as file:
            self.emotion_profiles = pickle.load(file)
            
if __name__ == "__main__":
    bigram_lm = Bigram_LM()

    with open("../dataset/corpus.txt") as file:
        tokenized_corpus = [line.split() for line in file.readlines()]
    with open("../dataset/labels.txt") as file:
        labels = [line.split()[0] for line in file.readlines()]

    # tokenized_corpus = [["i", "am", "very", "angry"], ["hey", "you", "there"]]
    # labels = ["angry", "joy"]

    bigram_lm.learn(tokenized_corpus, labels)
    print(bigram_lm.bigram_emotion_profile)
    # bigram_lm.bigram_emotion_scores()
    bigram_lm.load_emotion_profiles("emotion_profiles")
    # print(bigram_lm.emotion_profiles)
