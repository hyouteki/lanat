import pickle
from bigram_lm import Bigram_LM
from utils import emotion_scores

if __name__ == "__main__":
    with open("emotion_profiles", "rb") as file:
        emotion_profiles = pickle.load(file)
    starting_words = set()
    unigram_emotion_profiles = {"joy": dict(), "sadness": dict(), "surprise": dict(),
                        "fear": dict(), "anger": dict(), "love": dict()}
    with open("../dataset/corpus.txt", "r") as file:
        for line in file.readlines():
            starting_words.add(line.split()[0])
    for word in starting_words:
        for emotion_profile in emotion_scores(word):
            unigram_emotion_profiles[emotion_profile["label"]][
                (Bigram_LM().starting_token, word)] = emotion_profile["score"]
    for emotion in emotion_profiles:
        emotion_profiles[emotion].update(unigram_emotion_profiles[emotion])
    with open("emotion_profiles", "wb") as file:
        pickle.dump(emotion_profiles, file)
    
