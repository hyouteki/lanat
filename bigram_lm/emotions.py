from bigram_lm import Bigram_LM
import pickle
# import utils
import random

LAPLACE = False
KNESER_NEY = False

with open("../dataset/corpus.txt", "r") as file:
    tokenized_data = Bigram_LM().preprocess_data(file.readlines())
    
models = {"sadness": Bigram_LM(), "joy": Bigram_LM(), "surprise": Bigram_LM(),
          "fear": Bigram_LM(), "anger": Bigram_LM(), "love": Bigram_LM()}

for _, model in models.items():
    model.learn(tokenized_data)

with open("emotion_profiles", "rb") as file:
    emotion_profiles = pickle.load(file)

def meet_operator(prob1, prob2):
    return (prob1 + prob2)/2

for emotion, model in models.items():
    for bigram in model.bigram_counts.keys():
        prob1 = model.calc_bigram_probability(bigram, LAPLACE, KNESER_NEY)
        prob2 = emotion_profiles[emotion][bigram] if bigram in emotion_profiles[emotion] else 0
        model.set_bigram_probability(bigram, meet_operator(prob1, prob2))
    with open(f"{emotion}_probs_plus_half", "wb") as file:
        pickle.dump(model.bigram_probs, file)
        
def sample_next_word(model, previous_token):
    candidate_words = model.bigrams.get(previous_token, [])
    if not candidate_words:
        return None
    candidate_words_list = list(candidate_words)
    next_word_probs = [model.bigram_probs[(previous_token, word)] for word in candidate_words_list]
    return random.choices(candidate_words_list, next_word_probs)[0]

def generate_sentence(model, max_words=10):
    starting_bigram = random.choice(
        [bigram for bigram in model.bigram_counts if bigram[0] == model.starting_token])
    sentence = [starting_bigram[1]]
    for _ in range(max_words - 1):
        next_word = sample_next_word(model, sentence[-1])
        if next_word is None:
            break
        sentence.append(next_word)
    return ' '.join(sentence)

if __name__ == "__main__":
    for emotion, model in models.items():
        print(f"\nGenerated Sentence for {emotion.capitalize()} Model:")
        print(generate_sentence(model))
