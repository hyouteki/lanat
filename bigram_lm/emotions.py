from bigram_lm import Bigram_LM
import pickle
import random

LAPLACE = True
KNESER_NEY = False
MIN_SENTENCE_LEN = 15
MAX_SENTENCE_LEN = 25
THRESHOLD_1 = 0.5
THRESHOLD_2 = 0.7
THRESHOLD_3 = 0.9

with open("../dataset/corpus.txt", "r") as file:
    tokenized_data = Bigram_LM().preprocess_data(file.readlines())
    
models = {"sadness": Bigram_LM(), "joy": Bigram_LM(), "surprise": Bigram_LM(),
          "fear": Bigram_LM(), "anger": Bigram_LM(), "love": Bigram_LM()}

for _, model in models.items():
    model.learn(tokenized_data)

with open("emotion_profiles", "rb") as file:
    emotion_profiles = pickle.load(file)
            
def meet_operator_1(prob1, beta):
    return (0.9*prob1 + 0.1*beta)

def meet_operator_2(prob1, beta):
    return (0.6*prob1 + 0.4*beta)

def meet_operator_3(prob1, beta):
    return (0.1*prob1 + 0.9*beta)

def meet_operator_4(prob1, beta):
    return (0.3*prob1 + 0.7*beta)

for emotion, model in models.items():
    for bigram in model.bigram_counts.keys():
        prob1 = model.calc_bigram_probability(bigram, LAPLACE, KNESER_NEY)
        beta = emotion_profiles[emotion][bigram]

        max_bigram_emotion_score = beta
        # for bigram_probs in emotion_profiles.values():
        #     max_bigram_emotion_score = max(max_bigram_emotion_score, bigram_probs[bigram])

        if(max_bigram_emotion_score < THRESHOLD_1):
            model.set_bigram_probability(bigram, meet_operator_1(prob1, beta))
        elif(max_bigram_emotion_score < THRESHOLD_2):
            model.set_bigram_probability(bigram, meet_operator_2(prob1, beta))
        elif(max_bigram_emotion_score > THRESHOLD_3):
            model.set_bigram_probability(bigram, meet_operator_3(prob1, beta))
        else:
            model.set_bigram_probability(bigram, meet_operator_4(prob1, beta))

        
def sample_next_word(model, previous_token):
    candidate_words = model.bigrams.get(previous_token, [])
    if not candidate_words:
        return None
    candidate_words_list = list(candidate_words)
    next_word_probs = [model.bigram_probs[(previous_token, word)] for word in candidate_words_list]
    return random.choices(candidate_words_list, next_word_probs)[0]

def generate_sentence(model, max_words=MAX_SENTENCE_LEN):
    starting_bigrams = [bigram for bigram
                        in model.bigram_counts if bigram[0] == model.starting_token]
    starting_bigram_probs = [model.bigram_probs[bigram] for bigram
                             in model.bigram_counts if bigram[0] == model.starting_token]
    starting_word = random.choices(starting_bigrams, starting_bigram_probs)[0][1]
    sentence = [starting_word]
    for _ in range(max_words - 1):
        next_word = sample_next_word(model, sentence[-1])
        if next_word is None:
            if len(sentence) < MIN_SENTENCE_LEN:
                next_word = random.choices(list(model.bigram_probs.keys()),
                                           list(model.bigram_probs.values()))[0][0]
            else: break
        sentence.append(next_word)
    return ' '.join(sentence)

if __name__ == "__main__":
    generated_sentences = dict()
    for emotion, model in models.items():
        generated_sentences[emotion] = [generate_sentence(model) for _ in range(50)]
    with open("generated_sentences.lanat", "wb") as file:
        pickle.dump(generated_sentences, file)
