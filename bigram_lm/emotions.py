from bigram_lm import Bigram_LM
import utils
import random

sadness = Bigram_LM()
joy = Bigram_LM()
surprise = Bigram_LM()
fear = Bigram_LM()
anger = Bigram_LM()
love = Bigram_LM()

with open('lanat\dataset\corpus.txt', 'r') as file:
    data = file.readlines()
tokenized_data = [line.split() for line in data]
models = {"sadness": sadness, "joy": joy, "surprise": surprise, "fear": fear, "anger": anger, "love": love}

for emo, model in models.items():
    model.learn(tokenized_data)

for line in tokenized_data:
    previous_token = None
    for token in line:
        if previous_token is not None:
            string = previous_token + " " + token
            emo_score = utils.emotion_scores(string)
        else:
            emo_score = utils.emotion_scores(token)

        final_emo_scores = {score['label']: score['score'] for score in emo_score}

        for emo, model in models.items():
            model_probab = model.get_bigram_probability((previous_token, token), False, False)
            model_probab = (model_probab + final_emo_scores[emo]) / 2
            model.bigram_prob[(previous_token, token)] = model_probab

        previous_token = token


def sample_next_word(model, previous_token):
    candidate_words = model.bigrams.get(previous_token, [])
    if not candidate_words:
        return None
    candidate_words_list = list(candidate_words)
    next_word_probs = [model.bigram_prob.get((previous_token, word), 0) for word in candidate_words_list]
    return random.choices(candidate_words_list, next_word_probs)[0]

def generate_sentence(model, max_words=10):
    sentence = [random.choice(list(model.vocabulary))]
    
    for _ in range(max_words - 1):
        next_word = sample_next_word(model, sentence[-1])
        if next_word is None:
            break
        sentence.append(next_word)
    
    return ' '.join(sentence)


for emo, model in models.items():
    print(f"\nGenerated Sentence for {emo.capitalize()} Model:")
    print(generate_sentence(model))