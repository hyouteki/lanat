class BPE_Tokenizer:
    def __init__(self):
        self.filename = ""
        self.word_freq_pairs = []
        # assuming the vocabulary is english
        self.vocabulary = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '_'}
        self.merge_rules = []
        self.n_merges = 0

    def __gen_word_freq_pairs(self):
        tmp_map = dict()
        with open(self.filename, "r") as file:
            for word in file.read().split():
                if word in tmp_map:
                    tmp_map[word] += 1
                else:
                    tmp_map[word] = 1
        self.word_freq_pairs.extend([[[*word+'_'], freq] for word, freq in tmp_map.items()])

    def __iter_vocabulary(self):
        tmp_map = dict()
        merge_rule_map = dict()
        for word_freq_pair in self.word_freq_pairs:
            word, freq = word_freq_pair
            for i in range(len(word)-1):
                tmp = "".join(word[i:i+2])
                if tmp in tmp_map:
                    tmp_map[tmp] += freq
                else:
                    tmp_map[tmp] = freq
                if tmp not in merge_rule_map:
                    merge_rule_map[tmp] = (word[i], word[i+1])
        max_comb_val = ""
        max_comb_freq = 0
        for val, freq in tmp_map.items():
            if freq > max_comb_freq:
                max_comb_val = val
                max_comb_freq = freq
        self.vocabulary.add(max_comb_val)
        self.merge_rules.append(merge_rule_map[max_comb_val])
        for i in range(len(self.word_freq_pairs)):
            word, freq = self.word_freq_pairs[i]
            if len(word) == 1: continue 
            new_word = []
            j = 0
            while j < len(word):            
                tmp = "".join(word[j:j+2])
                if tmp == max_comb_val:
                    new_word.append(tmp)
                    j += 2
                else:
                    new_word.append(word[j])
                    j += 1
            self.word_freq_pairs[i][0] = new_word
                
    def learn_vocabulary(self, filename, n_merges):
        self.filename = filename
        self.n_merges = n_merges
        self.__gen_word_freq_pairs()
        for _ in range(self.n_merges):
            self.__iter_vocabulary()

    def save_merge_rules(self, filename):
        with open(filename, "w") as file:
            for merge_rule in self.merge_rules:
                file.write(f"{merge_rule[0]},{merge_rule[1]}\n")
        
    def debug(self):
        print(self.word_freq_pairs[:100])
        print(self.vocabulary)

if __name__ == "__main__":
    bpe_tokenizer = BPE_Tokenizer()
    bpe_tokenizer.learn_vocabulary("../dataset/corpus.txt", 500)
    bpe_tokenizer.save_merge_rules("tests/merge_rules.txt")
    bpe_tokenizer.debug()
