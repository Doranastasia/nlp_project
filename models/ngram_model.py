from collections import defaultdict, Counter
from typing import List

class NGramLanguageModel:
    def __init__(self, corpus: List[List[str]], n: int):
        self.n = n
        self.count_ngrams = defaultdict(int)  
        self.count_context = defaultdict(int)  

        for c in corpus:
            sentence_len = len(c)
            for i in range(sentence_len - n + 1):  
                ngram = tuple(c[i:i + n]) 
                self.count_ngrams[ngram] += 1
                context = ngram[:-1]  
                self.count_context[context] += 1

    def get_next_words_and_probs(self, prefix: List[str]) -> (List[str], List[float]):
        prefix_tuple = tuple(prefix)
        total_context_count = self.count_context.get(prefix_tuple, 0)

        if total_context_count == 0:
            return [], [] 

        continuation_candidates = {
            ngram[-1]: self.count_ngrams[ngram]
            for ngram in self.count_ngrams
            if len(ngram) == len(prefix_tuple) + 1 and ngram[:-1] == prefix_tuple
        }

        next_words = list(continuation_candidates.keys())
        probabilities = [count / total_context_count for count in continuation_candidates.values()]

        return next_words, probabilities
