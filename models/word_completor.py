from typing import List, Tuple
from collections import defaultdict
from prefix_tree import PrefixTree

class WordCompletor:
    def __init__(self, corpus):
        self.word_counts = defaultdict(int)
        self.word_counter = 0
        
        for text in corpus:
            for word in text:
                self.word_counts[word] += 1
                self.word_counter += 1
        
        vocabulary_words = list(self.word_counts.keys())
        
        self.prefix_tree = PrefixTree(vocabulary_words)

    def get_words_and_probs(self, prefix: str) -> (List[str], List[float]):
        words, probs = [], []
        words = self.prefix_tree.search_prefix(prefix)
        probs = [self.word_counts[word] / self.word_counter for word in words]
        return words, probs

