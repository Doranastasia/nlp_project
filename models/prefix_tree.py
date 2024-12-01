from typing import List, Tuple
from collections import defaultdict

class PrefixTreeNode:
    def __init__(self):
        self.children: dict[str, PrefixTreeNode] = {}
        self.is_end_of_word = False

class PrefixTree:
    def __init__(self, vocabulary: List[str]):
        self.root = PrefixTreeNode()  
        for word in vocabulary:
            self.add_word(word) 

    def add_word(self, word: str):
        current = self.root 
        for w in word:
            if w not in current.children:
                current.children[w] = PrefixTreeNode()
            current = current.children[w]
        current.is_end_of_word = True 

    def search_prefix(self, prefix: str) -> List[str]:
        def _find_all_leaves(node, current_word, result):
            if node.is_end_of_word:
                result.append(current_word)

            for c, child in node.children.items():
                _find_all_leaves(child, current_word + c, result)

        start_node = self.root
        for p in prefix:
            if p not in start_node.children:
                return []  
            start_node = start_node.children[p]
            
        result = []
        _find_all_leaves(start_node, prefix, result)
        return result

