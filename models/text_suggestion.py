from typing import List, Union

class TextSuggestion:
    def __init__(self, word_completor, n_gram_model):
        self.word_completor = word_completor
        self.n_gram_model = n_gram_model

    def suggest_text(self, text: Union[str, list], n_words=3, n_texts=1) -> list[list[str]]:
        suggestions = []

        words = text.strip().split() if isinstance(text, str) else text[:]
        
        if not words:
            return []

        last_word = words[-1]
        complets, probabilities = self.word_completor.get_words_and_probs(last_word)
        
        completed_word = complets[probabilities.index(max(probabilities))] if complets else last_word
 
        updated_words = words[:-1] + [completed_word]
        generated_text = [completed_word]
        
        n = self.n_gram_model.n
        context = updated_words[-(n - 1):] if n > 1 else []

        for _ in range(n_words):
            next_candidates, next_probabilities = self.n_gram_model.get_next_words_and_probs(context)
            if not next_candidates:
                break
            
            best_word = next_candidates[next_probabilities.index(max(next_probabilities))]
            generated_text.append(best_word)
            
            context = context[1:] + [best_word] if n > 1 else [best_word]
        
        suggestions.append(generated_text)
        return suggestions

