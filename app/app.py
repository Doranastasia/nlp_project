import reflex as rx
import pandas as pd
import numpy as np
import re
import email
import time

from IPython.display import clear_output
from collections import defaultdict, Counter
from typing import Union, List
from word_completor import WordCompletor 
from ngram_model import NGramLanguageModel  
from text_suggestion import TextSuggestion  
from prefix_tree import PrefixTree

def extract_cleaned_bodies(messages):
    cleaned_messages = []
    for msg in messages:
        try:
            email_obj = email.message_from_string(msg)
            raw_msg = email_obj.get_payload()           
            clean_msg = re.split(r'^\s*[-]+ Forwarded by.*$', raw_msg, flags=re.MULTILINE)[-1]
            clean_msg = re.sub(r"(?im)^(from:|to:|sent:|subject:|date:|cc:|bcc:|x-.*:).*", "", clean_msg)
            clean_msg = re.sub(r"(?im)^\s*\d{1,2}:\d{2}\s*(AM|PM)\s*.*\n", " ", clean_msg)
            clean_msg = re.sub(r"(?im)^\d{4,8}\s+(AM|PM)\s*.*$", "", clean_msg, flags=re.MULTILINE)
            clean_msg = re.sub(r'[^A-Za-z0-9 ,.\n]', '', clean_msg)
            clean_msg = re.sub(r'\t', ' ', clean_msg)
            clean_msg = re.sub(r'\n+', ' ', clean_msg)
            clean_msg = re.sub(r' {2,}', ' ', clean_msg).strip()
            
            cleaned_messages.append(clean_msg)
        except Exception as e:
            cleaned_messages.append(f"Error processing message: {e}")
    
    return cleaned_messages

def text_tokenization(text):
    text = text.lower() 
    tokens = re.findall(r'\w+|[^\w\s]', text) 
    return tokens

emails = pd.read_csv('C:/Users/Anastasia/Downloads/NLP/project/data/emails.csv') 
corpus = emails.iloc[:50, :].copy()

corpus = corpus[corpus['message'].apply(len) < 5000]
corpus['message_preprocessed'] = corpus['message'].apply(lambda x: extract_cleaned_bodies([x])[0])
corpus['message_tokenized'] = corpus['message_preprocessed'].apply(text_tokenization) 

word_completor = WordCompletor(corpus['message_tokenized'])
n_gram_model = NGramLanguageModel(corpus=corpus['message_tokenized'], n=2)
text_suggestion = TextSuggestion(word_completor, n_gram_model)

class State(rx.State):
    prompt = ""
    suggested_text = ""
    processing = False
    complete = False

    def set_prompt(self, prompt):
        self.prompt = prompt

    def get_suggestion(self):
        if not self.prompt.strip():
            return rx.window_alert("Please, enter the prompt.")
        self.processing, self.complete = True, False
        yield
        response = text_suggestion.suggest_text(list(self.prompt.split()), n_words=3, n_texts=1)
        self.suggested_text = ' '.join(response[0])
        self.processing, self.complete = False, True

def index():
    return rx.center(
        rx.vstack(
            rx.heading("Text suggestion:", size="8", color="#ffffff"),
            rx.heading("Made by Dorofeeva Anastasia, HSE ADD", size="5", color="#dcd7f2"),
            rx.input(
                placeholder="Enter a prompt... ",
                on_blur=State.set_prompt,
                width="25em",
                border_color="#b57edc",
                color="#ffffff",
                background_color="#6a4c9c"
            ),
            rx.button(
                "suggest continuation", 
                on_click=State.get_suggestion,
                width="25em",
                loading=State.processing, 
                background_color="#9b4d96",
                color="#ffffff",
                border_color="#7a3f7b"
            ),
            rx.cond(
                State.complete,
                rx.text(State.suggested_text, text_align="center", font_weight="bold", color="#ffffff")
            ),
            align="center",
        ),
        width="100%",
        height="100vh",
        background="linear-gradient(to right, #6a4c9c, #9b4d96)"
    )


app = rx.App()
app.add_page(index, title="Text suggestion online!")