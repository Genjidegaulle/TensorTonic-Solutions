import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """

        # Preprocess list of texts
        texts = [word.lower() for text in texts for word in text.split(' ')]
        texts = sorted(list(set(texts)))
        
        # First, add special tokens to vocab
        t = [self.pad_token,
             self.unk_token,
             self.bos_token,
             self.eos_token
            ]
        t.extend(texts)

        self.vocab_size = len(t)
        
        ### Word to ID dict first ###
        for index, word in enumerate(t):
            if word not in self.word_to_id:
                self.word_to_id[word] = index
                self.id_to_word[index] = word
                
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        encoded = []

        # Edge case
        if len(text) == 0:
            return encoded
        
        for t in text.split(" "):
            token = 1
            t = t.lower()
            if t in self.word_to_id:
                token = self.word_to_id[t]

            encoded.append(token)

        return encoded
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        decoded = []

        if len(ids) == 0:
            return ""

        for i in ids:
            word = self.unk_token
            if i in self.id_to_word.keys():
                word = self.id_to_word[i]
                
            decoded.append(word)

        print(decoded)
        word = " ".join(decoded)
        return word
