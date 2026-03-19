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

    def test_vocab(self):
        self.build_vocab(["GOOD", "MORNING", "morning", "son", "sOn", "hello", "world"])

        assert self.vocab_size == 9, self.vocab_size
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        full_text = []
        for text in texts:
            full_text.extend(text.lower().split(" "))
        lower_txt = list(set(full_text))

        # start by adding special tokens
        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token

        # now, add all text
        for i in range(len(lower_txt)):
            self.id_to_word[i+4] = lower_txt[i]

        # When completed, rotate
        self.word_to_id = {value: key for key, value in self.id_to_word.items()}

        self.vocab_size = len(self.id_to_word.keys())

        assert self.vocab_size == 10, texts
        
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        txt_lst = text.lower().split(" ")

        tokens = []
        for txt in txt_lst:
            if txt in self.word_to_id.keys():
                t = self.word_to_id[txt]
            else:
                t = self.word_to_id[self.unk_token]

            tokens.append(t)

        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """

        texts = []
        for id in ids:
            if id in self.id_to_word:
                texts.append(self.id_to_word[id])
            else:
                texts.append(self.id_to_word[1])

        text = " ".join(texts)
        assert text == "hello world", texts

        return text