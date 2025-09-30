import regex as re
from collections.abc import Iterable

PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

class Tokenizer:

    def __init__(
            self, 
            vocab: dict[int, bytes],
            merges: list[tuple[bytes, bytes]], 
            special_tokens=None):
        """ 
        Construct a tokenizer from a given
        vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens or [])
        self._special_set = set(self.special_tokens)

        # invert the vocab map
        self.tok_to_id: dict[bytes, int] = {tok: i for i, tok in self.vocab.items()}

        # add new special tokens to vocab, tok_to_id
        next_id = len(vocab)
        for tok in self.special_tokens:
            enc = tok.encode("utf-8")
            if enc not in self.tok_to_id:
                self.vocab[next_id] = enc
                self.tok_to_id[enc] = next_id
                next_id += 1

        # create a map showing merge order
        self.merge_num: dict[tuple[bytes, bytes], int] = {
            byte_pair: i for i, byte_pair in enumerate(self.merges)
        }

        # compile split regex
        self._split_regex = None
        if self.special_tokens:
            # reverse sort so longer special tokens are matched first
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_tokens = [
                re.escape(tok) for tok in sorted_special_tokens
            ]
            self._split_regex = re.compile(f"({'|'.join(escaped_tokens)})")
            
    @classmethod
    def from_files(
            cls, 
            vocab_filepath: str,
            merges_filepath: str,
            special_tokens: list[str] | None = None
    ) -> "Tokenizer":
        """
        Class method that constructs and return a Tokenizer from a 
        serialized vocabulary and list of merge and (optionally) a list of special tokens.
        """
        import json

        with open(vocab_filepath, "r") as vocab_file:
            vocab_json = json.load(vocab_file)
        vocab = {int(id): bytes.fromhex(tok) for id, tok in vocab_json.items()}

        with open(merges_filepath, "r") as merges_file:
            merges_json = json.load(merges_file)
        merges = [(bytes.fromhex(x), bytes.fromhex(y)) for x, y in merges_json]
        
        return cls(vocab, merges, special_tokens)
        
    def encode(
            self,
            text: str
    ) -> list[int]:
        """
        Encode an input text into a sequence of token IDs
        """
        # split text on special tokens
        if self._split_regex:
            split_chunks = self._split_regex.split(text)
        else: 
            split_chunks = [text]

        return self._apply_merges(split_chunks)
        
    def encode_iterable(
            self,
            iterable: Iterable[str]
    ) -> Iterable[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), 
        return a generator that lazily yields token IDs
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
        
    def decode(
            self,
            ids: list[int]
    ) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        byte_list: list[bytes] = []
        for token_id in ids:
            tok = self.vocab.get(token_id)
            if tok is None:
                continue
            byte_list.append(tok)
    
        return b"".join(byte_list).decode("utf-8", errors='replace')
    

    def _apply_merges(
        self,
        split_chunks: list[str]
    ) -> list[int]:
        """
        Given a list of strings (denoting chunks split on special tokens), 
        pretokenize each chunk, apply BPE merges, and return a list of token ids.
        Special tokens are added as single tokens and are never merged.
        """
        tokenized_text: list[int] = []

        for subchunk in split_chunks:

            if not subchunk: # in case the subchunk is empty
                continue
            
            # if subchunk is just a special token
            if subchunk in self._special_set:
                tokenized_text.append(self.tok_to_id[subchunk.encode("utf-8")])
                continue

            for word_match in PAT.finditer(subchunk):
                word = word_match.group(0)
                enc_word = [bytes([b]) for b in word.encode("utf-8")]

                # apply the merges to the word in order
                while (True):
                    
                    # find the highest priority merge in enc_word
                    earliest_merge = None
                    earliest_merge_pair = None

                    for x, y in zip(enc_word[:-1], enc_word[1:]):
                        num = self.merge_num.get((x,y))
                        if num is None:
                            continue
                        if earliest_merge is None or num < earliest_merge:
                            earliest_merge = num
                            earliest_merge_pair = (x, y)

                    if earliest_merge_pair is None:
                        break # no more merges

                    # execute highest priority merge
                    new_word: list[bytes] = []
                    i = 0
                    word_length = len(enc_word)

                    while i < word_length:

                        if i < word_length - 1 and (enc_word[i], enc_word[i+1]) == earliest_merge_pair:
                            new_word.append(b"".join(earliest_merge_pair))
                            i += 2
                        else: 
                            new_word.append(enc_word[i])
                            i += 1
                            
                    enc_word = new_word

                    if (len(enc_word) == 1):
                        break

                # add tokens to output
                for tok in enc_word:
                    tokenized_text.append(self.tok_to_id[tok])

        return tokenized_text    
