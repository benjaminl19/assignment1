import os
import json
import time
import argparse
from typing import List

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from tests import adapters
from cs336_basics import tokenizer

def save_vocab_and_merges(
        vocab: dict[int, bytes],
        merges: List[tuple[bytes, bytes]],
        output_dir: str
): 
    """
    Takes in the vocab, merges produced by adapters.run_train_bpe()
    and dumps them into .json files in a specified output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    # convert bytes entries to hex for .json
    vocab_hex = {int(freq): token.hex() for freq, token in vocab.items()}
    merges_hex = [[x.hex(), y.hex()] for (x, y) in merges]

    # write vocab and merges to .json in output_dir
    with open(os.path.join(output_dir, "vocab.json"), "w") as vocab_file:
        json.dump(vocab_hex, vocab_file)
    with open(os.path.join(output_dir, "merges.json"), "w") as merges_file:
        json.dump(merges_hex, merges_file)

def bpe_experiments():

    # parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab_size", type=int, default=10_000)
    ap.add_argument("--out_dir", default="tokens")
    ap.add_argument("--special", nargs="*", default=["<|endoftext|>"])
    args = ap.parse_args()

    # measure total run_train_bpe() runtime
    start_time = time.perf_counter()
    vocab, merges = adapters.run_train_bpe(args.data, args.vocab_size, args.special)
    total_time = time.perf_counter() - start_time

    # write vocab, merges to output_dir
    save_vocab_and_merges(vocab, merges, args.out_dir)

    longest_id, longest_tok = max(vocab.items(), key=lambda kv: len(kv[1]))
    print(f"Training time: {total_time:.2f}s")
    print(f"Vocab size: {len(vocab)} / {args.vocab_size}")
    print(f"Num merges: {len(merges)}")
    print(f"Longest token id/length: {longest_id}, {len(longest_tok)} bytes")
    print(f"Longest token string: {longest_tok.decode('utf-8')}")

def tokenizer_experiments():

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab_file", required=True)
    ap.add_argument("--merge_file", required=True)
    ap.add_argument("--num_documents", type=int, default=None)
    ap.add_argument("--out_dir", default="tokenized")
    ap.add_argument("--special", nargs="*", default=["<|endoftext|>"])
    args = ap.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer_ex = tokenizer.Tokenizer.from_files(args.vocab_file, args.merge_file, args.special)
    eot_id = tokenizer_ex.encode("<|endoftext|>")[0]

    with open(args.data, "r") as data_file:
        with open(os.path.join(args.out_dir, "tokenized.json"), "w") as output:

            output.write("[")
            dumped = False

            token_ids = tokenizer_ex.encode_iterable(data_file)
            docs_read = 0
            document_tokens = []

            while True:

                try:
                    tok = next(token_ids)
                except StopIteration:
                    # just break if EOF with no eot
                    break

                document_tokens.append(tok)

                # increment docs and dump to .json if end of text
                if tok == eot_id:
                    for d_tok in document_tokens:
                        json.dump(d_tok, output)
                        if dumped:
                            output.write(",")
                        dumped = True
                    document_tokens = []
                    docs_read += 1 

                # iterate through num_documents 
                if args.num_documents is not None and docs_read >= args.num_documents:
                    break
            
            output.write("]")

def main():
    tokenizer_experiments()

if __name__ == "__main__":
    main()
    

