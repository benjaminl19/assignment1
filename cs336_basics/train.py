import os
import json
import time
import argparse
from typing import List

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from tests import adapters

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

def main():
    bpe_experiments()

if __name__ == "__main__":
    main()
    

