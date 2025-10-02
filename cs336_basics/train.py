import os
import json, time, argparse
import wandb
import numpy as np
import torch
import torch.nn as nn
from typing import List

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from tests import adapters
from cs336_basics import tokenizer, transformer

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
                        if dumped:
                            output.write(",")
                        json.dump(d_tok, output)
                        dumped = True
                    document_tokens = []
                    docs_read += 1 

                # iterate through num_documents 
                if args.num_documents is not None and docs_read >= args.num_documents:
                    break
            
            output.write("]")

def training_loop():

    ap = argparse.ArgumentParser()

    ap.add_argument("--data", required=True)
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--device", required=True)
    ap.add_argument("--special", nargs="*", default=["<|endoftext|>"])

    # model hyperparameters
    ap.add_argument("--vocab_size", type=int, default=10_000)
    ap.add_argument("--context_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--d_ff", type=int, default=1344)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--num_heads", type=int, default=16)
    ap.add_argument("--rope_theta", type=float, default=10_000.0)

    # optimizer hyperparameters
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--betas", type=float, nargs=2, default=(0.9,0.999))
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--max_l2_norm", type=float, default=1.0)

    # schedule hyperparameters
    ap.add_argument("--warmup_iters", type=int, default=1000)
    ap.add_argument("--cosine_cycle_iters", type=int, default=20_000)
    ap.add_argument("--max_learning_rate", type=float, default=1e-4)
    ap.add_argument("--min_learning_rate", type=float, default=1e-5)
    ap.add_argument("--checkpoint_freq", type=int, default=500)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # get token ids for data
    with open(args.data, "r") as file: 
        data_ids = np.array(json.load(file))

    # initialize transformer and optimzer
    transformer_lm = transformer.TransformerLM(
        args.vocab_size, args.d_model, args.num_layers, args.num_heads, args.d_ff
        ).to(device)
    if args.lr is None:
        lr = args.max_learning_rate
    else:
        lr = args.lr
    
    AdamW = adapters.get_adamw_cls()
    optimizer = AdamW(
        transformer_lm.parameters(), lr, args.weight_decay, args.betas, args.eps
        )
    
    wandb.init(project="assignment1-basics", config=vars(args))

    # transformerlm training loop
    transformer_lm.train()
    for iter in range(args.cosine_cycle_iters):
        
        # update and apply lr according to schedule
        lr = adapters.run_get_lr_cosine_schedule(
            iter, 
            args.max_learning_rate,
            args.min_learning_rate,
            args.warmup_iters,
            args.cosine_cycle_iters
        )
        for param_group in optimizer.param_groups:
            param_group["alpha"] = lr

        # forward pass
        inputs, labels = adapters.run_get_batch(data_ids, args.batch_size, args.context_length, args.device)
        logits = transformer_lm(inputs, args.context_length, args.rope_theta)
        cross_entropy_loss = adapters.run_cross_entropy(
            logits.reshape(logits.shape[0] * logits.shape[1], logits.shape[2]), 
            labels.reshape(labels.shape[0] * labels.shape[1])
            )
        
        # backprop w/ grad clipping
        optimizer.zero_grad(set_to_none=True)
        cross_entropy_loss.backward()
        adapters.run_gradient_clipping(transformer_lm.parameters(), args.max_l2_norm)
        optimizer.step()

        # log iteration, learning rate, loss
        wandb.log({"iter": iter, "lr": lr, "loss": cross_entropy_loss.item()})

        # save checkpoint
        if args.checkpoint_freq and iter % args.checkpoint_freq == 0:
            adapters.run_save_checkpoint(
                transformer_lm, optimizer, iter, os.path.join(args.out_dir, f"checkpoint_{iter}.pt")
                )
    
    # save final weights
    adapters.run_save_checkpoint(
                transformer_lm, optimizer, args.cosine_cycle_iters, os.path.join(args.out_dir, "final.pt")
                )

def main():
    training_loop()

if __name__ == "__main__":
    main()
    

