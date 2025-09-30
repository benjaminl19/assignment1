import os
import regex as re
from typing import BinaryIO
from collections import Counter
from multiprocessing import get_context


PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            # Ensure finding all special tokens straddling mini-chunks
            mini_chunk = file.read(mini_chunk_size + len(split_special_token) - 1) 

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def count_chunk(args_tuple) -> dict[tuple[bytes, ...], int]:
    """ 
    Create a dict counting the number of appearances of each sequence
    of bytes (denoting a word) in the specified chunk. 
    """

    filename, start, end, special_tokens = args_tuple
    
    with open(filename, "rb") as tok_file:

        counts: dict[tuple[bytes, ...], int] = Counter()
        tok_file.seek(start)
        chunk = tok_file.read(end - start).decode("utf-8", errors="ignore")

        # Split on special tokens using regex
        escaped_tokens = [
            re.escape(tok) for tok in special_tokens
        ]
        split_regex = re.compile(f"(?:{'|'.join(escaped_tokens)})")
        split_chunks = split_regex.split(chunk)

        # Run pre-tokenization on your split chunks and store the counts for each pre-token
        for subchunk in split_chunks:

            if not subchunk: # in case the subchunk is empty
                continue

            for word in PAT.finditer(subchunk):
                text = word.group(0)
                enc_word = tuple(bytes([b]) for b in text.encode("utf-8"))

                counts[enc_word] += 1

    return counts


def pretokenizer(
        filename: str,
        num_processes: int,
        special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
    """
    Splits the file, runs count_chunk() on chunks in parallel, and updates
    a global dict containing the frequencies of each sequence of bytes
    for the entire file.
    """

    assert len(special_tokens) != 0, "Special_tokens is empty"

    with open(filename, "rb") as tok_file:

        # splits the file into the desired number of chunks
        ex_special_token = special_tokens[0].encode("utf-8", errors="ignore")
        boundaries = find_chunk_boundaries(tok_file, num_processes, ex_special_token)
        chunk_bounds = zip(boundaries[:-1], boundaries[1:])

        total: dict[tuple[bytes, ...], int] = Counter()

        # defines the args_tuple
        tasks = [(filename, start, end, special_tokens) for (start, end) in chunk_bounds]

        context = get_context("spawn")

        # executes calls to count_chunk() in parallel
        with context.Pool(processes=num_processes) as pool:

            for chunk_counts in pool.imap_unordered(count_chunk, tasks, chunksize=1):
                total.update(chunk_counts)
                
            return dict(total)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str]   
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.
    """

    # init and build vocab
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_index = 256
    for tok in special_tokens:
        vocab[next_index] = tok.encode("utf-8")
        next_index += 1

    num_merges = vocab_size - len(vocab)
    if num_merges < 0:
        raise ValueError("vocab size is too small")

    # init merges
    merges: list[tuple[bytes, bytes]] = []

    # run pretokenizer to get word counts
    num_processes: int = 8
    counts = pretokenizer(str(input_path), num_processes, special_tokens)

    # init and build pair_counts
    pair_counts: dict[tuple[bytes, bytes], int] = Counter()
    for word, freq in counts.items():
        for byte_pair in zip(word[:-1], word[1:]):
            pair_counts[byte_pair] += freq

    for _ in range(num_merges):

        # if there are no byte pair
        if not pair_counts:
            break

        # find max byte pair w/ lexiographic tiebreaking
        max_byte_pair =  max(
            pair_counts.items(), key=lambda item: (item[1], item[0])
        )[0]

        # add max byte pair to vocab, merges
        new_token = b"".join(max_byte_pair)
        vocab.update({next_index: new_token})
        next_index += 1
        merges.append(max_byte_pair)

        # iterate through every word to update pair_counts and populate new_counts 
        new_counts: dict[tuple[bytes, ...], int] = Counter()
        for word, freq in counts.items():

            word_length = len(word)

            i = 0
            new_word: list[bytes] = []

            while i < word_length:

                if i < word_length - 1 and (word[i], word[i+1]) == max_byte_pair:

                    # update pair_counts
                    pair_counts[(max_byte_pair)] -= freq

                    if i > 0:
                        pair_counts[(word[i - 1], max_byte_pair[0])] -= freq
                    if i + 2 < word_length:
                        pair_counts[(max_byte_pair[1], word[i + 2])] -= freq

                    if len(new_word) > 0:
                        pair_counts[new_word[-1], new_token] += freq

                    if i + 2 < word_length: # to avoid double count when next pair is also merged
                        if not (i + 3 < word_length and (word[i + 2], word[i + 3]) == max_byte_pair):
                            pair_counts[new_token, word[i+2]] += freq

                    # add byte pair to new_word
                    new_word.append(new_token)
                    i += 2 # skip forward 2 bytes

                else:
                    # add the next byte to new_word
                    new_word.append(word[i])
                    i += 1

            # upate new_counts with new_word
            new_counts[tuple(new_word)] += freq

        counts = new_counts

        # delete pairs where freq = 0
        for pair, freq in list(pair_counts.items()):
            if freq <= 0:
                del pair_counts[pair]
    
    return (vocab, merges)
