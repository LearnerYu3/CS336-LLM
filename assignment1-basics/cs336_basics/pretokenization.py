import os
from typing import BinaryIO
from collections import defaultdict
import regex as re
import tqdm
import multiprocessing
import gc
import traceback


class TokenizerTrainer:
    def __init__(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], 
                 processor_num: int):
        """
        Args:
            input_path: Path to the input file.
            vocab_size: The target vocabulary size.
            special_tokens: List of special tokens to use.
            processor_num: Number of processors to use for training.
        """
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.processor_num = processor_num

        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for i in range(len(special_tokens)):
            self.vocab[256+i] = special_tokens[i].encode("utf-8")
        self.merges = []
        self.chunk_multiplier = 10

        self.pretokens = dict()
        self.pretokens_index = dict()
        self.bp_freq = dict()

    def find_chunk_boundaries(self,
        file: BinaryIO, 
        desired_num_chunks: int, 
        split_special_token: bytes
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), (
            "Must represent special token as a bytestring"
        )

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
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

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

    def train_bpe(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        with open(self.input_path, "rb") as f:
            delimiters = "|".join(self.special_tokens)
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            boundaries = self.find_chunk_boundaries(f, self.processor_num, b"<|endoftext|>")

            idx = 0
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")

                # Run pre-tokenization on your chunk and store the counts for each pre-token
                sequences = re.split(delimiters, chunk)

                for seq in sequences:
                    words = re.finditer(PAT, seq)
                    for word in words:
                        pretoken = tuple(word.group().encode("utf-8"))
                        if pretoken not in self.pretokens_index:
                            self.pretokens_index[pretoken] = idx
                            self.pretokens[idx] = [pretoken, 1]
                            idx += 1
                        else:
                            self.pretokens[self.pretokens_index[pretoken]][1] += 1

            self.init_bp_freq()

            for _ in tqdm.tqdm(range(self.vocab_size - len(self.vocab)), desc="Training BPE"):
                self.merge_bpe()

        return self.vocab, self.merges

    def train_bpe_pool(self):
        num_chunks = self.processor_num * self.chunk_multiplier
        with open(self.input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

        pool = multiprocessing.Pool(self.processor_num)

        tasks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            tasks.append((self.input_path, start, end, self.special_tokens))

        print(f"Starting {len(tasks)} tasks with {self.processor_num} processors")

        pbar = tqdm.tqdm(total=len(tasks), desc="Pretokenizing & Merging")
        gc.disable()

        self.token_freqs = dict()
        for worker_result in pool.imap_unordered(TokenizerTrainer.pretokenize_file_offset, tasks):
            if worker_result:
                self._incremental_merge(worker_result)
            pbar.update(1)

        pbar.close()
        pool.close()
        pool.join()

        self.pretokens = {i: [token, count] for i, (token, count) in enumerate(self.token_freqs.items())}

        del self.token_freqs
        gc.enable()
        gc.collect()

        print("Initializing BP frequencies...")
        self.init_bp_freq()

        for i in tqdm.tqdm(range(self.vocab_size - len(self.vocab)), desc="Merge byte pairs"):
            self.merge_bpe()

        return self.vocab, self.merges


    def _incremental_merge(self, pretokens: dict[tuple, int]):
        for token, count in pretokens.items():
            if token in self.token_freqs:
                self.token_freqs[token] += count
            else:
                self.token_freqs[token] = count

    @staticmethod
    def pretokenize_file_offset(args):
        file_path, start, end, special_tokens_list = args
        try:
            with open(file_path, "rb") as f:
                f.seek(start)
                chunk_str = f.read(end - start).decode("utf-8", errors="ignore")
            pretokens = {}
            delimiters = "|".join(special_tokens_list)
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            splited_chunk = re.split(delimiters, chunk_str)
            for seg in splited_chunk:
                tokens = re.finditer(PAT, seg)
                for token in tokens:
                    pretoken = tuple(token.group().encode("utf-8"))
                    if pretoken not in pretokens:
                        pretokens[pretoken] = 0
                    pretokens[pretoken] += 1
            return pretokens
        except Exception as e:
            print(f"Error in process {os.getpid()}: {e}")
            print(traceback.format_exc())
            return None


    def init_bp_freq(self):
        for pretoken_idx, pretoken_data in tqdm.tqdm(self.pretokens.items(), desc="Initializing BP frequency"):
            pretoken, pretoken_cnt = pretoken_data
            bytelist = list(zip(pretoken[:-1], pretoken[1:]))
            for bp in bytelist:
                if bp not in self.bp_freq:
                    self.bp_freq[bp] = [0, []]
                self.bp_freq[bp][0] += pretoken_cnt

            for bp in bytelist:
                self.bp_freq[bp][1].append(pretoken_idx)

    def merge_bpe(self):
        pair = max(self.bp_freq, key=lambda x: (self.bp_freq[x][0], self.vocab[x[0]], self.vocab[x[1]]))
        index1, index2 = pair
        new_index = len(self.vocab)
        self.vocab[new_index] = self.vocab[index1] + self.vocab[index2]
        self.merges.append((self.vocab[index1], self.vocab[index2]))

        self.merge(pair, new_index)

    def merge(self, pair, new_index):
        pair_pretokens = self.bp_freq[pair][1]
        for pretoken_idx in pair_pretokens:
            bytestr = self.pretokens[pretoken_idx][0]
            new_pretoken = bytestr
            i = 0
            while i < len(new_pretoken) - 1:
                slide_window = new_pretoken[i:i+2]
                if pair == slide_window:
                    pre_count = self.pretokens[pretoken_idx][1]
                    if i > 0:
                        new_pair = (new_pretoken[i-1], new_index)
                        if new_pair not in self.bp_freq:
                            self.bp_freq[new_pair] = [pre_count, [pretoken_idx]]
                        else:
                            self.bp_freq[new_pair][0] += pre_count
                            if not self.bp_freq[new_pair][1] or self.bp_freq[new_pair][1][-1] != pretoken_idx:
                                self.bp_freq[new_pair][1].append(pretoken_idx)

                        overlap_pair = (new_pretoken[i-1], pair[0])
                        self.bp_freq[overlap_pair][0] -= pre_count
                        if self.bp_freq[overlap_pair][0] <= 0:
                            self.bp_freq.pop(overlap_pair)

                    if i + 2 < len(new_pretoken):
                        new_pair = (new_index, new_pretoken[i+2])
                        if new_pair not in self.bp_freq:
                            self.bp_freq[new_pair] = [pre_count, [pretoken_idx]]
                        else:
                            self.bp_freq[new_pair][0] += pre_count
                            if not self.bp_freq[new_pair][1] or self.bp_freq[new_pair][1][-1] != pretoken_idx:
                                self.bp_freq[new_pair][1].append(pretoken_idx)

                        overlap_pair = (pair[1], new_pretoken[i+2])
                        self.bp_freq[overlap_pair][0] -= pre_count
                        if self.bp_freq[overlap_pair][0] <= 0:
                            self.bp_freq.pop(overlap_pair)

                    self.bp_freq[pair][0] -= pre_count
                    if self.bp_freq[pair][0] <= 0:
                        self.bp_freq.pop(pair)

                    new_pretoken = new_pretoken[:i] + (new_index, ) + new_pretoken[i+2:]

                i += 1

            self.pretokens[pretoken_idx][0] = new_pretoken

if __name__ == "__main__":
    tokenizer = TokenizerTrainer(
        "../data/TinyStoriesV2-GPT4-train.txt",
        10000,
        ["<|endoftext|>"],
        4
    )

    # vocab, merges = tokenizer.train_bpe()
    vocab, merges = tokenizer.train_bpe_pool()

    max_len = 0
    max_idx = -1
    for idx, token in vocab.items():
        if len(token) > max_len:
            max_len = len(token)
            max_idx = idx
    print(max_idx, max_len, vocab[max_idx])

