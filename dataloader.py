import glob
import torch
from torch.utils.data import IterableDataset
import pyarrow.parquet as pq
from transformers import AutoTokenizer


class FineWebDataset(IterableDataset):
    def __init__(
            self,
            data_dir: str,
            tokenizer_name: str = "gpt2",
            seq_len: int = 1024
        ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.seq_len = seq_len

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.model_max_length = int(1e30)  # override max-length to prevent seq length warnings
        
        self.files = sorted(glob.glob(f"{data_dir}/**/*.parquet", recursive=True))
        if not self.files:
            raise FileNotFoundError(f"No .parquet shards found in {data_dir}")


    def _get_worker_shards(self):
        """Splits Parquet shards across DDP ranks and DataLoader workers."""
        worker_info = torch.utils.data.get_worker_info()
        
        if torch.distributed.is_initialized():  # used for multi-GPU setup
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1

        # shard splitting across GPUs (ranks)
        rank_files = self.files[rank::world_size]  

        # shard splitting across DataLoader workers per GPU
        if worker_info is None:
            return rank_files  # single process, load data as is
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            num_streams = world_size * num_workers
            assert len(self.files) > num_streams, \
                f"Not enough shards ({len(self.files)}) for the given streams ({num_streams}), reduce num_workers"

            return rank_files[worker_id::num_workers]  # multi-process, split rank_files across workers


    def __iter__(self):
        shards = self._get_worker_shards()
        token_buffer = []

        for shard_path in shards:
            parquet_file = pq.ParquetFile(shard_path)
            
            for rg_idx in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(rg_idx, columns=["text"])
                texts = table["text"].to_pylist()

                for text in texts:
                    if not text.strip():
                        continue

                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    tokens.append(self.tokenizer.eos_token_id)
                    token_buffer.extend(tokens)

                    while len(token_buffer) >= self.seq_len + 1:
                        chunk = token_buffer[:self.seq_len+1]
                        token_buffer = token_buffer[self.seq_len:]

                        x = torch.tensor(chunk[:-1], dtype=torch.long)
                        y = torch.tensor(chunk[1:], dtype=torch.long)

                        yield {"input_ids": x, "labels": y}
