from transformers import TrOCRProcessor, VisionEncoderDecoderModel, VisionEncoderDecoderConfig
from PIL import Image

import torch
import pandas as pd
from transformers import get_constant_schedule_with_warmup
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import evaluate
import numpy as np
import yaml

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from unidecode import unidecode
from transformers import PreTrainedTokenizerBase, BatchEncoding
from typing import List, Union, Dict, Optional, Tuple
import logging
import gc

from transformers import (
    RobertaTokenizer,
    TrOCRConfig,
    TrOCRForCausalLM,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    ViTConfig,
    ViTImageProcessor,
    ViTModel,
)


def read_vocab(vocab="vocab.txt"):
    with open(vocab, "rb") as f:
        vocab_str = f.read()
    output = "".join(vocab_str.decode().split())
    return ' ' + output


class CharacterTokenizer(PreTrainedTokenizer):
    def __init__(self, characters: Sequence[str], model_max_length: int, **kwargs):
        """Character tokenizer for Hugging Face transformers.

        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=6. Following are list of all of the special tokens with
                their corresponding ids:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                an id (starting at 7) will be assigned to each character.

            model_max_length (int): Model maximum sequence length.
        """
        self.characters = characters
        self.model_max_length = model_max_length
        bos_token = AddedToken("[BOS]", lstrip=False, rstrip=False)
        eos_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        sep_token = AddedToken("[SEP]", lstrip=False, rstrip=False)
        cls_token = AddedToken("[CLS]", lstrip=False, rstrip=False)
        pad_token = AddedToken("[PAD]", lstrip=False, rstrip=False)
        unk_token = AddedToken("[UNK]", lstrip=False, rstrip=False)
        mask_token = AddedToken("[MASK]", lstrip=True, rstrip=False)

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=False,
            model_max_length=model_max_length,
            **kwargs,
        )

    def get_vocab(self):
        return self._vocab_str_to_int

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        if index > len(self._vocab_int_to_str):
          return self._vocab_int_to_str[6]
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        result = cls + token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = [1] + ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def create_token_type_ids_from_sequences(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        result = len(cls + token_ids_0 + sep) * [0]
        if token_ids_1 is not None:
            result += len(token_ids_1 + sep) * [1]
        return result

    def get_config(self) -> Dict:
        return {
            "char_ords": [ord(ch) for ch in self.characters],
            "model_max_length": self.model_max_length,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "CharacterTokenizer":
        cfg = {}
        cfg["characters"] = [chr(i) for i in config["char_ords"]]
        cfg["model_max_length"] = config["model_max_length"]
        return cls(**cfg)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        cfg = self.get_config()
        with open(cfg_file, "w") as f:
            json.dump(cfg, f, indent=4)

    @classmethod
    def from_pretrained(cls, save_directory: Union[str, os.PathLike], **kwargs):
        cfg_file = Path(save_directory) / "tokenizer_config.json"
        with open(cfg_file) as f:
            cfg = json.load(f)
        return cls.from_config(cfg)


class SynthDataset(Dataset):
    def __init__(self, root_dir, label_file, processor, max_target_length=128):
        self.root_dir = root_dir
        if label_file.endswith('json'):
            with open(label_file, 'r') as f:
                data = json.load(f)
        else:
            with open(label_file) as f:
                data = np.load('path/to/text_labels_correct_headers.npy', allow_pickle=True).item()

        # Convert the dictionary into a list of dictionaries suitable for DataFrame
        rows = []
        for file_name, file_data in data.items():
            row = {'file_name': file_name, 'text': file_data['text']}
            rows.append(row)

        # Create a DataFrame from the list of dictionaries
        self.df = pd.DataFrame(rows)

        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        # some file names end with jp instead of jpg, the two lines below fix this
        if file_name.endswith('jp'):
            file_name = file_name + 'g'
        if not file_name.endswith('.jpg'):
            file_name = file_name + '.jpg'
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = np.array([label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels])

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding


if __name__ == '__main__':
    root_dir = "/home/tarch/datasets/synthetic/lines/english_samples/"
    json_file = "/home/tarch/datasets/synthetic/lines/english_samples/output.json"
    encoder_config = ViTConfig(image_size=384, qkv_bias=False)
    image_processor = ViTImageProcessor(size=encoder_config.image_size)
    vocab = read_vocab()
    tokenizer = CharacterTokenizer(vocab, model_max_length=127)
    processor = TrOCRProcessor(image_processor, tokenizer)

    synth_dataset = SynthDataset(root_dir, json_file, processor)
    dataloader = DataLoader(synth_dataset, batch_size=1, shuffle=True)
    values = next(iter(dataloader))
    print(values['labels'])
    label_ids = values['labels']
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    print(processor.batch_decode(label_ids, skip_special_tokens=True))