import argparse
from pathlib import Path

import requests
import torch
from PIL import Image
import numpy as np

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
from transformers.utils import logging
from transformers import BertTokenizer, BertModel, PreTrainedTokenizerBase
from typing import List, Union, Dict
import torch
from transformers.models.trocr.custom_tokenizer.tokenizer import CharacterTokenizer
USE_MY_PROCESSOR = False
# https://huggingface.co/helboukkouri/character-bert/tree/main

def read_vocab(vocab="vocab.txt"):
    with open(vocab, "rb") as f:
        vocab_str = f.read()
    return "".join(vocab_str.decode().split())

vocab = read_vocab()
logging.set_verbosity_info()
logger = logging.get_logger(__name__)

encoder_config = ViTConfig(image_size=384, qkv_bias=False)

image_processor = ViTImageProcessor(size=encoder_config.image_size)
tokenizer = CharacterTokenizer(vocab, model_max_length=128)

processor = TrOCRProcessor(image_processor, tokenizer)

path = "g:/s3/synthetic_data/one_line/english/000000000.jpg"
if Path(path).exists():
    random_image = Image.open(path)
else:
    random_image = np.random.random([512,512,3])

random_text = "Hello, I am a random text Ö∞"
batch = {"images": [random_image], "text": [random_text]}
b = processor(**batch)
print(b["labels"])
print(b)
