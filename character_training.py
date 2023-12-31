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
import argparse

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
import sys

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

USE_MY_PROCESSOR = False


def read_vocab(vocab="vocab.txt"):
    with open(vocab, "rb") as f:
        vocab_str = f.read()
    output = "".join(vocab_str.decode().split())
    return ' ' + output


def get_model_and_proc(configurations):
    vocab = read_vocab()
    encoder_config = ViTConfig(image_size=384, qkv_bias=False)
    image_processor = ViTImageProcessor(size=encoder_config.image_size)
    tokenizer = CharacterTokenizer(vocab, model_max_length=128)
    processor = TrOCRProcessor(image_processor, tokenizer)
    processor.image_processor.size['height'] = configurations['image_height']
    processor.image_processor.size['width'] = configurations['image_width']

    config = VisionEncoderDecoderConfig.from_pretrained(configurations['pretrained_model'])
    config.use_learned_position_embeddings = configurations['use_learned_embeddings']
    config.decoder.use_learned_position_embeddings = configurations['use_learned_embeddings']
    config.encoder.image_size = (configurations['image_height'], configurations['image_width'])
    config.max_length = 75

    model = VisionEncoderDecoderModel.from_pretrained(configurations['pretrained_model'], config=config, ignore_mismatched_sizes=True)
    if configurations['load_model']:
        state = torch.load(configurations['load_model_path'])
        model.load_state_dict(state['model_state_dict'])
    return model, processor


def trocr_image(model, processor, image):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(generated_text)


def new_iam_df(label_directory):
    rows = []
    with open(label_directory, 'r') as label_file:
        for line in label_file:
            data = line.split('|')
            file_name = f'{data[0]}.jpg'
            label = ' '.join(data[1:]).replace('\n', '')
            new_row = {'file_name': file_name, 'text': label}
            rows.append(new_row)
    df = pd.DataFrame(rows, columns=['file_name', 'text'])
    return df


def set_model_tokens(model, processor, token_id):
    model.config.decoder_start_token_id = token_id  # processor.tokentizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id


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


class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
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


class SynthDataset(Dataset):
    def __init__(self, root_dir, label_file, processor, max_target_length=128):
        self.root_dir = root_dir
        rows = []
        if label_file.endswith('json'):
            with open(label_file, 'r') as f:
                data = json.load(f)
                for file_name, file_data in data.items():
                    row = {'file_name': file_name, 'text': file_data['text']}
                    rows.append(row)
        else:
            path = Path(label_file)
            data = np.load(path, allow_pickle=True).item()
            rows = []
            for file_name, file_data in data.items():
                row = {'file_name': file_data['ImageName'], 'text': file_data['Paragraph_orig']}
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


class TrainingLoop:

    def __init__(self, model, processor, optimizer, device, lr, scheduler, num_epochs, synth_epochs, num_synth_images):

        self.model = model
        self.processor = processor
        self.optimizer = optimizer
        self.lr = lr
        self.device = device
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.synth_epochs = synth_epochs
        self.num_synth_images = num_synth_images

        self.losses_arr = []
        self.cer_validation_arr = []
        self.validation_arr = []

        self.cer_metric = evaluate.load('cer')

    def compute_cer(self, pred_ids, label_ids):
        pred_str = self.processor.batch_decode(
            pred_ids, skip_special_tokens=True)
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        label_str = self.processor.batch_decode(
            label_ids, skip_special_tokens=True)
        cer = self.cer_metric.compute(
            predictions=pred_str, references=label_str)
        return cer

    def print_cer(self, dataloader, eval_or_train):
        self.model.eval()
        cers = []
        with torch.no_grad():
            for index, batch in enumerate(dataloader):
                # run batch generation
                outputs = self.model.generate(
                    batch["pixel_values"].to(self.device))
                # compute metrics
                cer = self.compute_cer(pred_ids=outputs,
                                       label_ids=batch["labels"])
                cers.append(cer)
                if index > 10000:
                    break

        valid_norm = np.sum(cers) / len(cers)
        print(f"{eval_or_train}: ", valid_norm)
        self.model.train()

    def print_eval_cer(self, eval_dataloader):
        self.print_cer(eval_dataloader, "VALIDATION CER")

    def print_train_cer(self, train_dataloader):
        self.print_cer(train_dataloader, "TRAIN CER")

    def print_synth_cer(self, synth_dataloader):
        self.print_cer(synth_dataloader, 'SYNTH CER')

    def train(self, synth_dataloader, train_dataloader, eval_dataloader):
        # self.print_eval_cer(eval_dataloader)
        # self.print_train_cer(train_dataloader)

        for synth_epoch in range(self.synth_epochs):
            print(f'Synth Epoch is {synth_epoch}')
            synth_loss = 0.0
            self.model.train()
            epoch_index = 1
            for index, batch in enumerate(synth_dataloader):
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                print(batch['pixel_values'])
                outputs = self.model(**batch)
                loss = outputs.loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                synth_loss += loss.detach().item()
                self.scheduler.step()
                gc.collect()
                if index % 5 == 0:
                    print(index)
                epoch_index = index
                if index > self.num_synth_images:
                    break
            torch.save(
                {
                    'epoch': epoch_index,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': synth_loss
                }, "/home/sdslinn/home/models/recent/test_model"
            )
            print(f"Synth Epoch {synth_epoch} Loss: ", synth_loss / epoch_index)

            # Evaluate on eval set every 5 epochs

            if synth_epoch % 5 == 0:
                self.model.eval()
                valid_cer = []
                with torch.no_grad():
                    for batch in eval_dataloader:
                        # run batch generation
                        outputs = self.model.generate(
                            batch["pixel_values"].to(self.device))
                        # compute metrics
                        cer = self.compute_cer(pred_ids=outputs, label_ids=batch["labels"])
                        valid_cer.append(cer)

                valid_norm = np.sum(valid_cer) / len(valid_cer)
                print(f"Synth Validation CER: {valid_norm}, synth_epoch: {synth_epoch}")
                self.cer_validation_arr.append(valid_norm)
                self.print_synth_cer(synth_dataloader)

        for epoch in range(self.num_epochs):
            print(f'Epoch is {epoch}')
            train_loss = 0.0
            self.model.train()
            i = 0
            for batch in train_dataloader:
                i = i + 1
                print(i)
                for k, v in batch.items():
                    batch[k] = v.to(self.device)

                outputs = self.model(**batch)

                loss = outputs.loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.detach().item()

                self.scheduler.step()
                gc.collect()
            print(f"Epoch {epoch} Loss: ", train_loss / len(train_dataloader))

            # Evaluate on eval set every 5 epochs
            if epoch % 5 == 0:
                self.model.eval()

                valid_cer = []
                with torch.no_grad():
                    for batch in eval_dataloader:
                        # run batch generation
                        outputs = self.model.generate(
                            batch["pixel_values"].to(self.device))
                        # compute metrics
                        cer = self.compute_cer(pred_ids=outputs, label_ids=batch["labels"])
                        valid_cer.append(cer)

                valid_norm = np.sum(valid_cer) / len(valid_cer)
                print("Validation CER: ", valid_norm)
                self.cer_validation_arr.append(valid_norm)

                self.print_train_cer(train_dataloader)
            torch.save(
                {
                    'epoch': epoch_index,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': synth_loss
                }, "/home/sdslinn/home/models/recent/test_model"
            )


class Main:

    def __init__(self, configuration):
        self.__dict__.update(locals())
        self.model, self.processor = get_model_and_proc(self.configuration)
        image_directory = self.configuration['image_directory']
        synth_directory = self.configuration['synth_directory']
        synth_labels = self.configuration['synth_labels']

        train_df = new_iam_df(self.configuration['train_label_directory'])
        print(train_df.head())
        test_df = new_iam_df(self.configuration['test_label_directory'])

        train_dataset = IAMDataset(
            root_dir=image_directory, df=train_df, processor=self.processor
        )
        self.train_dataset = train_dataset

        eval_dataset = IAMDataset(
            root_dir=image_directory, df=test_df, processor=self.processor
        )
        self.eval_dataset = eval_dataset

        synth_dataset = SynthDataset(
            root_dir=synth_directory, label_file=synth_labels, processor=self.processor,
        )
        self.synth_dataset = synth_dataset

        self.batch_size = self.configuration['batch_size']
        self.num_workers = self.configuration['num_workers']

    def train(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Running on device', device)
        self.model.to(device)

        set_model_tokens(self.model, self.processor, self.configuration['token_id'])

        optimizer = optim.AdamW(self.model.parameters(), lr=self.configuration['learning_rate'])

        # scheduler = StepLR(optimizer, step_size=1, gamma=1.03)
        scheduler = get_constant_schedule_with_warmup(optimizer, self.configuration['warmup_steps'])
        synth_dataloader = DataLoader(self.synth_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        eval_dataloader = DataLoader(self.eval_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        training_loop = TrainingLoop(
            model=self.model, processor=self.processor, optimizer=optimizer, device=device, lr=self.configuration['learning_rate'],
            scheduler=scheduler, num_epochs=self.configuration['num_epochs'], synth_epochs=self.configuration['synth_epochs'], num_synth_images=self.configuration['num_synth_images']
        )
        print('Starting training \n')
        training_loop.train(synth_dataloader, train_dataloader, eval_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Synth training set.")
    parser.add_argument("--yml_file", '-f', help="YML file with settings")

    args = parser.parse_args()

    with open(f'/home/sdslinn/home/{args.yml_file}') as config_file:
        configs = yaml.load(config_file, Loader=yaml.FullLoader)

    configs['learning_rate'] = float(configs['learning_rate'])
    configs['image_height'] = int(configs['image_height'])
    configs['image_width'] = int(configs['image_width'])
    configs['num_epochs'] = int(configs['num_epochs'])
    configs['batch_size'] = int(configs['batch_size'])
    configs['warmup_steps'] = int(configs['warmup_steps'])
    configs['token_id'] = int(configs['token_id'])
    configs['num_workers'] = int(configs['num_workers'])
    configs['synth_epochs'] = int(configs['synth_epochs'])
    configs['num_synth_images'] = int(configs['num_synth_images'])
    print(configs)
    main = Main(configs)
    main.train()