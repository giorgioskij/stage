import logging
from transformers import T5Tokenizer, T5EncoderModel
from typing import List, Optional, Dict, Sequence, Tuple
import random
import torch
from torch import Tensor

from stage.conditioning.embedder import Embedder, LinearProjectionEmbedder
from stage.conditioning.embedded_condition import EmbeddedCondition


class T5Embedder(LinearProjectionEmbedder):
    """T5-based TextConditioner.

    Args:
        name (str): Name of the T5 model.
        output_dim (int): Output dim of the conditioner.
        finetune (bool): Whether to fine-tune T5 at train time.
        word_dropout (float, optional): Word dropout probability.
        normalize_text (bool, optional): Whether to apply text normalization.
    """
    MODELS = [
        "t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b",
        "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
        "google/flan-t5-xl", "google/flan-t5-xxl"
    ]
    MODELS_DIMS = {
        "t5-small": 512,
        "t5-base": 768,
        "t5-large": 1024,
        "t5-3b": 1024,
        "t5-11b": 1024,
        "google/flan-t5-small": 512,
        "google/flan-t5-base": 768,
        "google/flan-t5-large": 1024,
        "google/flan-t5-3b": 1024,
        "google/flan-t5-11b": 1024,
    }

    def __init__(self,
                 embedding_dim: int,
                 t5_on_cpu: bool,
                 name: str = "t5-base",
                 finetune: bool = False,
                 word_dropout: float = 0.3):
        assert name in self.MODELS, f"Unrecognized t5 model name (should in {self.MODELS})"
        super().__init__(self.MODELS_DIMS[name], embedding_dim)
        self.name = name
        self.finetune = finetune
        self.word_dropout = word_dropout
        self.t5_on_cpu: bool = t5_on_cpu
        if self.t5_on_cpu and finetune:
            raise ValueError("Can't finetune t5 if it's locked on cpu")

        # Let's disable logging temporarily because T5 will vomit some errors otherwise.
        # thanks https://gist.github.com/simon-weber/7853144
        previous_level = logging.root.manager.disable
        logging.disable(logging.ERROR)
        self.t5_tokenizer = T5Tokenizer.from_pretrained(
            name,
            clean_up_tokenization_spaces=False,
        )

        t5 = T5EncoderModel.from_pretrained(name).train(mode=finetune)

        if self.t5_on_cpu:
            t5 = t5.cpu()
            self.__dict__["t5"] = t5
        else:
            self.t5 = t5

        if not self.finetune:
            for p in self.t5.parameters():
                p.requires_grad = False

        # if not self.finetune:
        #     if self.t5_on_cpu:
        #         t5 = t5.cpu()
        #     self.__dict__["t5"] = t5.eval()
        #     for p in self.t5.parameters():
        #         p.requires_grad = False
        # else:
        #     self.t5 = t5

        # with warnings.catch_warnings():

        #     warnings.simplefilter("ignore")
        #     try:
        #         self.t5_tokenizer = T5Tokenizer.from_pretrained(name)
        #         t5 = T5EncoderModel.from_pretrained(name).train(mode=finetune)
        #     finally:
        #         logging.disable(previous_level)
        # if finetune:
        #     self.t5 = t5
        # else:
        #     # this makes sure that the t5 models is not part
        #     # of the saved checkpoint
        #     self.__dict__['t5'] = t5

        # self.normalize_text = normalize_text
        # if normalize_text:
        #     self.text_normalizer = WhiteSpaceTokenizer(1,
        #                                                lemma=True,
        #                                                stopwords=True)

    # def to(self, *args, **kwargs):
    #     return super().to("cpu")

    # def cuda(self, *args, **kwargs):
    #     return self.to("cpu")

    def tokenize(self, x: Sequence[Optional[str]]) -> Dict[str, torch.Tensor]:
        # if current sample doesn't have a certain attribute, replace with empty string
        entries: List[str] = [xi if xi is not None else "" for xi in x]
        # if self.normalize_text:
        #     _, _, entries = self.text_normalizer(  # type: ignore
        #         entries, return_text=True)
        if self.word_dropout > 0. and self.training:
            new_entries = []
            for entry in entries:
                words = [
                    word for word in entry.split(" ")
                    if random.random() >= self.word_dropout
                ]
                new_entries.append(" ".join(words))
            entries = new_entries

        empty_idx = torch.LongTensor(
            [i for i, xi in enumerate(entries) if xi == ""])

        inputs = self.t5_tokenizer(entries, return_tensors='pt',
                                   padding=True).to(
                                       next(iter(self.parameters())).device)
        mask = inputs['attention_mask']
        mask[empty_idx, :] = 0  # zero-out index where the input is non-existant
        return inputs

    def forward(self,
                descriptions: List[str],
                duplicate_for_cfg: bool = False) -> EmbeddedCondition:

        if duplicate_for_cfg:
            descriptions = descriptions + ([""] * len(descriptions))

        tokenized = self.tokenize(descriptions)
        embedding, mask = self.embed(tokenized)
        mask = mask.bool()
        return EmbeddedCondition(embedding, mask)

    def embed(self, inputs: Dict[str, torch.Tensor]) -> Tuple[Tensor, Tensor]:
        if not self.finetune:
            self.t5.eval()
        mask = inputs['attention_mask']
        with torch.set_grad_enabled(self.finetune):
            if self.t5_on_cpu:
                inputs = {k: v.to("cpu") for k, v in inputs.items()}
            with torch.autocast(device_type="cuda", enabled=False):
                embeds = self.t5(**inputs).last_hidden_state
        embeds = self.output_proj(embeds.to(self.output_proj.weight))
        embeds = (embeds * mask.unsqueeze(-1).to(embeds))
        return embeds, mask

    def null_condition(self, batch_size: int):
        return EmbeddedCondition(
            torch.zeros(batch_size,
                        1,
                        self.embedding_dim,
                        dtype=torch.float32,
                        device=self.output_proj.weight.device),
            torch.ones(batch_size,
                       1,
                       dtype=torch.bool,
                       device=self.output_proj.weight.device))


class T5EmbedderCPU(T5Embedder):

    def __init__(self, embedding_dim: int):
        super().__init__(embedding_dim,
                         t5_on_cpu=True,
                         name="t5-base",
                         finetune=False,
                         word_dropout=0.3)


class T5EmbedderGPU(T5Embedder):

    def __init__(self, embedding_dim: int):
        super().__init__(embedding_dim,
                         t5_on_cpu=False,
                         name="t5-base",
                         finetune=False,
                         word_dropout=0.3)


if __name__ == "__main__":
    from stage.utils.inspection import print_params
    from time import time

    t5 = T5EmbedderCPU(1024).eval().cpu()
    # print_params(t5, 1, False)

    descriptions = ["the quick"]

    embedded_desc = t5(descriptions)
    # print(embedded_desc)
    # print(embedded_desc.data)
    # print(embedded_desc.mask)

    # print(embedded_desc.data[..., -1, ...])
    # print(embedded_desc.data[0, -1, ...])
