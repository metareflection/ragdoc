from typing import List
from transformers import AutoTokenizer, AutoModel
from scoring import mean_pool
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Tuple
from scoring import score, score_against_space
import pickle
import numpy as np
import torch

class StarEncoder:
    MASK_TOKEN = "<mask>"
    SEPARATOR_TOKEN = "<sep>"
    PAD_TOKEN = "<pad>"
    CLS_TOKEN = "<cls>"

    def __init__(self, max_length, device) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bigcode/starencoder",
            truncation_side="right",
        )
        self.model = AutoModel.from_pretrained(
            "bigcode/starencoder",
        ).to(device)
        self.tokenizer.add_special_tokens({"pad_token": self.PAD_TOKEN})
        self.tokenizer.add_special_tokens({"sep_token": self.SEPARATOR_TOKEN})
        self.tokenizer.add_special_tokens({"cls_token": self.CLS_TOKEN})
        self.tokenizer.add_special_tokens({"mask_token": self.MASK_TOKEN})
        self.max_length = max_length
        self.device = device

    def tokenizer_encode(self, sentences: List[str]):
        return self.tokenizer(
            [f"{self.CLS_TOKEN}{sentence}{self.SEPARATOR_TOKEN}" for sentence in sentences],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

    def get_emb(self, toks):
        outputs = self.model(**toks)
        return mean_pool(toks['attention_mask'], outputs.last_hidden_state)


class EmbeddingSpace:
    def __init__(self, from_file=None):
        if from_file is not None:
            with open(from_file, "rb") as f:
                es = pickle.load(f)
                self.embs = es.embs
                self.solutions = es.solutions
                self.max_length = es.max_length
                self.model_name = es.model_name
        else:
            self.embs = None
            self.solutions = []
            self.max_length = 0
            self.model_name = None

    def load_model(self):
        return StarEncoder(self.max_length, device="cuda" if torch.cuda.is_available() else "cpu")

    def add_examples(self, model: StarEncoder, examples: Tuple[List[str], List[str]]):
        # TODO: avoid dups
        prompts, solutions = examples
        ex_toks = model.tokenizer_encode(prompts)
        ex_emb = model.get_emb(ex_toks)
        if self.embs is None:
            self.embs = ex_emb
            self.solutions = solutions
        else:
            assert self.embs is not None
            self.embs = np.concatenate((self.embs, ex_emb))
            self.solutions.extend(solutions)

    def most_similar(self, model: StarEncoder, prompt: str, top_n: int = 5) -> List[Tuple[str, float]]:
        assert self.embs is not None
        # bound n by the number of solutions
        top_n = min(top_n, len(self.solutions))
        prompt_toks = model.tokenizer_encode([prompt])
        prompt_emb = model.get_emb(prompt_toks)[0]
        scores = score_against_space(self.embs, prompt_emb)
        # get the indices of the top n scores
        top_n_indices = np.argsort(scores)[-top_n:][::-1]
        solutions = [(self.solutions[i], scores[i])
                     for i in top_n_indices]
        return solutions

    def score_matrix(self):
        return score(self.embs)

    def size(self):
        return len(self.solutions)


def score(mean_pooled):
    scores = np.zeros((mean_pooled.shape[0], mean_pooled.shape[0]))
    for i in range(mean_pooled.shape[0]):
        scores[i, :] = cosine_similarity(
            [mean_pooled[i]],
            mean_pooled
        )[0]
    return scores


def score_against_space(space, example):
    """
    Scores an example against an embedding space.
    Both must be mean pooled.
    """
    return cosine_similarity(
        [example],
        space
    )[0]
