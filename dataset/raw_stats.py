import datasets
import numpy as np
from transformers import AutoTokenizer

ds = datasets.load_dataset("metareflection/dafny-docs", split="train")
tokenizer = AutoTokenizer.from_pretrained("bigcode/starencoder")

toks = []
for i, ex in enumerate(ds):
    toks.append(len(tokenizer.encode(ex["content"], add_special_tokens=False)))

print("Mean:", np.mean(toks))
print("Median:", np.median(toks))
print("Max:", np.max(toks))
print("Min:", np.min(toks))

from matplotlib import pyplot as plt
plt.hist(toks, bins=100, cumulative=True, density=True)
plt.show()

