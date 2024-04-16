import datasets
import re
from transformers import AutoTokenizer
from typing import List

ds = datasets.load_dataset("metareflection/dafny-docs", split="train")
tokenizer = AutoTokenizer.from_pretrained("bigcode/starencoder")


def remove_html_comments(string):
    return re.sub(r'<!--.*?-->', '', string)


def too_big(text):
    return len(tokenizer.encode(text, add_special_tokens=False)) > 1000


def segment(text) -> List[str]:
    if not too_big(text):
        return [text]
    else:
        print("Splitting")
        lines = text.split("\n")
        header = lines[0]
        rest = lines[1:]
        chunks = []
        chunk = ""
        in_codeblock = False
        for l in rest:
            chunk += l + "\n"
            if "```" in l:
                if in_codeblock:
                    chunks.append(chunk)
                    chunk = ""
                in_codeblock = not in_codeblock

        if chunk and too_big(chunk + chunks[-1]):
            chunks[-1] += chunk

        # filter chunks that are too big
        chunks = [c for c in chunks if not too_big(c)]

        # unite chunks until too big
        new_chunks = []
        chunk = ""
        while chunks:
            c = chunks.pop(0)
            if too_big(chunk + c):
                new_chunks.append(chunk)
                chunk = c
            else:
                chunk += c

        if chunk:
            new_chunks.append(chunk)

        new_chunks = [header + "\n" + c for c in new_chunks]
        for c in new_chunks:
            print(f"############ CHUNK ############")
            print(c)

        assert all(not too_big(c) for c in new_chunks)

        return new_chunks


docs = []
for ex in ds:
    text = remove_html_comments(ex["content"]).strip()
    docs.extend(segment(text))

chunked_docs = datasets.Dataset.from_dict({"content": docs})
chunked_docs.push_to_hub("metareflection/dafny-docs-chunked", private=True)
