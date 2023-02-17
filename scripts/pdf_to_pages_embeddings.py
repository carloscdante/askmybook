from dotenv import load_dotenv
import os
import tiktoken

enc = tiktoken.encoding_for_model("text-embedding-ada-002")

# Use load_env to trace the path of .env:
load_dotenv('.env')

import pandas as pd
from typing import Set
from transformers import GPT2TokenizerFast
import argparse, sys

from itertools import islice

import numpy as np

from PyPDF2 import PdfReader

import pandas as pd
import openai
import csv
import numpy as np
import os
import pickle
from transformers import GPT2TokenizerFast

openai.api_key = os.environ["OPENAI_API_KEY"]

COMPLETIONS_MODEL = "text-davinci-003"

MODEL_NAME = "curie"

DOC_EMBEDDINGS_MODEL = f"text-embedding-ada-002"

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch

def count_tokens(text: str) -> int:
    """count the number of tokens in a string"""
    return len(tokenizer.encode(text))

def extract_pages(
    page_text: str,
    index: int,
) -> str:
    """
    Extract the text from the page
    """
    if len(page_text) == 0:
        return []

    content = " ".join(page_text.split())
    outputs = [("Page " + str(index), content, count_tokens(content)+4)]

    return outputs

parser=argparse.ArgumentParser()

parser.add_argument("--pdf", help="Name of PDF")

args=parser.parse_args()

filename = f"{args.pdf}"

reader = PdfReader(filename)

res = []
i = 1
for page in reader.pages:
    res += extract_pages(page.extract_text(), i)
    i += 1
df = pd.DataFrame(res, columns=["title", "content", "tokens"])
# df = df[df.tokens<2046]
df = df.reset_index().drop('index',axis=1) # reset index
df.head()

df.to_csv(f'{filename}.pages.csv', index=False)

def get_embedding(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def chunked_tokens(text, chunk_length):
    encoding = tiktoken.get_encoding(enc.name)
    tokens = encoding.encode(text)
    chunks_iterator = batched(tokens, chunk_length)
    yield from chunks_iterator

def len_safe_get_embedding(text, model=DOC_EMBEDDINGS_MODEL, max_tokens=8191, average=True):
    chunk_embeddings = []
    chunk_lens = []
    for chunk in chunked_tokens(text, max_tokens):
        chunk_embeddings.append(get_embedding(chunk, model=model))
        chunk_lens.append(len(chunk))
    if average:
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)  # normalizes length to 1
        chunk_embeddings = chunk_embeddings.tolist()
    return chunk_embeddings


# def get_doc_embedding(text: str) -> list[float]:
#     return get_embedding(text, DOC_EMBEDDINGS_MODEL)

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str], list[float]]:
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.

    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: len_safe_get_embedding(r.content) for idx, r in df.iterrows()
    }

# CSV with exactly these named columns:
# "title", "0", "1", ... up to the length of the embedding vectors.

doc_embeddings = compute_doc_embeddings(df)

print(doc_embeddings)

with open(f'{filename}.embeddings.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["title"] + list(range(1536)))
    for i, embedding in list(doc_embeddings.items()):
        print('now formatting ', embedding)
        writer.writerow(["Page " + str(i + 1)] + embedding)
