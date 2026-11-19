from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types


load_dotenv()

# Initialize clients
gemini = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"]
)

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM=3072
splitter = SentenceSplitter(chunk_size=1000,chunk_overlap=200)

def load_and_chunk_pdf(path:str):
    docs = PDFReader().load_data(file=path)
    texts = [doc.text for doc in docs if getattr(doc,"text",None)]
    chunks = []
    print(type(splitter))
    for text in texts:
        chunks.extend(splitter.split_text(text))

    return chunks

def embed_text(chunks:list[str]) -> list[list[float]]:
    results = gemini.models.embed_content(
        model=EMBED_MODEL,
        contents=chunks,
        config=types.EmbedContentConfig(
            output_dimensionality=EMBED_DIM
        )
    )
    embeddings = results.embeddings

    if embeddings is None:
        raise ValueError("Gemini returned no embeddings")

    vectors: list[list[float]] = []

    for embedding in embeddings:
        if embedding.values is None:
            raise ValueError("Embedding values are None")

        vectors.append(embedding.values)

    return vectors
