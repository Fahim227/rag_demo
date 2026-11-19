import uuid

from dotenv import load_dotenv
from fastapi import FastAPI
import logging, inngest
import inngest.fast_api
from data_loader import load_and_chunk_pdf, embed_text, gemini
from google.genai import types
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGQueryResults, RAGQueryResults, RAGSearchResults, RAGUpsertSuccess
load_dotenv()


inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

@inngest_client.create_function(
    fn_id="RAG: Inngest PDF",
    name="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id",pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks,source_id=source_id)

    def _upsert(chunk_and_src: RAGChunkAndSrc) -> RAGUpsertSuccess:
        chunks = chunk_and_src.chunks
        source_id = chunk_and_src.source_id
        vectors = embed_text(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        qdrant_storage = QdrantStorage()
        qdrant_storage.upsert(ids,vectors,payloads)
        return  RAGUpsertSuccess(ingested=len(chunks))


    chunks_and_src = await ctx.step.run("load-and-chunk",lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert",lambda: _upsert(chunks_and_src), output_type=RAGUpsertSuccess)
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_answer(ctx: inngest.Context):
    def _query(question,top_k:int=5) -> RAGSearchResults:
        query_vec= embed_text([question])[0]
        storage = QdrantStorage()
        result = storage.search(query_vec,top_k)
        return RAGSearchResults(contexts=result["contexts"],sources=result["sources"])

    def _answer(question: str, search_results: RAGSearchResults) -> RAGQueryResults:
        context_block = "\n\n".join(f"- {c}" for c in search_results.contexts)
        prompt = (
            "Use the following context to answer the question.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
            "Answer concisely using only the context above. "
            "If the context does not contain the answer, say you don't know."
        )
        response = gemini.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You answer questions using only the provided context.",
                temperature=0.2,
            ),
        )
        return RAGQueryResults(
            answer=(response.text or "").strip(),
            sources=search_results.sources,
            num_contexts=len(search_results.contexts),
        )

    query =   ctx.event.data["question"]
    top_k =   ctx.event.data.get("top_k",5)
    found = await ctx.step.run("embed-and-search",lambda :_query(question=query,top_k=top_k),output_type=RAGSearchResults)

    result = await ctx.step.run("llm-answer", lambda: _answer(query, found), output_type=RAGQueryResults)
    return result.model_dump()



app = FastAPI()

inngest.fast_api.serve(app,inngest_client,[rag_ingest_pdf,rag_query_answer])