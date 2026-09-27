from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import config


class RAGService:
    """Service handling vector embedding, Chroma indexing, and RAG retrieval using Google AI Embeddings."""

    @staticmethod
    def get_embeddings() -> GoogleGenerativeAIEmbeddings:
        if not config.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is not set in environment or .env file.")
        return GoogleGenerativeAIEmbeddings(
            model=config.GEMINI_EMBED_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            max_retries=6
        )

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join([doc.page_content for doc in docs])

    @classmethod
    def build_rag_chain(cls, transcript: str):
        """Builds an in-memory session-isolated Chroma RAG pipeline using Google Embeddings."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.RAG_CHUNK_SIZE,
            chunk_overlap=config.RAG_CHUNK_OVERLAP
        )
        chunks = splitter.split_text(transcript)

        docs = [
            Document(page_content=chunk, metadata={"chunk_index": i})
            for i, chunk in enumerate(chunks)
        ]

        embeddings = cls.get_embeddings()

        # In-memory Chroma vector store scoped to current processing session
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embeddings
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.RAG_TOP_K}
        )

        llm = ChatGoogleGenerativeAI(
            model=config.GEMINI_MODEL,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=0.3,
            max_retries=6
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise and precise. If quoting someone, mention it clearly.

Context from meeting transcript:
{context}"""
            ),
            ("human", "{question}")
        ])

        rag_chain = (
            {
                "context": retriever | RunnableLambda(cls.format_docs),
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain

    @staticmethod
    def ask_question(rag_chain, question: str) -> str:
        """Invokes RAG chain with user query."""
        if not question or not question.strip():
            return "Please provide a valid question."
        try:
            return rag_chain.invoke(question.strip())
        except Exception as e:
            err_msg = str(e)
            if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg or "Quota exceeded" in err_msg:
                return "⚠️ Rate limit reached on Gemini Free Tier. Please wait a few seconds and try your question again."
            raise e

    @staticmethod
    def stream_question(rag_chain, question: str):
        """Streams RAG chain tokens for real-time response generation."""
        if not question or not question.strip():
            yield "Please provide a valid question."
            return
        try:
            for chunk in rag_chain.stream(question.strip()):
                yield chunk
        except Exception as e:
            err_msg = str(e)
            if "RESOURCE_EXHAUSTED" in err_msg or "429" in err_msg or "Quota exceeded" in err_msg:
                yield "⚠️ Rate limit reached on Gemini Free Tier. Please wait a few seconds and try again."
            else:
                raise e

