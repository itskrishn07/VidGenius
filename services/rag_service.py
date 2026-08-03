from typing import List
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import config


class RAGService:
    """Service handling vector embedding, Chroma indexing, and RAG retrieval using Mistral AI Embeddings."""

    @staticmethod
    def get_embeddings() -> MistralAIEmbeddings:
        if not config.MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY is not set in environment or .env file.")
        return MistralAIEmbeddings(
            model=config.MISTRAL_EMBED_MODEL,
            mistral_api_key=config.MISTRAL_API_KEY
        )

    @staticmethod
    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join([doc.page_content for doc in docs])

    @classmethod
    def build_rag_chain(cls, transcript: str):
        """Builds an in-memory session-isolated Chroma RAG pipeline using Mistral Embeddings."""
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

        llm = ChatMistralAI(
            model=config.MISTRAL_MODEL,
            mistral_api_key=config.MISTRAL_API_KEY,
            temperature=0.3
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
        return rag_chain.invoke(question.strip())
