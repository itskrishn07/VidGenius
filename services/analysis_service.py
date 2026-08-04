import concurrent.futures
from typing import List, Tuple
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import config


class AnalysisService:
    """Service handling transcript summarization, title generation, and structured insight extraction."""

    _llm = None

    @classmethod
    def get_llm(cls, temperature: float = 0.3) -> ChatMistralAI:
        if not config.MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY is not set in environment or .env file.")
        return ChatMistralAI(
            model=config.MISTRAL_MODEL,
            mistral_api_key=config.MISTRAL_API_KEY,
            temperature=temperature
        )

    @classmethod
    def generate_title(cls, transcript: str) -> str:
        """Generates a concise meeting title (max 8 words)."""
        llm = cls.get_llm(temperature=0.3)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Based on the meeting transcript, generate a short professional meeting title (max 8 words). Only return the title, nothing else."),
            ("human", "{text}"),
        ])
        chain = (
            RunnablePassthrough()
            | RunnableLambda(lambda x: {"text": x[:2000]})
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain.invoke(transcript)

    @classmethod
    def summarize(cls, transcript: str) -> str:
        """Performs parallel Map-Reduce summarization over the transcript."""
        llm = cls.get_llm(temperature=0.3)
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
        chunks = splitter.split_text(transcript)

        map_prompt = ChatPromptTemplate.from_messages([
            ("system", "Summarize this portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ])
        map_chain = map_prompt | llm | StrOutputParser()

        # Run chunk map summaries in parallel
        if len(chunks) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(chunks), 4)) as executor:
                chunk_summaries = list(executor.map(lambda c: map_chain.invoke({"text": c}), chunks))
        else:
            chunk_summaries = [map_chain.invoke({"text": chunks[0]})]

        combined_text = "\n\n".join(chunk_summaries)

        reduce_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert meeting summarizer. Combine these partial summaries into one final professional meeting summary in bullet points."),
            ("human", "{text}"),
        ])
        reduce_chain = (
            RunnablePassthrough()
            | RunnableLambda(lambda x: {"text": x})
            | reduce_prompt
            | llm
            | StrOutputParser()
        )
        return reduce_chain.invoke(combined_text)

    @classmethod
    def _build_extractor_chain(cls, system_prompt: str):
        llm = cls.get_llm(temperature=0.2)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{text}"),
        ])
        return (
            RunnablePassthrough()
            | RunnableLambda(lambda x: {"text": x})
            | prompt
            | llm
            | StrOutputParser()
        )

    @classmethod
    def extract_action_items(cls, transcript: str) -> str:
        chain = cls._build_extractor_chain(
            "You are an expert meeting analyst. From the meeting transcript, extract all action items. "
            "For each provide:\n- Task description\n- Owner (who is responsible)\n- Deadline (if mentioned, else write 'Not specified')\n\n"
            "Format as a numbered list. If none found say 'No action items found.'"
        )
        return chain.invoke(transcript)

    @classmethod
    def extract_key_decisions(cls, transcript: str) -> str:
        chain = cls._build_extractor_chain(
            "You are an expert meeting analyst. From the meeting transcript, extract all key decisions made. "
            "Format as a numbered list. If none found say 'No key decisions found.'"
        )
        return chain.invoke(transcript)

    @classmethod
    def extract_questions(cls, transcript: str) -> str:
        chain = cls._build_extractor_chain(
            "From the meeting transcript, extract all unresolved questions or topics needing follow-up. "
            "Format as a numbered list. If none found say 'No open questions found.'"
        )
        return chain.invoke(transcript)

    @classmethod
    def extract_all_insights(cls, transcript: str) -> Tuple[str, str, str]:
        """Extracts action items, key decisions, and open questions concurrently in parallel."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_actions = executor.submit(cls.extract_action_items, transcript)
            future_decisions = executor.submit(cls.extract_key_decisions, transcript)
            future_questions = executor.submit(cls.extract_questions, transcript)

            action_items = future_actions.result()
            decisions = future_decisions.result()
            questions = future_questions.result()

        return action_items, decisions, questions

