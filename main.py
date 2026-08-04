import tempfile
from services import AudioService, TranscriptionService, AnalysisService, RAGService
from models import PipelineResult


def run_pipeline(source: str, language: str = "english") -> PipelineResult:
    """Runs the full analysis pipeline using temporary file management."""
    print("🚀 Starting VidGenius Pipeline...")

    with tempfile.TemporaryDirectory() as temp_dir:
        print("🔊 Processing media source (downloading/converting & chunking)...")
        chunks = AudioService.process_source(source, temp_dir)

        print(f"📝 Transcribing audio ({len(chunks)} chunk(s)) using {language.upper()} engine...")
        transcript = TranscriptionService.transcribe_all(chunks, language=language, temp_dir=temp_dir)

    print(f"✅ Raw transcript generated ({len(transcript)} characters).")

    print("🏷️ Generating session title...")
    title = AnalysisService.generate_title(transcript)

    print("📋 Generating Map-Reduce summary...")
    summary = AnalysisService.summarize(transcript)

    print("🔍 Extracting action items, key decisions, and open questions...")
    action_items = AnalysisService.extract_action_items(transcript)
    decisions = AnalysisService.extract_key_decisions(transcript)
    questions = AnalysisService.extract_questions(transcript)

    print("🧠 Building RAG vector store with Mistral AI Embeddings...")
    rag_chain = RAGService.build_rag_chain(transcript)

    return PipelineResult(
        title=title,
        transcript=transcript,
        summary=summary,
        action_items=action_items,
        key_decisions=decisions,
        open_questions=questions,
        rag_chain=rag_chain
    )


if __name__ == "__main__":
    source_input = input("Enter YouTube URL or local file path: ").strip()
    language_input = input("Language (english/hinglish) [default: english]: ").strip() or "english"

    if not source_input:
        print("❌ Error: Media source path/URL cannot be empty.")
        exit(1)

    result = run_pipeline(source_input, language_input)

    print("\n" + "=" * 60)
    print(f"📌 TITLE: {result.title}")
    print(f"\n📋 SUMMARY:\n{result.summary}")
    print(f"\n✅ ACTION ITEMS:\n{result.action_items}")
    print(f"\n🔑 KEY DECISIONS:\n{result.key_decisions}")
    print(f"\n❓ OPEN QUESTIONS:\n{result.open_questions}")
    print("=" * 60)

    print("\n💬 Chat with your meeting recording (type 'exit' to quit)\n")
    while True:
        user_q = input("You: ").strip()
        if user_q.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not user_q:
            continue
        answer = RAGService.ask_question(result.rag_chain, user_q)
        print(f"\n🤖 Assistant: {answer}\n")