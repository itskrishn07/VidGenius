import os
import tempfile
from services import AudioService, TranscriptionService, AnalysisService, RAGService

# Test YouTube media URL

# https://www.youtube.com/watch?v=_Q-e_nczWqM&t=223s
TEST_SOURCE = "https://youtu.be/K_N7686lCss?si=5frAASK71vAfTqAp"
TEST_LANGUAGE = "hinglish"

print("🧪 Running VidGenius Production Integration Test...")

try:
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"1. Processing audio from source into temp dir: {temp_dir}")
        chunks = AudioService.process_source(TEST_SOURCE, temp_dir)
        print(f"   ✓ Audio processed successfully into {len(chunks)} chunk(s).")

        print("2. Transcribing audio chunks...")
        transcript = TranscriptionService.transcribe_all(chunks, language=TEST_LANGUAGE, temp_dir=temp_dir)
        print(f"   ✓ Transcript generated ({len(transcript)} chars).")

    print("\n3. Testing Analysis Service...")
    title = AnalysisService.generate_title(transcript)
    print(f"   ✓ Title: {title}")

    summary = AnalysisService.summarize(transcript)
    print(f"   ✓ Summary Length: {len(summary)} chars")

    action_items = AnalysisService.extract_action_items(transcript)
    print(f"   ✓ Action Items extracted.")

    print("\n4. Testing RAG Service (Mistral AI Embeddings)...")
    rag_chain = RAGService.build_rag_chain(transcript)
    test_answer = RAGService.ask_question(rag_chain, "What is the main topic discussed?")
    print(f"   ✓ RAG Answer: {test_answer[:150]}...")

    print("\n🎉 ALL PIPELINE TESTS PASSED SUCCESSFULLY!")

except Exception as e:
    print(f"\n❌ Integration Test Failed: {e}")
    raise e
