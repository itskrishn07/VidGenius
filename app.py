import streamlit as st
import tempfile
import time
from services import AudioService, TranscriptionService, AnalysisService, RAGService
from models import PipelineResult
from ui import inject_custom_css, render_step_bar

# ─── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VidGenius",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Inject Styling ───────────────────────────────────────────────────────────
inject_custom_css()

# ─── Session State Initialization ─────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">🎬 VidGenius</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<span class="badge badge-purple" style="margin-bottom:0.5rem;display:inline-block">Input Media</span>', unsafe_allow_html=True)
    source = st.text_input(
        "YouTube URL or File Path",
        placeholder="",
        help="Paste a YouTube link or local media file path."
    )

    language = st.selectbox("Speech Language", ["english", "hinglish"], index=0)

    run_btn = st.button("⚡  Analyse", use_container_width=True)

    if st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)
        for step, icon, label in [
            ("audio",      "🔊", "Audio Acquisition"),
            ("transcript", "📝", "Transcription"),
            ("title",      "🏷️", "Title Generation"),
            ("summary",    "📋", "Summarisation"),
            ("extract",    "🔍", "Insight Extraction"),
            ("rag",        "🧠", "RAG Vector Store"),
        ]:
            render_step_bar(label, step, icon, st.session_state.pipeline_steps)

# ─── Main View Header ────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">VidGenius</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Summarise · Chat with your media</div>', unsafe_allow_html=True)

st.markdown("---")

# ─── Pipeline Execution ───────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please enter a valid YouTube URL or local file path.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        progress_placeholder = st.empty()

        def set_step(k, v):
            st.session_state.pipeline_steps[k] = v

        try:
            with progress_placeholder.container():
                st.info("⚙️ Processing media pipeline — observe progress in sidebar…")

            # Use TemporaryDirectory for safe automatic cleanup of downloaded WAVs and chunks
            with tempfile.TemporaryDirectory() as temp_dir:
                set_step("audio", "active")
                chunks = AudioService.process_source(source, temp_dir)
                set_step("audio", "done")

                set_step("transcript", "active")
                transcript = TranscriptionService.transcribe_all(chunks, language=language, temp_dir=temp_dir)
                set_step("transcript", "done")

            # Non-file dependent analysis & vector embedding
            set_step("title", "active")
            title = AnalysisService.generate_title(transcript)
            set_step("title", "done")

            set_step("summary", "active")
            summary = AnalysisService.summarize(transcript)
            set_step("summary", "done")

            set_step("extract", "active")
            action_items = AnalysisService.extract_action_items(transcript)
            decisions = AnalysisService.extract_key_decisions(transcript)
            questions = AnalysisService.extract_questions(transcript)
            set_step("extract", "done")

            set_step("rag", "active")
            rag_chain = RAGService.build_rag_chain(transcript)
            set_step("rag", "done")

            result_obj = PipelineResult(
                title=title,
                transcript=transcript,
                summary=summary,
                action_items=action_items,
                key_decisions=decisions,
                open_questions=questions,
                rag_chain=rag_chain
            )

            st.session_state.result = result_obj
            st.session_state.pipeline_done = True
            progress_placeholder.success("✅ Media analysis complete!")
            time.sleep(0.4)
            progress_placeholder.empty()
            st.rerun()

        except Exception as e:
            for k in ["audio", "transcript", "title", "summary", "extract", "rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            progress_placeholder.error(f"❌ Pipeline Error: {e}")

# ─── Render Results Dashboard ──────────────────────────────────────────────────
if st.session_state.result:
    res: PipelineResult = st.session_state.result

    # Session Title Banner
    st.markdown(f"""
    <div class="card">
        <div class="card-title">📌 Media Title</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:var(--text)">
            {res.title}
        </div>
    </div>""", unsafe_allow_html=True)

    # Executive Summary & Transcript Split View
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📋 Executive Summary</div>
            <div class="card-content">{res.summary}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        with st.expander("📝 Complete Transcript", expanded=False):
            st.markdown(f'<div class="transcript-box">{res.transcript}</div>', unsafe_allow_html=True)

    # Structured Insights Grid
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">✅ Action Items</div>
            <div class="card-content">{res.action_items}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🔑 Key Decisions</div>
            <div class="card-content">{res.key_decisions}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">❓ Open Questions</div>
            <div class="card-content">{res.open_questions}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Interactive RAG Chat ─────────────────────────────────────────────────
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:1.2rem;font-weight:700;margin-bottom:1rem">💬 Interactive Meeting Q&A</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-end">
                    <span class="chat-label user-label">You</span>
                    <div class="chat-bubble user-bubble">{msg['content']}</div>
                </div>"""
            else:
                chat_html += f"""
                <div class="chat-msg" style="align-items:flex-start">
                    <span class="chat-label bot-label">🤖 Assistant</span>
                    <div class="chat-bubble bot-bubble">{msg['content']}</div>
                </div>"""
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2rem">
            <div style="font-size:2rem;margin-bottom:0.5rem">💬</div>
            <div style="color:var(--text-muted);font-size:0.85rem">Ask any question to search and query your meeting recording.</div>
        </div>""", unsafe_allow_html=True)

    chat_col1, chat_col2 = st.columns([5, 1], gap="small")
    with chat_col1:
        user_query = st.text_input("Your question", placeholder="What decisions were made regarding timeline?", label_visibility="collapsed")
    with chat_col2:
        send_btn = st.button("Send →", use_container_width=True)

    if send_btn and user_query.strip():
        with st.spinner("Searching transcript context…"):
            answer = RAGService.ask_question(res.rag_chain, user_query.strip())
        st.session_state.chat_history.append({"role": "user", "content": user_query.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # Initial Empty State
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:5rem 2rem;text-align:center">
        <div style="font-size:4rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready for Analysis
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:400px;line-height:1.7">
            Paste a YouTube URL or local file path in the sidebar, choose your language, and click <strong>Analyse</strong>.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">Groq Whisper</span>
            <span class="badge badge-cyan">Mistral AI Summaries</span>
            <span class="badge badge-green">Mistral Embeddings RAG</span>
        </div>
    </div>""", unsafe_allow_html=True)