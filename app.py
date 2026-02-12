import os
import tempfile
import streamlit as st
import re
from rag import build_vectorstore_from_pdf, answer_question

st.set_page_config(
    page_title="Smart PDF Chat (RAG)",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main Background - Clean White */
    .stApp {
        background: #FFFFFF;
    }

    /* Sidebar Styling - Purple Theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F8F7FF 0%, #F3F1FF 100%);
        border-right: 1px solid #E9E3FF;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown {
        color: #1F1F1F !important;
    }

    /* Main Content Area */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }

    /* Title Styling */
    h1 {
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        background: linear-gradient(135deg, #7C3AED 0%, #C026D3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* Caption/Subtitle (FIXED) */
    .stCaption {
        color: #111827 !important;
        font-size: 1rem !important;
        margin-bottom: 2rem !important;
    }

    /* Buttons - Purple Theme */
    .stButton > button {
        background: linear-gradient(135deg, #7C3AED 0%, #9333EA 100%);
        color: white !important;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.3);
        width: 100%;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #6D28D9 0%, #7C3AED 100%);
        box-shadow: 0 6px 16px rgba(124, 58, 237, 0.4);
        transform: translateY(-2px);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: white;
        border: 2px dashed #C4B5FD;
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #7C3AED;
        background: #FAF9FF;
    }

    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] small {
        color: #111827 !important;
    }

    [data-testid="stFileUploader"] small {
        color: #6B7280 !important;
    }

    /* Slider */
    .stSlider {
        padding: 1rem 0;
    }

    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #7C3AED 0%, #C026D3 100%);
    }

    .stSlider label {
        color: #111827 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Chat Messages */
    .stChatMessage {
        background: white;
        border-radius: 16px;
        padding: 1.25rem;
        margin: 1rem 0;
        border: 1px solid #F3F4F6;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        opacity: 1 !important;
        filter: none !important;
    }

    /* User Messages */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #7C3AED 0%, #9333EA 100%);
        border: none;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.2);
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) p,
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) span {
        color: white !important;
    }

    /* Assistant Messages */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: #FAFAFA;
        border: 1px solid #E5E7EB;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) p,
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) span {
        color: #000000 !important;
        opacity: 1 !important;
    }

    /* Chat message markdown */
    [data-testid="stMarkdownContainer"] {
        color: #000000 !important;
        opacity: 1 !important;
    }

    /* Chat Input */
    .stChatInputContainer {
        border-top: 1px solid #F3F4F6;
        padding-top: 1rem;
        background: white;
    }

    .stChatInput {
        border-radius: 12px;
    }

    .stChatInput > div {
        border-radius: 12px;
        border: 2px solid #E9E3FF;
        background: white;
    }

    .stChatInput > div:focus-within {
        border-color: #7C3AED;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1);
    }

    /* Fix chat input text visibility */
    .stChatInput input,
    .stChatInput textarea {
        color: #ffffff !important;
        font-size: 1.5rem !important;
    }

    .stChatInput input::placeholder,
    .stChatInput textarea::placeholder {
        color: #9333EA !important;
        opacity: 1 !important;
    }

    /* Success/Info/Warning Messages */
    .stSuccess {
        background: linear-gradient(135deg, #D1FAE5 0%, #A7F3D0 100%);
        color: #065F46 !important;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        border: 1px solid #6EE7B7;
        font-weight: 500;
    }

    .stInfo {
        background: linear-gradient(135deg, #E0E7FF 0%, #C7D2FE 100%);
        color: #3730A3 !important;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        border: 1px solid #A5B4FC;
        font-weight: 500;
    }

    /* Divider */
    hr {
        margin: 1.5rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #E9E3FF, transparent);
    }

    /* Sidebar Headers */
    [data-testid="stSidebar"] h2 {
        color: #7C3AED !important;
        font-weight: 700 !important;
        font-size: 1.25rem !important;
        margin-bottom: 1rem !important;
    }

    [data-testid="stSidebar"] h3 {
        color: #111827 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-top: 1rem !important;
        margin-bottom: 0.75rem !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #7C3AED !important;
        border-right-color: #7C3AED !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #F9FAFB;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #7C3AED, #C026D3);
        border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #6D28D9, #A21CAF);
    }

    /* Ensure all paragraph text is visible */
    p {
        color: #111827 !important;
    }
</style>
""", unsafe_allow_html=True)

# Main Title and Caption
st.title("📄 Smart PDF Chat — RAG")
st.caption("Upload a PDF, then chat with it using RAG (FAISS + local embeddings + Groq LLM)")

# Initialize Session State
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "chat" not in st.session_state:
    st.session_state.chat = []

# Sidebar for controls and uploads
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Top-K chunks to retrieve", 2, 8, 4)

    st.divider()
    st.subheader("📤 Upload PDF")
    pdf_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if pdf_file is not None:
        with st.spinner("🔄 Indexing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.read())
                tmp_path = tmp.name

            try:
                st.session_state.vectorstore = build_vectorstore_from_pdf(tmp_path)
                st.success("✅ PDF indexed! You can now ask questions.")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.chat = []
        st.rerun()

    st.divider()
    st.markdown("""
    <div style='background: white; padding: 1rem; border-radius: 12px; border: 1px solid #E9E3FF;'>
        <p style='margin: 0; font-size: 0.85rem; color: #000000;'>
            <strong style='color: #000000;'>💡 Tip:</strong><br/>
            Increase Top-K for more context in complex questions.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main Chat Interface
if st.session_state.vectorstore is None:
    st.markdown("""
    <div style='text-align: center; padding: 4rem 2rem; background: linear-gradient(135deg, #F8F7FF 0%, #FAFAFA 100%); border-radius: 20px; margin-top: 2rem;'>
        <div style='font-size: 4rem; margin-bottom: 1rem;'>📚</div>
        <h2 style='color: #111827; margin-bottom: 1rem;'>No PDF Loaded</h2>
        <p style='color: #6B7280; font-size: 1.1rem; max-width: 500px; margin: 0 auto;'>
            Upload a PDF from the sidebar to start chatting with your documents using AI
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 3rem;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 16px; text-align: center; border: 1px solid #F3F4F6; box-shadow: 0 2px 8px rgba(0,0,0,0.04);'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>📄</div>
            <h3 style='color: #111827; margin-bottom: 0.5rem; font-size: 1.1rem;'>Upload</h3>
            <p style='color: #6B7280; margin: 0; font-size: 0.9rem;'>Upload any PDF document</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 16px; text-align: center; border: 1px solid #F3F4F6; box-shadow: 0 2px 8px rgba(0,0,0,0.04);'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>🤖</div>
            <h3 style='color: #111827; margin-bottom: 0.5rem; font-size: 1.1rem;'>AI Analysis</h3>
            <p style='color: #6B7280; margin: 0; font-size: 0.9rem;'>Smart RAG technology</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style='background: white; padding: 2rem; border-radius: 16px; text-align: center; border: 1px solid #F3F4F6; box-shadow: 0 2px 8px rgba(0,0,0,0.04);'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>💬</div>
            <h3 style='color: #111827; margin-bottom: 0.5rem; font-size: 1.1rem;'>Get Answers</h3>
            <p style='color: #ffffff; margin: 0; font-size: 0.9rem;'>Ask any question</p>
        </div>
        """, unsafe_allow_html=True)

    st.stop()

# Display previous messages
for u, a in st.session_state.chat:
    with st.chat_message("user"):
        st.write(u)
    with st.chat_message("assistant"):
        st.write(a)

# Handle new questions
question = st.chat_input("💭 Ask something about the PDF…")

if question:
    with st.chat_message("user"):
        st.write(question)

    # ✅ define q first
    q = question.lower().strip()

    # ✅ Word-based greeting detection (prevents "hi" inside "which")
    greeting_words = {"hi", "hello", "hey", "salam", "assalam", "aoa"}
    greeting_phrases = {
        "good morning", "good evening", "good afternoon",
        "how are you", "what's up", "whats up"
    }

    tokens = re.findall(r"[a-z']+", q)  # words only

    is_greeting = False

    # phrase greetings
    if any(phrase in q for phrase in greeting_phrases):
        is_greeting = True

    # single word greetings if first word is greeting
    elif tokens and tokens[0] in greeting_words:
        is_greeting = True

    # short messages like "hi bro", "hello there"
    elif len(tokens) <= 3 and any(t in greeting_words for t in tokens):
        is_greeting = True

    if is_greeting:
        ans = (
            "Hello 👋\n\n"
            "I'm your PDF assistant.\n"
            "Ask me anything about the document you uploaded — "
            "summaries, explanations, details, anything 🙂"
        )
    else:
        with st.spinner("🤔 Thinking..."):
            ans = answer_question(
                vectorstore=st.session_state.vectorstore,
                question=question,
                chat_history=st.session_state.chat,
                k=k,
            )

    st.session_state.chat.append((question, ans))

    with st.chat_message("assistant"):
        st.write(ans)
