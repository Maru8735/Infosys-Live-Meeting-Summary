import streamlit as st
import pandas as pd
import time
from models import STTModel, Diarizer, Summarizer
from evaluation import get_benchmark_report, calculate_wer
from export import export_as_json, export_as_markdown, export_as_csv
from fpdf import FPDF
import yagmail
import os

# Configure page layout to use full width
st.set_page_config(layout="wide", page_title="Live Meeting Summarizer")

# Pure Glass (Glassmorphism) UI Styling
st.markdown("""
<style>
    /* Multi-layered blended background - Purple & Orange Blend */
    body {
        background-image: 
            radial-gradient(ellipse at 30% 50%, rgba(200, 50, 255, 0.25) 0%, transparent 50%),
            radial-gradient(ellipse at 70% 60%, rgba(255, 140, 50, 0.22) 0%, transparent 50%),
            linear-gradient(180deg, rgba(15, 10, 40, 0.4) 0%, rgba(10, 5, 25, 0.3) 100%),
            url('https://raw.githubusercontent.com/Maru8735/Infosys-Live-Meeting-Summary/main/background.jpeg');
        background-size: cover, cover, cover, cover;
        background-position: center, center, center, center;
        background-attachment: fixed;
        background-blend-mode: screen, screen, multiply, normal;
        background-color: #0a0a1a;
    }
    
    html {
        background-image: 
            radial-gradient(ellipse at 30% 50%, rgba(200, 50, 255, 0.25) 0%, transparent 50%),
            radial-gradient(ellipse at 70% 60%, rgba(255, 140, 50, 0.22) 0%, transparent 50%),
            linear-gradient(180deg, rgba(15, 10, 40, 0.4) 0%, rgba(10, 5, 25, 0.3) 100%),
            url('https://raw.githubusercontent.com/Maru8735/Infosys-Live-Meeting-Summary/main/background.jpeg');
        background-size: cover, cover, cover, cover;
        background-position: center, center, center, center;
        background-attachment: fixed;
        background-blend-mode: screen, screen, multiply, normal;
        background-color: #0a0a1a;
    }
    
    /* Enhanced overlay for transparency effect with blend */
    [data-testid="stAppViewContainer"] {
        background: 
            linear-gradient(180deg, rgba(5, 0, 20, 0.25) 0%, rgba(8, 3, 15, 0.15) 100%) !important;
        backdrop-filter: blur(3px);
        -webkit-backdrop-filter: blur(3px);
    }
    
    /* Remove default padding from top */
    .block-container {
        padding-top: 0rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    
    /* Main container with pure glass effect */
    [data-testid="stMainBlockContainer"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    }
    
    /* Glass effect for input fields */
    input, textarea, [data-testid="stTextInput"], [data-testid="stTextArea"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        color: #FFFFFF !important;
    }
    
    input::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    /* Glass effect for file uploader */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }
    
    /* Glass effect for buttons */
    .stButton > button {
        background: rgba(76, 175, 80, 0.7) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 10px !important;
        color: white !important;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
    }
    
    .stButton > button:hover {
        background: rgba(76, 175, 80, 0.9) !important;
    }
    
    /* Glass effect for segment boxes */
    .segment-box {
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        padding: 1.5rem;
        margin: 0.8rem 0;
        color: #FFFFFF;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.08);
    }
    
    /* Text styling */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF !important;
    }
    
    p, label, span, div {
        color: #FFFFFF !important;
    }
    
    [data-testid="stMarkdownContainer"] {
        color: #FFFFFF !important;
    }
    
    [data-testid="stMarkdownContainer"] p {
        color: #FFFFFF !important;
    }
    
    /* Tab styling */
    [data-testid="stTabs"] button {
        color: #FFFFFF !important;
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }
    
    [data-testid="stTabs"] button:hover {
        background: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Info/Warning/Error message boxes */
    .stAlert {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

st.title("🎙️ Live Meeting Summarizer")
st.markdown("---")

# Session State Initialization
if 'segments' not in st.session_state:
    st.session_state.segments = []
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'processing_done' not in st.session_state:
    st.session_state.processing_done = False

# Sidebar
st.sidebar.title("Settings")
model_choice = st.sidebar.selectbox("Select STT Model", ["Whisper (High Accuracy)", "Vosk (Fast/Local)"])
use_diarization = st.sidebar.checkbox("Enable Speaker Diarization", value=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload & Process", "📝 Analysis & Summary", "📊 Benchmarks", "💾 Export"])

with tab1:
    st.header("Upload Meeting Recording")
    audio_file = st.file_uploader("Upload meeting recording (WAV, MP3, M4A)", type=["wav", "mp3", "m4a"])

    if audio_file is not None:
        st.audio(audio_file)

        if st.button("Start Processing"):
            import tempfile
            import os

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tmp_file.write(audio_file.getbuffer())
                audio_path = tmp_file.name

            try:
                with st.status("Processing Audio...", expanded=True) as status:

                    st.write("📌 Step 1: Initializing models...")
                    try:
                        stt = STTModel(model_name="whisper" if "Whisper" in model_choice else "vosk")
                        diarizer = Diarizer()
                        summarizer = Summarizer()
                        st.write("✓ Models initialized")
                    except Exception as e:
                        st.error(f"Failed to initialize models: {str(e)}")
                        raise

                    st.write("🎙️ Step 2: Transcribing audio (1-3 minutes for longer files)...")
                    try:
                        full_transcript = stt.transcribe(audio_path)
                        if "ERROR" in full_transcript or "failed" in full_transcript.lower():
                            st.error(f"Transcription error: {full_transcript}")
                            raise Exception(full_transcript)
                        st.write(f"✓ Transcription complete ({len(full_transcript)} characters)")
                    except Exception as e:
                        st.error(f"Transcription failed: {str(e)}")
                        raise

                    st.write("👥 Step 3: Extracting speaker segments...")
                    try:
                        st.session_state.segments = diarizer.get_segments(audio_path)
                        if st.session_state.segments:
                            st.session_state.segments[0]["text"] = full_transcript
                        st.write(f"✓ Extracted {len(st.session_state.segments)} segment(s)")
                    except Exception as e:
                        st.error(f"Diarization failed: {str(e)}")
                        raise

                    st.write("✨ Step 4: Generating summary...")
                    try:
                        st.session_state.summary = summarizer.summarize(full_transcript)
                        st.write(f"✓ Summary generated ({len(st.session_state.summary)} characters)")
                    except Exception as e:
                        st.error(f"Summarization failed: {str(e)}")
                        raise

                    st.session_state.processing_done = True
                    status.update(label="✅ Processing Complete!", state="complete", expanded=False)
                    st.success("Successfully processed meeting!")

            except Exception as e:
                st.error(f"❌ Processing Error: {str(e)}")
                import traceback
                st.write("**Traceback:**")
                st.code(traceback.format_exc())
            finally:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
    else:
        st.info("📤 Upload an audio file (WAV, MP3, or M4A) to get started!")

with tab2:
    if st.session_state.processing_done:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Transcript")
            for seg in st.session_state.segments:
                st.markdown(f"""
                <div class="segment-box">
                    <strong>{seg['speaker']}</strong> ({seg['start']}s – {seg['end']}s)<br>
                    {seg['text']}
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.subheader("Summary")
            st.info(st.session_state.summary)

            st.subheader("Statistics")
            total_words = sum(len(s['text'].split()) for s in st.session_state.segments)
            st.metric("Total Words", total_words)
            st.metric("Speakers", len(set(s['speaker'] for s in st.session_state.segments)))
    else:
        st.warning("⚠️ Process an audio file first in the 'Upload & Process' tab.")

with tab3:
    st.header("Model Benchmarks")
    benchmarks = get_benchmark_report()

    df = pd.DataFrame(benchmarks).T
    st.dataframe(df, use_container_width=True)

    st.markdown("**WER**: Word Error Rate (lower is better)")

with tab4:
    if st.session_state.processing_done:
        st.header("Export Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            json_data = export_as_json({
                "segments": st.session_state.segments,
                "summary": st.session_state.summary
            })
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name="meeting_transcript.json",
                mime="application/json"
            )

        with col2:
            md_data = export_as_markdown(st.session_state.segments, st.session_state.summary)
            st.download_button(
                label="📥 Download Markdown",
                data=md_data,
                file_name="meeting_transcript.md",
                mime="text/markdown"
            )

        with col3:
            csv_data = export_as_csv(st.session_state.segments)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="meeting_transcript.csv",
                mime="text/csv"
            )
    else:
        st.warning("⚠️ Process an audio file first to enable export.")
