"""Streamlit UI for the Humaniser Document Transformation Platform."""
import asyncio

import streamlit as st

from src.config import STYLE_PROFILES
from src.main import transform_document

# --- Page Configuration ---
st.set_page_config(
    page_title="Humaniser AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Aesthetics ---
st.markdown("""
<style>
    /* Global background and typography */
    .stApp {
        background-color: #0E1117;
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.8rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #A0AEC0;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    /* Text Areas */
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #2D3748 !important;
        background-color: #1A202C !important;
        color: #E2E8F0 !important;
        padding: 1rem;
        font-size: 1rem;
        line-height: 1.5;
    }
    .stTextArea textarea:focus {
        border-color: #4ECDC4 !important;
        box-shadow: 0 0 0 1px #4ECDC4 !important;
    }
    /* Buttons */
    .stButton button {
        width: 100%;
        border-radius: 8px;
        background: linear-gradient(90deg, #4ECDC4 0%, #556270 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.6rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(78, 205, 196, 0.4);
    }
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #1A202C;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def run_async(coroutine):
    """Helper to execute async pipelines in Streamlit's sync environment."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)


# --- Sidebar Navigation ---
with st.sidebar:
    st.title("Humaniser AI ✨")
    st.markdown("Compiler-driven Document Style Transformation")
    st.divider()
    
    api_key = st.text_input(
        "Gemini API Key", 
        type="password", 
        help="Requires gemini-1.5-flash access."
    )
    
    profile = st.selectbox(
        "Style Profile", 
        options=list(STYLE_PROFILES.keys()),
        format_func=lambda x: x.title()
    )
    
    source_format = st.radio("Document Format", options=["markdown", "latex"], index=0)
    
    st.divider()
    st.markdown("### Settings")
    max_retries = st.slider("Validation Max Retries", min_value=1, max_value=5, value=3)


# --- Main Dashboard ---
st.markdown('<p class="main-header">Transform Your Document</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Rewrite your text using rigorous compiler validation constraints.</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Source Document")
    source_text = st.text_area("Input your Markdown or LaTeX text here...", height=400, label_visibility="collapsed")
    
    if st.button("Humanize ✨"):
        if not api_key:
            st.error("Please enter a valid Gemini API Key in the sidebar.")
        elif not source_text.strip():
            st.warning("Please enter some text to transform.")
        else:
            with st.spinner("Compiling Transformation Pipeline..."):
                try:
                    final_text, ir = run_async(
                        transform_document(
                            source_text=source_text,
                            source_format=source_format,
                            target_profile=profile,
                            api_key=api_key,
                            max_retries=max_retries
                        )
                    )
                    st.session_state['final_text'] = final_text
                    st.session_state['ir'] = ir
                except Exception as e:
                    st.error(f"Pipeline Execution Failed: {e}")

with col2:
    st.markdown("### Output Document")
    if 'final_text' in st.session_state:
        st.text_area("Transformed text", value=st.session_state['final_text'], height=400, disabled=True, label_visibility="collapsed")
    else:
        st.text_area("Transformed text", value="Your rewritten text will appear here...", height=400, disabled=True, label_visibility="collapsed")

# --- Observability Metrics ---
if 'ir' in st.session_state:
    st.divider()
    st.markdown("### 🔍 Compiler Observability Metrics")
    st.markdown("Dive into the IR nodes and see exactly how the LLM was constrained and validated.")
    
    ir = st.session_state['ir']
    
    for i, node in enumerate(ir.nodes):
        if not node.editable:
            continue
            
        with st.expander(f"Paragraph {node.node_index}"):
            st.markdown(f"**Original:** {node.original_text}")
            st.markdown(f"**Rewritten:** {node.rewritten_text}")
            
            mc1, mc2, mc3 = st.columns(3)
            
            with mc1:
                st.markdown("**Feature Vector**")
                if node.feature_vector:
                    st.json(node.feature_vector.model_dump())
                    
            with mc2:
                st.markdown("**Transformation Plan**")
                if node.transformation_plan:
                    st.write("Selected Instructions:")
                    for instruction in node.transformation_plan.selected_instructions:
                        st.markdown(f"- {instruction}")
                        
            with mc3:
                st.markdown("**Validation Report**")
                if node.validation_report:
                    val = node.validation_report
                    if val.overall_valid:
                        st.success("Validation Passed")
                    else:
                        st.error("Validation Failed (Fallback triggered)")
                    st.json(val.model_dump())
