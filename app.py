import streamlit as st
from PIL import Image
from io import BytesIO
from PyPDF2 import PdfReader
import fitz  # PyMuPDF
import pytesseract
from datetime import datetime

# Import the agent logic from our new file
from agent import build_agent, file_analysis_tool

# =====================
# Page Config and Setup
# =====================
st.set_page_config(page_title="🤖 AI Agent Workshop", page_icon="🧠", layout="wide")

# --- Securely load API keys from Streamlit Secrets ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    pollinations_token = st.secrets["POLLINATIONS_TOKEN"]
    groq_api_key = st.secrets["GROQ_API_KEY"]
    tavily_api_key = st.secrets["TAVILY_API_KEY"] 
except KeyError as e:
    st.error(f"❌ Missing Secret: {e}. Please add it to your Streamlit Secrets.")
    st.stop()

# =====================
# Main Application UI
# =====================
st.title("🧠 AI Agent Workshop")
st.write("I am an AI agent with access to a suite of tools. How can I assist you?")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "text" in message:
            st.markdown(message["text"])
        if "image" in message:
            st.image(message["image"], caption=message.get("caption"))

# --- Sidebar for File Upload and Utilities ---
with st.sidebar:
    st.header("📂 File Analysis")
    uploaded_file = st.file_uploader(
        "Upload a file to ask questions about it",
        type=["pdf", "txt", "py", "js", "html", "css"]
    )
    
    st.header("🧭 Utilities")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("### 💡 AI Tip of the Day")
    st.info("The agent now uses `gemini-1.5-pro-latest` for critical reasoning and `gemini-1.5-flash` for speed!")
    
    st.markdown("### 🕒 Current Time")
    st.write(datetime.now().strftime("%d %B %Y, %I:%M:%S %p"))

# --- Main Chat Input Logic ---
if prompt := st.chat_input("Ask me to analyze text, create an image, or query a file..."):
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # --- AGENTIC WORKFLOW STARTS HERE ---
        if uploaded_file:
            # --- PATH 1: FILE ANALYSIS ---
            with st.spinner(f"Reading and analyzing `{uploaded_file.name}`..."):
                try:
                    file_bytes = uploaded_file.read()
                    file_text = ""
                    if "pdf" in uploaded_file.type:
                        reader = PdfReader(BytesIO(file_bytes))
                        for page in reader.pages: file_text += page.extract_text() or ""
                        if not file_text.strip():
                            st.info("No text layer found, performing OCR...")
                            doc = fitz.open(stream=file_bytes, filetype="pdf")
                            for page in doc:
                                pix = page.get_pixmap()
                                img = Image.open(BytesIO(pix.tobytes("png")))
                                file_text += pytesseract.image_to_string(img)
                    else:
                        file_text = file_bytes.decode("utf-8", errors="ignore")

                    if not file_text.strip():
                        st.error("Could not extract any text from the uploaded file.")
                    else:
                        response_stream = file_analysis_tool(prompt, file_text, google_api_key)
                        full_response = st.write_stream(response_stream)
                        st.session_state.messages.append({"role": "assistant", "text": full_response})
                except Exception as e:
                    st.error(f"Error processing file: {e}")
                    st.session_state.messages.append({"role": "assistant", "text": f"Error: {e}"})
        else:
            # --- PATH 2: GENERAL PURPOSE AGENT (NO FILE) ---
            with st.spinner("Agent is deciding which tool to use..."):
                # Build the agent, passing the keys from st.secrets
                agent = build_agent(google_api_key, groq_api_key, pollinations_token , tavily_api_key)
                result = agent.invoke({"query": prompt})
                final_response = result.get("final_response", {})

                if isinstance(final_response, str):
                    st.markdown(final_response)
                    st.session_state.messages.append({"role": "assistant", "text": final_response})
                elif isinstance(final_response, dict) and "image" in final_response:
                    st.image(final_response["image"], caption=final_response.get("caption"))
                    st.session_state.messages.append({"role": "assistant", "image": final_response["image"], "caption": final_response.get("caption")})
                else: # Handle errors returned by tools
                    error_message = final_response.get("error", "Sorry, something went wrong.")
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "text": f"Error: {error_message}"})
    
    st.rerun()