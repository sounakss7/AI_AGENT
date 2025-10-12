import streamlit as st
from PIL import Image
from io import BytesIO
from PyPDF2 import PdfReader
import fitz
import pytesseract
from datetime import datetime
import re # Import re for filename sanitization

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
st.write("I can search the web, create images, analyze documents, and more!")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "text" in message and message["text"]:
            st.markdown(message["text"])
        if "image" in message and message["image"]:
            st.image(message["image"], caption=message.get("caption"))
            # Note: The download button won't persist in the history view for simplicity.

# --- Sidebar for File Upload and Utilities ---
with st.sidebar:
    st.header("📂 File Analysis")
    uploaded_file = st.file_uploader("Upload a file to ask questions about it", type=["pdf", "txt", "py", "js", "html", "css"])
    st.header("🧭 Utilities")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("### 💡 AI Tip of the Day")
    st.info("You can now download generated images and copy text responses directly from the chat!")
    st.markdown("### 🕒 Current Time")
    st.write(datetime.now().strftime("%d %B %Y, %I:%M:%S %p"))

# --- Main Chat Input Logic ---
if prompt := st.chat_input("Ask about the latest news, create an image, or query a file..."):
    st.session_state.messages.append({"role": "user", "text": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # --- AGENTIC WORKFLOW STARTS HERE ---
        if uploaded_file:
            with st.spinner(f"Reading and analyzing `{uploaded_file.name}`..."):
                # (File analysis logic is unchanged)
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
                        # We store the markdown version for cleaner history display
                        st.session_state.messages.append({"role": "assistant", "text": full_response})
                except Exception as e:
                    st.error(f"Error processing file: {e}")
                    st.session_state.messages.append({"role": "assistant", "text": f"Error: {e}"})
        else:
            with st.spinner("Agent is deciding which tool to use..."):
                agent = build_agent(google_api_key, groq_api_key, pollinations_token, tavily_api_key)
                result = agent.invoke({"query": prompt})
                final_response = result.get("final_response", {})

                # --- NEW FEATURE: Handle Text Response with Copy Button ---
                if isinstance(final_response, str):
                    # Use st.code to display the text, which includes a copy button
                    st.code(final_response, language=None)
                    st.session_state.messages.append({"role": "assistant", "text": final_response})

                # --- NEW FEATURE: Handle Image Response with Download Button ---
                if isinstance(final_response, dict) and "image" in final_response:
                    img_data = final_response["image"]
                    caption = final_response.get("caption")
                    st.image(img_data, caption=caption)
                    # Convert PIL Image to bytes for downloading
                    buf = BytesIO()
                    img_data.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    # Create a safe filename from the prompt
                    safe_prompt = re.sub(r'[^a-zA-Z0-9\s]', '', prompt).strip().replace(' ', '_')
                    file_name = f"{safe_prompt[:50]}.png"
                    st.download_button(
                        label="📥 Download Image",
                        data=byte_im,
                        file_name=file_name,
                        mime="image/png"
                    )
                    st.session_state.messages.append({"role": "assistant", "image": img_data, "caption": caption})
                    
                else: # Handle errors
                    error_message = final_response.get("error", "Sorry, something went wrong.")
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "text": f"Error: {error_message}"})
    
    # Rerun to clear the file uploader widget and sync state
    st.rerun()