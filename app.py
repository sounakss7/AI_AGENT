import streamlit as st
from PIL import Image
from io import BytesIO
from PyPDF2 import PdfReader
import fitz
import pytesseract
from datetime import datetime
import re
import time
import pandas as pd
import random  # <-- NEW: Import for AI Tip of the Day
from urllib.parse import quote_plus  # <-- NEW: Import for Google Search URL encoding

# Import the agent logic
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

# ===============================================
# Initialize Session State for Chat and Metrics
# ===============================================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "metrics" not in st.session_state:
    st.session_state.metrics = {
        "total_requests": 0,
        "tool_usage": {"Comparison": 0, "Image Gen": 0, "Web Search": 0, "File Analysis": 0},
        "total_latency": 0.0,
        "average_latency": 0.0,
        "accuracy_feedback": {"👍": 0, "👎": 0},
        "last_query_details": {}
    }

# =====================
# Main Application UI
# =====================
st.title("🧠 AI Agent Workshop")
st.write("I can search the web, create images, analyze documents, and more!")

# =======================================================
# Sidebar with All Features
# =======================================================
with st.sidebar:
    ### NEW: GOOGLE SEARCH WIDGET ###
    st.header("🔍 Google Search")
    search_query = st.text_input("Search the web directly...", key="google_search")
    if st.button("Search"):
        if search_query:
            encoded_query = quote_plus(search_query)
            search_url = f"https://www.google.com/search?q={encoded_query}"
            st.markdown(f'<a href="{search_url}" target="_blank">Open Google search results for "{search_query}"</a>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a search query.")

    st.header("📂 File Analysis")
    uploaded_file = st.file_uploader("Upload a file to ask questions about it", type=["pdf", "txt", "py", "js", "html", "css"])
    
    st.header("🧭 Utilities")
    if st.button("Clear Chat History & Reset Metrics"):
        st.session_state.messages = []
        st.session_state.metrics = {
            "total_requests": 0, "tool_usage": {"Comparison": 0, "Image Gen": 0, "Web Search": 0, "File Analysis": 0},
            "total_latency": 0.0, "average_latency": 0.0, "accuracy_feedback": {"👍": 0, "👎": 0}, "last_query_details": {}
        }
        st.rerun()

    ### NEW: AI TIP OF THE DAY ###
    st.markdown("### 💡 AI Tip of the Day")
    ai_tips = [
        "Ask the agent about current events to see the Web Search tool in action!",
        "Try asking the agent to 'draw a picture of...' to test its image generation.",
        "Upload a Python file and ask the agent to 'give this code a score out of 10' to test file analysis.",
        "Complex questions like 'compare Python and JavaScript' will trigger the detailed Comparison tool.",
    ]
    st.info(random.choice(ai_tips))
    
    ### NEW: LIVE CLOCK ###
    st.markdown("### 🕒 Live Server Time")
    st.info(datetime.now().strftime("%d %B %Y, %I:%M:%S %p"))


    st.header("📊 Data & Insights")
    metrics = st.session_state.metrics
    # (The rest of the dashboard code is unchanged)
    col1, col2 = st.columns(2)
    col1.metric("Total Requests", metrics["total_requests"])
    col2.metric("Avg. Latency", f"{metrics['average_latency']:.2f} s")
    st.subheader("Tool Usage")
    if metrics["total_requests"] > 0:
        tool_df = pd.DataFrame(list(metrics["tool_usage"].items()), columns=['Tool', 'Count'])
        st.bar_chart(tool_df.set_index('Tool'))
    else:
        st.info("No queries yet.")
    st.subheader("User Feedback (Accuracy)")
    if (metrics["accuracy_feedback"]["👍"] + metrics["accuracy_feedback"]["👎"]) > 0:
        feedback_df = pd.DataFrame(list(metrics["accuracy_feedback"].items()), columns=['Feedback', 'Count'])
        st.bar_chart(feedback_df.set_index('Feedback'))
    else:
        st.info("No feedback yet.")
    with st.expander("🕵️ See Last Query Details"):
        st.json(metrics["last_query_details"])

# (The main chat history and input logic below remains the same as the previous version)
# ... The rest of your app.py file ...
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "text" in message:
            st.markdown(message["text"])
        if "image_bytes" in message:
            img = Image.open(BytesIO(message["image_bytes"]))
            st.image(img, caption=message.get("caption"))

if prompt := st.chat_input("Ask about the latest news, create an image, or query a file..."):
    st.session_state.messages.append({"role": "user", "text": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Agent is working..."):
            start_time = time.time()
            tool_used_key = ""
            
            if uploaded_file:
                tool_used_key = "File Analysis"
                file_bytes = uploaded_file.read()
                file_text = ""
                # Your file reading logic here...
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
                
                response_stream = file_analysis_tool(prompt, file_text, google_api_key)
                full_response = st.write_stream(response_stream)
                st.session_state.messages.append({"role": "assistant", "text": full_response})

            else:
                agent = build_agent(google_api_key, groq_api_key, pollinations_token, tavily_api_key)
                result = agent.invoke({"query": prompt})
                final_response = result.get("final_response", {})
                
                route = result.get("route", "comparison_chat")
                if route == "comparison_chat": tool_used_key = "Comparison"
                elif route == "image_generator": tool_used_key = "Image Gen"
                elif route == "web_search": tool_used_key = "Web Search"

                if isinstance(final_response, str):
                    st.session_state.messages.append({"role": "assistant", "text": final_response})
                elif isinstance(final_response, dict) and "image" in final_response:
                    img_data = final_response["image"]
                    buf = BytesIO()
                    img_data.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.session_state.messages.append({
                        "role": "assistant", "image_bytes": byte_im,
                        "caption": final_response.get("caption", prompt)
                    })
                else:
                    error_message = final_response.get("error", "Sorry, something went wrong.")
                    st.session_state.messages.append({"role": "assistant", "text": f"Error: {error_message}"})
            
            end_time = time.time()
            latency = end_time - start_time
            
            metrics = st.session_state.metrics
            metrics["total_requests"] += 1
            if tool_used_key:
                metrics["tool_usage"][tool_used_key] += 1
            metrics["total_latency"] += latency
            metrics["average_latency"] = metrics["total_latency"] / metrics["total_requests"]
            metrics["last_query_details"] = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "tool_used": tool_used_key,
                "latency_seconds": round(latency, 2)
            }
    
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    message_id = len(st.session_state.messages) - 1

    if f"feedback_{message_id}" not in st.session_state:
        feedback_cols = st.columns(10)
        if feedback_cols[0].button("👍", key=f"good_{message_id}"):
            st.session_state.metrics["accuracy_feedback"]["👍"] += 1
            st.session_state[f"feedback_{message_id}"] = "given"
            st.toast("Thanks for your feedback!")
            time.sleep(1)
            st.rerun()

        if feedback_cols[1].button("👎", key=f"bad_{message_id}"):
            st.session_state.metrics["accuracy_feedback"]["👎"] += 1
            st.session_state[f"feedback_{message_id}"] = "given"
            st.toast("Thanks for your feedback!")
            time.sleep(1)
            st.rerun()