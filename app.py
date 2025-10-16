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
import random
from urllib.parse import quote_plus
import asyncio
import json # --- NEW IMPORT for pretty-printing dictionaries ---

# Import the agent logic (no changes needed in agent.py)
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
# --- NEW: Add a key to store the agent's trace ---
if "trajectory" not in st.session_state:
    st.session_state.trajectory = []
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
# Main Application UI & Sidebar (Unchanged)
# =====================
st.title("🧠 AI Agent Workshop")
st.write("I can search the web, create images, analyze documents, and more!")

with st.sidebar:
    st.header("🔍 Google Search")
    search_query = st.text_input("Search the web directly...", key="google_search")
    if st.button("Search"):
        if search_query:
            encoded_query = quote_plus(search_query)
            search_url = f"https://www.google.com/search?q={encoded_query}"
            st.markdown(f'<a href="{search_url}" target="_blank">Open Google search results</a>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a search query.")

    st.header("📂 File Analysis")
    uploaded_file = st.file_uploader("Upload a file to ask questions about it", type=["pdf", "txt", "py", "js", "html", "css"])
    
    st.header("🧭 Utilities")
    if st.button("Clear Chat History & Reset Metrics"):
        st.session_state.messages = []
        st.session_state.trajectory = [] # --- ADDED: Clear trajectory on reset ---
        st.session_state.metrics = {
            "total_requests": 0, "tool_usage": {"Comparison": 0, "Image Gen": 0, "Web Search": 0, "File Analysis": 0},
            "total_latency": 0.0, "average_latency": 0.0, "accuracy_feedback": {"👍": 0, "👎": 0}, "last_query_details": {}
        }
        st.rerun()

    st.markdown("### 💡 AI Tip of the Day")
    st.info(random.choice([
        "The Comparison tool uses both Gemini 1.5 Flash and Llama 3.1 8B for a robust answer.",
        "Ask about current events to see the Web Search tool in action!",
        "Upload a Python file and ask for a score to test File Analysis."
    ]))
    
    st.markdown("### 🕒 Live Server Time")
    st.info(datetime.now().strftime("%d %B %Y, %I:%M:%S %p"))

    st.header("🤖 Model Benchmarks")
    with st.expander("See Industry Benchmark Scores"):
        st.markdown("**Note:** These are public scores for the models used in this agent's Comparison tool.")
        benchmark_data = {
            "MMLU": {"Llama-3.1-8B Instant": 74.2, "Gemini-2.5 Flash": 82.1, "help": "Measures general knowledge and problem-solving."},
            "HumanEval": {"Llama-3.1-8B Instant": 70.3, "Gemini-2.5 Flash": 83.5, "help": "Measures Python code generation ability."},
            "GSM8K": {"Llama-3.1-8B Instant": 84.2, "Gemini-2.5 Flash": 91.1, "help": "Measures grade-school math reasoning."}
        }
        for bench, scores in benchmark_data.items():
            llama_score, flash_score = scores["Llama-3.1-8B Instant"], scores["Gemini-2.5 Flash"]
            st.markdown(f"**{bench}**")
            c1, c2 = st.columns(2)
            c1.metric("Llama-3.1-8B Instant (Groq)", f"{llama_score}%", delta=f"{round(llama_score - flash_score, 1)}%", help=scores["help"])
            c2.metric("Gemini-2.5 Flash", f"{flash_score}%", delta=f"{round(flash_score - llama_score, 1)}%", help=scores["help"])

    st.header("📊 Live Agent Performance")
    metrics = st.session_state.metrics
    col1, col2 = st.columns(2)
    col1.metric("Total Requests", metrics["total_requests"])
    col2.metric("Avg. Latency", f"{metrics['average_latency']:.2f} s")
    st.subheader("📈 Live App Accuracy (User Feedback)")
    total_feedback = metrics["accuracy_feedback"]["👍"] + metrics["accuracy_feedback"]["👎"]
    if total_feedback > 0:
        positive_rate = (metrics["accuracy_feedback"]["👍"] / total_feedback) * 100
        st.metric("Positive Feedback Rate", f"{positive_rate:.1f}%", help="Based on user 👍/👎 clicks.")
    else:
        st.info("No feedback yet to calculate accuracy.")
    st.subheader("🛠️ Tool Usage")
    if metrics["total_requests"] > 0:
        tool_df = pd.DataFrame(list(metrics["tool_usage"].items()), columns=['Tool', 'Count'])
        st.bar_chart(tool_df.set_index('Tool'))
    with st.expander("🕵️ See Last Query Details"):
        st.json(metrics["last_query_details"])

# ===============================================
# Main Chat Display Logic (Unchanged)
# ===============================================
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "text" in message:
            st.markdown(message["text"])
        if "image_bytes" in message:
            img = Image.open(BytesIO(message["image_bytes"]))
            st.image(img, caption=message.get("caption"))

# =================================================================================
# --- MODIFIED: AGDebugger Logic with More Detailed Tracing ---
# =================================================================================

# --- MODIFIED: This function now captures much more detail ---
async def run_agent_and_capture_trajectory(agent, prompt):
    """
    Runs the agent using astream_events and captures a detailed trace of its execution,
    including inputs and outputs for each step.
    """
    trace_steps = []
    current_step = {}
    final_response = None
    tool_used = "N/A"

    async for event in agent.astream_events({"query": prompt}, version="v1"):
        kind = event["event"]
        
        if kind == "on_chain_start":
            if event["name"] != "LangGraph":
                current_step = {
                    "name": event["name"],
                    "input": event["data"].get("input"),
                    "output": None
                }
                # Capture the friendly tool name
                if event['name'] == "comparison_chat": tool_used = "Comparison"
                elif event['name'] == "image_generator": tool_used = "Image Gen"
                elif event['name'] == "web_search": tool_used = "Web Search"

        if kind == "on_chain_end":
            if event["name"] != "LangGraph":
                output = event["data"].get("output")
                if current_step.get("name") == event["name"]:
                    current_step["output"] = output
                    trace_steps.append(current_step)
                    current_step = {}
                
                if isinstance(output, dict) and 'final_response' in output:
                    final_response = output['final_response']

    return final_response, trace_steps, tool_used

# --- NEW: Helper function to pretty-print dictionaries, handling non-serializable objects ---
def pretty_print_dict(d):
    def safe_converter(o):
        if isinstance(o, (Image.Image, bytes)):
            return f"<{type(o).__name__} object>"
        return str(o)
    
    # Check if the object is a dict before trying to dump it
    if not isinstance(d, dict):
        return f"```\n{str(d)}\n```"
        
    return "```json\n" + json.dumps(d, indent=2, default=safe_converter) + "\n```"


if prompt := st.chat_input("Ask about the latest news, create an image, or query a file..."):
    st.session_state.messages.append({"role": "user", "text": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Agent is working..."):
            start_time = time.time()
            tool_used_key = ""

            if uploaded_file:
                # File Analysis logic remains separate
                tool_used_key = "File Analysis"
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
                
                response_stream = file_analysis_tool(prompt, file_text, google_api_key)
                full_response = st.write_stream(response_stream)
                st.session_state.messages.append({"role": "assistant", "text": full_response})
            
            else:
                agent = build_agent(google_api_key, groq_api_key, pollinations_token, tavily_api_key)
                
                # Run the async function to get the final answer and the detailed trace
                final_response, trace_steps, tool_used_key = asyncio.run(run_agent_and_capture_trajectory(agent, prompt))

                # Store the structured trace for the debug view
                st.session_state.trajectory.append({"prompt": prompt, "steps": trace_steps})

                # --- Display the final response (logic is unchanged) ---
                if isinstance(final_response, str):
                    st.markdown(final_response)
                    st.session_state.messages.append({"role": "assistant", "text": final_response})
                elif isinstance(final_response, dict) and "image" in final_response:
                    img_data = final_response["image"]
                    buf = BytesIO()
                    img_data.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.image(byte_im, caption=final_response.get("caption", prompt))
                    st.session_state.messages.append({
                        "role": "assistant", "image_bytes": byte_im, "text": f"Image generated for: *{prompt}*",
                        "caption": final_response.get("caption", prompt)
                    })
                else:
                    error_message = final_response.get("error", "Sorry, something went wrong.")
                    st.markdown(f"Error: {error_message}")
                    st.session_state.messages.append({"role": "assistant", "text": f"Error: {error_message}"})
            
            # --- Metrics Recording (logic is unchanged) ---
            end_time = time.time()
            latency = end_time - start_time
            metrics = st.session_state.metrics
            metrics["total_requests"] += 1
            if tool_used_key and tool_used_key in metrics["tool_usage"]:
                metrics["tool_usage"][tool_used_key] += 1
            metrics["total_latency"] += latency
            metrics["average_latency"] = metrics["total_latency"] / metrics["total_requests"]
            metrics["last_query_details"] = {
                "timestamp": datetime.now().isoformat(), "prompt": prompt,
                "tool_used": tool_used_key, "latency_seconds": round(latency, 2)
            }
            
            st.rerun()

# --- MODIFIED: Display the detailed Agent Trajectory / Debug View ---
if st.session_state.trajectory:
    with st.expander("🕵️ Agent Trajectory / Debug View", expanded=False):
        for run in reversed(st.session_state.trajectory):
            st.markdown(f"#### Prompt: *'{run['prompt']}'*")
            for step in run['steps']:
                st.markdown(f"##### 🎬 Step: `{step['name']}`")
                
                # Display Input
                with st.container(border=True):
                    st.markdown("**Input:**")
                    st.markdown(pretty_print_dict(step['input']), unsafe_allow_html=True)
                
                # Display Output
                with st.container(border=True):
                    st.markdown("**Output:**")
                    st.markdown(pretty_print_dict(step['output']), unsafe_allow_html=True)
                
            st.markdown("---")


# --- Feedback buttons logic (Unchanged) ---
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