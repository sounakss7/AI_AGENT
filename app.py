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
import json

# --- UPDATE 1: Import from your agent_app.py file ---
from agent_app import build_agent, file_analysis_tool

# --- UPDATE 2: create_copy_button modified for better compatibility ---
def create_copy_button(text_to_copy: str, button_key: str):
    """
    Creates a button in Streamlit that copies the given text to the clipboard
    using document.execCommand('copy') for better iFrame compatibility.
    """
    button_id = f"copy_btn_{button_key}"
    text_id = f"text_{button_key}"

    # The HTML part: a hidden element to hold the text and a button
    # Using <textarea> is crucial for .select() to work
    html_code = f"""
        <textarea id="{text_id}" style="position: absolute; left: -9999px;">{text_to_copy}</textarea>
        <button id="{button_id}">Copy Text</button>
    """

    # The JavaScript part: updated to use execCommand
    js_code = f"""
        <script>
            document.getElementById("{button_id}").addEventListener("click", function() {{
                var textElem = document.getElementById("{text_id}");
                
                // --- Fallback for iOS ---
                var isOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
                if (isOS) {{
                    var range = document.createRange();
                    range.selectNodeContents(textElem);
                    var selection = window.getSelection();
                    selection.removeAllRanges();
                    selection.addRange(range);
                    textElem.setSelectionRange(0, 999999);
                }} else {{
                    textElem.select();
                }}

                try {{
                    var successful = document.execCommand('copy');
                    var btn = document.getElementById("{button_id}");
                    var originalText = btn.innerHTML;
                    if (successful) {{
                        btn.innerHTML = 'Copied!';
                    }} else {{
                        btn.innerHTML = 'Error';
                    }}
                    setTimeout(function() {{
                        btn.innerHTML = originalText;
                    }}, 2000);
                }} catch (err) {{
                    console.error('Could not copy text: ', err);
                }}
                
                // Deselect text
                if (window.getSelection) {{
                    window.getSelection().removeAllRanges();
                }} else if (document.selection) {{
                    document.selection.empty();
                }}
            }});
        </script>
    """
    
    st.components.v1.html(html_code + js_code, height=50)

# =====================
# Page Config and Setup
# =====================
st.set_page_config(page_title="🤖 AI Agent Workshop", page_icon="🧠", layout="wide")

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
if "trajectory" not in st.session_state:
    st.session_state.trajectory = []
if "metrics" not in st.session_state:
    # --- UPDATE 3: Added "Code Reviewer" to tool_usage ---
    st.session_state.metrics = {
        "total_requests": 0,
        "tool_usage": {"Comparison": 0, "Image Gen": 0, "Web Search": 0, "File Analysis": 0, "Code Reviewer": 0},
        "total_latency": 0.0,
        "average_latency": 0.0,
        "accuracy_feedback": {"👍": 0, "👎": 0},
        "last_query_details": {}
    }

# =====================
# Main Application UI & Sidebar
# =====================
st.title("🧠 AI Agent Workshop")
st.write("I can search the web, create images, analyze documents, review code, and more!")

with st.sidebar:
    st.header("🔍 Google Search")
    search_query = st.text_input("Search the web directly...", key="google_search")
    if st.button("Search"):
        if search_query:
            encoded_query = quote_plus(search_query)
            search_url = f"https.google.com/search?q={encoded_query}"
            st.markdown(f'<a href="{search_url}" target="_blank">Open Google search results</a>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a search query.")

    st.header("📂 File Analysis")
    uploaded_file = st.file_uploader("Upload a file to ask questions about it", type=["pdf", "txt", "py", "js", "html", "css"])
    
    st.header("🧭 Utilities")
    if st.button("Clear Chat History & Reset Metrics"):
        st.session_state.messages = []
        st.session_state.trajectory = []
        # --- UPDATE 3: Added "Code Reviewer" to reset logic ---
        st.session_state.metrics = {
            "total_requests": 0,
            "tool_usage": {"Comparison": 0, "Image Gen": 0, "Web Search": 0, "File Analysis": 0, "Code Reviewer": 0},
            "total_latency": 0.0, "average_latency": 0.0, "accuracy_feedback": {"👍": 0, "👎": 0}, "last_query_details": {}
        }
        st.rerun()

    st.markdown("### 💡 AI Tip of the Day")
    st.info(random.choice([
        "The Comparison tool uses both Gemini and Groq for a robust answer.",
        "Ask about current events to see the Web Search tool in action!",
        "Upload a Python file and ask for a score to test File Analysis.",
        "Paste a code snippet and ask 'review this code' to test the Code Reviewer."
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

# ===================================================================
# Main Chat Display Logic with Custom Copy Button
# ===================================================================
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if "text" in message:
            st.markdown(message["text"])
            if message["role"] == "assistant":
                create_copy_button(message["text"], button_key=f"text_copy_{i}")

        if "image_bytes" in message:
            img = Image.open(BytesIO(message["image_bytes"]))
            st.image(img, caption=message.get("caption"))
            
            st.download_button(
                label="⬇️ Download Image",
                data=message["image_bytes"],
                file_name=f"generated_image_{i}.png",
                mime="image/png",
                key=f"download_btn_{i}"
            )
# =================================================================================
# Agent Debugger Logic
# =================================================================================

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
                # --- UPDATE 4: Added "code_reviewer" to tool tracking ---
                if event['name'] == "comparison_chat": tool_used = "Comparison"
                elif event['name'] == "image_generator": tool_used = "Image Gen"
                elif event['name'] == "web_search": tool_used = "Web Search"
                elif event['name'] == "code_reviewer": tool_used = "Code Reviewer"

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

def pretty_print_dict(d):
    def safe_converter(o):
        if isinstance(o, (Image.Image, bytes)):
            return f"<{type(o).__name__} object>"
        return str(o)
    
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
                tool_used_key = "File Analysis"
                file_bytes = uploaded_file.read()
                file_text = ""
                if "pdf" in uploaded_file.type:
                    try:
                        reader = PdfReader(BytesIO(file_bytes))
                        for page in reader.pages: file_text += page.extract_text() or ""
                        if not file_text.strip():
                            st.info("No text layer found, performing OCR...")
                            doc = fitz.open(stream=file_bytes, filetype="pdf")
                            for page in doc:
                                pix = page.get_pixmap()
                                img = Image.open(BytesIO(pix.tobytes("png")))
                                file_text += pytesseract.image_to_string(img)
                    except Exception as e:
                        st.error(f"Error reading PDF: {e}")
                        file_text = ""
                else:
                    file_text = file_bytes.decode("utf-8", errors="ignore")
                
                response_stream = file_analysis_tool(prompt, file_text, google_api_key)
                full_response = st.write_stream(response_stream)
                st.session_state.messages.append({"role": "assistant", "text": full_response})
            
            else:
                agent = build_agent(google_api_key, groq_api_key, pollinations_token, tavily_api_key)
                
                final_response, trace_steps, tool_used_key = asyncio.run(run_agent_and_capture_trajectory(agent, prompt))

                st.session_state.trajectory.append({"prompt": prompt, "steps": trace_steps})

                # --- UPDATE 5: New display logic for Code Reviewer ---
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
                
                elif isinstance(final_response, dict) and "score" in final_response:
                    # This is a Code Review response
                    score = final_response.get('score', 0)
                    feedback = final_response.get('feedback', 'No feedback provided.')
                    errors = final_response.get('errors', [])
                    corrected_code = final_response.get('corrected_code', '# No code returned.')

                    st.metric(label="Code Review Score", value=f"{score} / 100")
                    
                    with st.expander("Detailed Feedback & Errors", expanded=True):
                        st.markdown(feedback)
                        if errors:
                            st.error("Errors Found:")
                            for error in errors:
                                st.markdown(f"- `{error}`")
                    
                    st.markdown("**Corrected Code:**")
                    st.code(corrected_code, language="python")
                    
                    # Create a text-based version for chat history
                    text_response = (
                        f"**Code Review Score: {score}/100**\n\n"
                        f"**Feedback:**\n{feedback}\n\n"
                        f"**Errors:**\n- {'- '.join(errors) if errors else 'None'}\n\n"
                        f"**Corrected Code:**\n```python\n{corrected_code}\n```"
                    )
                    st.session_state.messages.append({"role": "assistant", "text": text_response})

                else:
                    error_message = final_response.get("error", "Sorry, something went wrong.")
                    st.markdown(f"Error: {error_message}")
                    st.session_state.messages.append({"role": "assistant", "text": f"Error: {error_message}"})
            
            # --- Metrics Recording (This logic is dynamic and needs no change) ---
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

# --- Agent Trajectory / Debug View ---
if st.session_state.trajectory:
    with st.expander("🕵️ Agent Trajectory / Debug View", expanded=False):
        for run in reversed(st.session_state.trajectory):
            st.markdown(f"#### Prompt: *'{run.get('prompt', 'N/A')}'*")
            
            steps = run.get('steps', [])
            
            for step in steps:
                st.markdown(f"##### 🎬 Step: `{step.get('name', 'Unknown Step')}`")
                
                with st.container(border=True):
                    st.markdown("**Input:**")
                    st.markdown(pretty_print_dict(step.get('input', {})), unsafe_allow_html=True)
                
                with st.container(border=True):
                    st.markdown("**Output:**")
                    st.markdown(pretty_print_dict(step.get('output', {})), unsafe_allow_html=True)
                
            st.markdown("---")


# --- Feedback buttons logic ---
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