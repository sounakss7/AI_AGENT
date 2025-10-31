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

# --- NEW: Import gTTS ---
from gtts import gTTS

# Import the agent logic
from agent import build_agent, file_analysis_tool

# --- NEW: A helper function to generate audio in memory ---
def generate_audio_from_text(text: str) -> bytes | None:
    """Generates MP3 audio from text and returns it as bytes."""
    # Clean text to remove markdown for better speech
    text = re.sub(r'(\*\*|##|###|####|`|```)', '', text)
    if not text or not text.strip():
        return None
    try:
        audio_fp = BytesIO()
        tts = gTTS(text=text, lang='en')
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp.getvalue()
    except Exception as e:
        print(f"Error generating TTS audio: {e}")
        return None

# --- NEW: A custom function to create a copy-to-clipboard button ---
def create_copy_button(text_to_copy: str, button_key: str):
    """
    Creates a button in Streamlit that copies the given text to the clipboard.
    """
    # Unique IDs for HTML elements
    button_id = f"copy_btn_{button_key}"
    text_id = f"text_{button_key}"

    # The HTML part: a hidden element to hold the text and a button
    html_code = f"""
        <textarea id="{text_id}" style="position: absolute; left: -9999px;">{text_to_copy}</textarea>
        <button id="{button_id}">Copy Text</button>
    """

    # The JavaScript part: finds the elements and adds the copy logic
    js_code = f"""
        <script>
            document.getElementById("{button_id}").addEventListener("click", function() {{
                var text = document.getElementById("{text_id}").value;
                navigator.clipboard.writeText(text).then(function() {{
                    var btn = document.getElementById("{button_id}");
                    var originalText = btn.innerHTML;
                    btn.innerHTML = 'Copied!';
                    setTimeout(function() {{
                        btn.innerHTML = originalText;
                    }}, 2000);
                }}, function(err) {{
                    console.error('Could not copy text: ', err);
                }});
            }});
        </script>
    """
    
    # Combine and render using st.components.v1.html
    st.components.v1.html(html_code + js_code, height=50)

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
            search_url = f"[https://www.google.com/search?q=](https://www.google.com/search?q=){encoded_query}"
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


# ===================================================================
# --- MODIFIED: Main Chat Display Logic with On-Demand Audio Button ---
# ===================================================================
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # --- Display Text Response ---
        if "text" in message:
            st.markdown(message["text"])
            
            # --- NEW: On-Demand Audio Logic ---
            if message["role"] == "assistant":
                audio_bytes = message.get("audio_bytes")
                
                if audio_bytes:
                    # If audio is generated, show the player
                    st.audio(audio_bytes, format="audio/mp3")
                else:
                    # If no audio, show the "Listen" button
                    if st.button("🎧 Listen", key=f"listen_btn_{i}"):
                        with st.spinner("Generating audio..."):
                            # Generate audio
                            new_audio_bytes = generate_audio_from_text(message["text"])
                            if new_audio_bytes:
                                # Update the message in session state
                                st.session_state.messages[i]["audio_bytes"] = new_audio_bytes
                                # Rerun to display the audio player
                                st.rerun()
                            else:
                                st.error("Could not generate audio.")

                # Add the copy button
                create_copy_button(message["text"], button_key=f"text_copy_{i}")

        # --- Display Image with a Download Button ---
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
# --- AGDebugger Logic (Unchanged) ---
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

def pretty_print_dict(d):
    def safe_converter(o):
        if isinstance(o, (Image.Image, bytes)):
            return f"<{type(o).__name__} object>"
        return str(o)
    
    if not isinstance(d, dict):
        return f"```\n{str(d)}\n```"
        
    return "```json\n" + json.dumps(d, indent=2, default=safe_converter) + "\n```"


# =================================================================================
# --- MODIFIED: Main Chat Input Logic (REMOVED automatic audio generation) ---
# =================================================================================
if prompt := st.chat_input("Ask about the latest news, create an image, or query a file..."):
    st.session_state.messages.append({"role": "user", "text": prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("Agent is working..."):
            start_time = time.time()
            tool_used_key = ""

            if uploaded_file:
                # --- This is the File Analysis Path ---
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
                
                # --- MODIFIED: Store None for audio, it will be generated on-demand ---
                st.session_state.messages.append({"role": "assistant", "text": full_response, "audio_bytes": None})
            
            else:
                # --- This is the Agent Path ---
                agent = build_agent(google_api_key, groq_api_key, pollinations_token, tavily_api_key)
                
                final_response, trace_steps, tool_used_key = asyncio.run(run_agent_and_capture_trajectory(agent, prompt))
                st.session_state.trajectory.append({"prompt": prompt, "steps": trace_steps})

                if isinstance(final_response, str):
                    st.markdown(final_response)
                    # --- MODIFIED: Store None for audio ---
                    st.session_state.messages.append({"role": "assistant", "text": final_response, "audio_bytes": None})
                
                elif isinstance(final_response, dict) and "image" in final_response:
                    img_data = final_response["image"]
                    buf = BytesIO()
                    img_data.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    st.image(byte_im, caption=final_response.get("caption", prompt))
                    st.session_state.messages.append({
                        "role": "assistant", "image_bytes": byte_im, "text": f"Image generated for: *{prompt}*",
                        "caption": final_response.get("caption", prompt)
                        # No audio for image-only responses
                    })
                
                else:
                    error_message = final_response.get("error", "Sorry, something went wrong.")
                    st.markdown(f"Error: {error_message}")
                    # --- MODIFIED: Store None for audio ---
                    st.session_state.messages.append({"role": "assistant", "text": f"Error: {error_message}", "audio_bytes": None})
            
            # --- Metrics Recording (logic is unchanged) ---
            # This timer now stops *before* any audio is generated, giving an accurate latency.
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

# --- MODIFIED: Debug View (Unchanged from your version) ---
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
            st.rerun() . 
