import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from io import BytesIO
from PyPDF2 import PdfReader
import fitz # PyMuPDF
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

# --- NEW: Imports for chat history conversion ---
from langchain.schema import HumanMessage, AIMessage, BaseMessage

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
    # Note: Using st.markdown for the button to apply streamlit styling
    html_code = f"""
        <textarea id="{text_id}" style="position: absolute; left: -9999px; opacity: 0;">{text_to_copy}</textarea>
        <button id="{button_id}" class="stButton">Copy Text</button>
    """
    
    # The JavaScript part: finds the elements and adds the copy logic
    js_code = f"""
<script>
    // Ensure this script runs only once or can handle re-runs
    if (!window.copyListeners) {{
        window.copyListeners = new Set();
    }}

    if (!window.copyListeners.has("{button_id}")) {{
        document.addEventListener('click', function(event) {{
            if (event.target.id === "{button_id}") {{
                var text = document.getElementById("{text_id}").value;
                
                navigator.clipboard.writeText(text).then(function() {{
                    var btn = document.getElementById("{button_id}");
                    if (btn) {{
                        var originalText = btn.innerHTML;
                        btn.innerHTML = 'Copied!';
                        setTimeout(function() {{
                            btn.innerHTML = originalText;
                        }}, 2000);
                    }}
                }}, function(err) {{
                    console.error('Could not copy text: ', err);
                }});
            }}
        }}, {{ capture: true }}); // Use capture to ensure event is caught
        window.copyListeners.add("{button_id}");
    }}
</script>
    """
    
    # Combine and render using st.components.v1.html
    st.components.v1.html(html_code + js_code, height=50)

# =======================================================
# --- NEW: FUNCTION TO SET ANIMATED GRADIENT BG ---
# =======================================================
def set_animated_fluid_background():
    """
    Sets a "Fluid Nebula" animated background - (Deep Blue & Purple theme)
    """
    st.markdown(
         """
        <style>
        @keyframes fluidMove {
            0% { background-position: 0% 50%; }
            25% { background-position: 100% 50%; }
            50% { background-position: 100% 100%; }
            75% { background-position: 0% 100%; }
            100% { background-position: 0% 50%; }
        }

        .stApp {
            /* --- THIS IS YOUR NEW DEEP BLUE GRADIENT --- */
            background: linear-gradient(45deg, #0a0c27, #001f5a, #4a0d6a, #0052D4);
            background-size: 300% 300%;
            animation: fluidMove 20s ease infinite;
            color: #ffffff;
        }
        
        /* --- Updated Component Styling --- */
        [data-testid="stSidebar"] > div:first-child {
            /* Base color is still dark indigo */
            background-color: rgba(10, 12, 39, 0.8);
        }
        
        /* Main chat area background */
        .st-emotion-cache-16txtl3 {
            background-color: transparent; 
        }
        
        /* Chat bubbles are now tinted deep blue */
        [data-testid="chat-message-container"] {
            background-color: rgba(0, 31, 90, 0.7);
            border-radius: 10px;
            padding: 10px !important;
            margin-bottom: 10px;
        }
        
        /* Make Streamlit's default button match the copy button */
        .stButton>button {
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            border: 1px solid #0052D4;
            background-color: #001f5a;
            color: white;
        }
        .stButton>button:hover {
            background-color: #0a0c27;
            border: 1px solid #4a0d6a;
        }
        
        /* Style the custom copy button */
        button#copy_btn {
            border-radius: 0.5rem;
            padding: 0.25rem 0.75rem;
            font-size: 0.8rem;
            border: 1px solid #0052D4;
            background-color: #001f5a;
            color: white;
            margin-top: 10px;
        }
        button#copy_btn:hover {
            background-color: #0a0c27;
            border: 1px solid #4a0d6a;
        }
        
        </style>
        """,
         unsafe_allow_html=True
     )
# =======================================================
# =====================
# Page Config and Setup
# =====================
st.set_page_config(page_title="🤖 AI Agent Workshop", page_icon="🧠", layout="wide")
set_animated_fluid_background()
# --- Securely load API keys from Streamlit Secrets ---
try:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
    pollinations_token = st.secrets["POLLINATIONS_TOKEN"]
    groq_api_key = st.secrets["GROQ_API_KEY"]
    mistral_api_key = st.secrets["MISTRAL_API_KEY"]
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
    st.session_state.metrics = {
        "total_requests": 0,
        "tool_usage": {"Comparison": 0, "Image Gen": 0, "Web Search": 0, "File Analysis": 0},
        "total_latency": 0.0,
        "average_latency": 0.0,
        "accuracy_feedback": {"👍": 0, "👎": 0},
        "last_query_details": {}
    }

# =====================
# Main Application UI & Sidebar
# =====================
st.title("🧠 AI Agent Workshop")
st.write("I can search the web, create images, analyze documents, and chain tasks together!")

with st.sidebar:
    st.header("🔍 Google Search")
    search_query = st.text_input("Search the web directly...", key="google_search")
    if st.button("Search"):
        if search_query:
            encoded_query = quote_plus(search_query)
            search_url = f"[https://www.google.com/search?q=](https://www.google.com/search?q=){encoded_query}"
            st.link_button("Open Google search results", url=search_url) 
        else:
            st.warning("Please enter a search query.")

    st.header("📂 File Analysis")
    uploaded_file = st.file_uploader("Upload a file to ask questions about it", type=["pdf", "txt", "py", "js", "html", "css"])
    
    st.header("🧭 Utilities")
    if st.button("Clear Chat History & Reset Metrics"):
        st.session_state.messages = []
        st.session_state.trajectory = []
        st.session_state.metrics = {
            "total_requests": 0, "tool_usage": {"Comparison": 0, "Image Gen": 0, "Web Search": 0, "File Analysis": 0},
            "total_latency": 0.0, "average_latency": 0.0, "accuracy_feedback": {"👍": 0, "👎": 0}, "last_query_details": {}
        }
        st.rerun()

    st.markdown("### 💡 AI Tip of the Day")
    st.info(random.choice([
        "Try a multi-step query! 'Search for the latest news on AI, then write an essay about it.'",
        "Ask about current events to see the Web Search tool in action!",
        "Upload a Python file and ask for a score to test File Analysis."
    ]))
    
    st.markdown("### 🕒 Live Server Time (IST)")
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

# --- NEW: Helper function to convert chat history ---
def convert_st_messages_to_langchain(st_messages: list[dict]) -> list[BaseMessage]:
    """Converts Streamlit message history to LangChain BaseMessage objects."""
    langchain_messages = []
    for msg in st_messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg.get("text", "")))
        elif msg["role"] == "assistant":
            # Only add text content to the history for the agent to read
            if msg.get("text"):
                langchain_messages.append(AIMessage(content=msg.get("text", "")))
    return langchain_messages

# =================================================================================
# --- MODIFIED: AGDebugger Logic to handle multi-step responses ---
# =================================================================================
async def run_agent_and_capture_trajectory(agent, prompt: str, history: list[BaseMessage]):
    """
    Runs the agent using astream_events and captures a detailed trace,
    the final list of responses, and all tools used.
    """
    trace_steps = []
    current_step = {}
    # --- MODIFIED: Capture a list of responses and tools ---
    all_responses = []
    tools_used = []

    # This is the input to the graph
    config = {"query": prompt, "history": history, "intermediate_responses": []}

    async for event in agent.astream_events(config, version="v1"):
        kind = event["event"]
        
        if kind == "on_chain_start":
            if event["name"] != "LangGraph":
                current_step = {
                    "name": event["name"],
                    "input": event["data"].get("input"),
                    "output": None
                }
                # --- MODIFIED: Capture friendly tool names in a list ---
                tool_name = "N/A"
                if event['name'] == "comparison_chat": tool_name = "Comparison"
                elif event['name'] == "image_generator": tool_name = "Image Gen"
                elif event['name'] == "web_search": tool_name = "Web Search"
                
                # Only add if it's a tool (not 'router') and not already logged
                if tool_name != "N/A" and tool_name not in tools_used:
                    tools_used.append(tool_name)

        if kind == "on_chain_end":
            if event["name"] != "LangGraph":
                output = event["data"].get("output")
                if current_step.get("name") == event["name"]:
                    current_step["output"] = output
                    trace_steps.append(current_step)
                    current_step = {}
                
                # --- MODIFIED: Get the full list of intermediate responses ---
                if isinstance(output, dict) and 'intermediate_responses' in output:
                    all_responses = output['intermediate_responses']

    # --- MODIFIED: Return the list of responses and tools ---
    return all_responses, trace_steps, tools_used

def pretty_print_dict(d):
    """Utility to safely print dictionaries containing complex objects."""
    def safe_converter(o):
        if isinstance(o, (Image.Image, bytes)):
            return f"<{type(o).__name__} object>"
        if isinstance(o, BaseMessage):
            return f"[{o.type}] {str(o.content)[:100]}..."
        try:
            return str(o)
        except Exception:
            return f"<Unserializable object: {type(o).__name__}>"
    
    if not isinstance(d, dict):
        return f"```\n{safe_converter(d)}\n```"
        
    return "```json\n" + json.dumps(d, indent=2, default=safe_converter) + "\n```"


# =================================================================================
# --- MODIFIED: Main Chat Input Logic (Handles history and multi-step output) ---
# =================================================================================
if prompt := st.chat_input("Ask about the latest news, create an image, or query a file..."):
    st.session_state.messages.append({"role": "user", "text": prompt})
    
    start_time = time.time()
    tools_used_list = [] # --- MODIFIED: Now a list

    if uploaded_file:
        # --- This is the File Analysis Path (Bypasses agent) ---
        tools_used_list = ["File Analysis"]
        with st.chat_message("assistant"):
            with st.spinner("Analyzing file..."):
                file_bytes = uploaded_file.read()
                file_text = ""
                
                if "pdf" in uploaded_file.type:
                    try:
                        reader = PdfReader(BytesIO(file_bytes))
                        for page in reader.pages: file_text += page.extract_text() or ""
                    except Exception as e:
                        st.warning(f"PyPDF2 failed ({e}), falling back to OCR...")
                        file_text = "" # Ensure it's empty to trigger OCR
                        
                    if not file_text.strip():
                        st.info("No text layer found, performing OCR...")
                        try:
                            doc = fitz.open(stream=file_bytes, filetype="pdf")
                            for page in doc:
                                pix = page.get_pixmap()
                                img = Image.open(BytesIO(pix.tobytes("png")))
                                file_text += pytesseract.image_to_string(img)
                            doc.close()
                        except Exception as ocr_e:
                            st.error(f"OCR failed: {ocr_e}")
                            file_text = "Error: Could not read PDF."
                else:
                    try:
                        file_text = file_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        file_text = file_bytes.decode("latin-1", errors="ignore")
                
                response_stream = file_analysis_tool(prompt, file_text, google_api_key)
                full_response = st.write_stream(response_stream)
                
                st.session_state.messages.append({"role": "assistant", "text": full_response, "audio_bytes": None})
    
    else:
        # --- MODIFIED: This is the Agent Path (Multi-Step) ---
        with st.spinner("Agent is working..."):
            agent = build_agent(google_api_key, groq_api_key, pollinations_token, tavily_api_key, mistral_api_key)
            
            # --- NEW: Convert history for the agent ---
            # We pass all messages *except* the last one (which is the new prompt)
            langchain_history = convert_st_messages_to_langchain(st.session_state.messages[:-1])
            
            # --- MODIFIED: Run agent and get lists back ---
            step_responses, trace_steps, tools_used_list = asyncio.run(
                run_agent_and_capture_trajectory(agent, prompt, langchain_history)
            )
            st.session_state.trajectory.append({"prompt": prompt, "steps": trace_steps})

        # --- NEW: Loop through each step and display it as a separate message ---
        if not step_responses:
            with st.chat_message("assistant"):
                error_msg = "Sorry, I ran into an issue and couldn't complete the task."
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "text": error_msg, "audio_bytes": None})
        
        for i, response in enumerate(step_responses):
            is_last_step = (i == len(step_responses) - 1)
            
            with st.chat_message("assistant"):
                if isinstance(response, str):
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "text": response, "audio_bytes": None})
                
                elif isinstance(response, dict) and "image" in response:
                    img_data = response["image"]
                    buf = BytesIO()
                    img_data.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    caption = response.get("caption", "Generated image")
                    st.image(byte_im, caption=caption)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "image_bytes": byte_im, 
                        "text": f"Image generated: *{caption}*",
                        "caption": caption
                    })
                
                else:
                    error_message = response.get("error", f"An intermediate step failed: {response}")
                    st.markdown(f"Error: {error_message}")
                    st.session_state.messages.append({"role": "assistant", "text": f"Error: {error_message}", "audio_bytes": None})
            
            # Add a visual pause to show the agent "thinking"
            if not is_last_step:
                with st.spinner("Agent is continuing to the next step..."):
                    time.sleep(1.5) # Visual pause
    
    # --- Metrics Recording (logic is now modified for list) ---
    end_time = time.time()
    latency = end_time - start_time
    metrics = st.session_state.metrics
    metrics["total_requests"] += 1
    
    # --- MODIFIED: Log all tools used in the chain ---
    if tools_used_list:
        for tool_name in tools_used_list:
            if tool_name in metrics["tool_usage"]:
                metrics["tool_usage"][tool_name] += 1
    else:
        if not uploaded_file: # If it was an agent run but no tools were identified (e.g., error)
            pass # We might want to log this as a "Routing Error"
            
    metrics["total_latency"] += latency
    metrics["average_latency"] = metrics["total_latency"] / metrics["total_requests"]
    metrics["last_query_details"] = {
        "timestamp": datetime.now().isoformat(), "prompt": prompt,
        "tool_used": ", ".join(tools_used_list) if tools_used_list else "N/A", # <-- Now shows all tools
        "latency_seconds": round(latency, 2)
    }
    
    st.rerun()

# --- MODIFIED: Debug View ---
if st.session_state.trajectory:
    with st.expander("🕵️ Agent Trajectory / Debug View", expanded=False):
        for run in reversed(st.session_state.trajectory):
            st.markdown(f"#### Prompt: *'{run.get('prompt', 'N/A')}'*")
            
            steps = run.get('steps', [])
            
            if not steps:
                st.write("No steps were captured for this run. This might indicate a routing failure.")
                
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