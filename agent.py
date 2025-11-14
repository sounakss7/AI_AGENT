import os
import re
import requests
from io import BytesIO
from PIL import Image
# --- MODIFIED: Added BaseMessage for history typing ---
from typing import TypedDict, Optional, list, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
import concurrent.futures
from functools import partial
from tavily import TavilyClient
from urllib.parse import quote_plus
import logging

# =======================================================================================
# TOOL 1: THE COMPARISON & EVALUATION WORKFLOW (Unchanged)
# =======================================================================================

def choose_groq_model(prompt: str):
    p = prompt.lower()
    if any(x in p for x in ["python", "code", "algorithm", "bug", "function", "script" ,"information" , "analysis" , "solution" , "NLP" ,"essay" , "mathematics" ,"research" ,"reasoning"]):
        return "openai/gpt-oss-120b"
    else:
        return "llama-3.1-8b-instant"

def query_groq(prompt: str, groq_api_key: str):
    """Queries Groq and returns a dict with model_name and content, or an error string."""
    model = choose_groq_model(prompt)
    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048}
    try:
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            return {"model_name": model, "content": content} 
        return f"❌ Groq API Error: {resp.text}"
    except Exception as e:
        return f"⚠️ Groq Error: {e}"

def query_mistral_judge(prompt: str, mistral_api_key: str):
    """A dedicated function to call Mistral for judging."""
    model = "mistral-small-latest"
    headers = {"Authorization": f"Bearer {mistral_api_key}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1024}
    try:
        resp = requests.post("https://api.mistral.ai/v1/chat/completions", json=data, headers=headers)
        resp.raise_for_status() 
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as http_err:
        logging.error(f"Mistral Judge HTTP Error: {http_err} - {resp.text}")
        return "Error: The Mistral judge failed to provide an evaluation (HTTP error)."
    except Exception as e:
        logging.error(f"Mistral Judge Exception: {e}")
        return f"Error: The Mistral judge ran into an exception: {e}"

def comparison_and_evaluation_tool(query: str, google_api_key: str, groq_api_key: str, mistral_api_key: str) -> str:
    """
    Runs a query through Gemini and Groq, has a MISTRAL AI judge evaluate the best response,
    and formats everything into a comprehensive answer.
    """
    print("---TOOL: Executing Comparison (Judged by Mistral)---")
    
    fast_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
    gemini_model_name = "gemini-2.5-flash"
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_gemini = executor.submit(lambda: fast_llm.invoke(query).content)
        future_groq = executor.submit(query_groq, query, groq_api_key)
        
        gemini_response = future_gemini.result()
        groq_result = future_groq.result()

    groq_response = ""
    groq_model_name = "Groq (Error)"

    if isinstance(groq_result, dict):
        groq_response = groq_result["content"]
        groq_model_name = groq_result["model_name"]
    else:
        groq_response = groq_result

    judge_prompt = f"""
    You are an impartial AI evaluator. Compare two responses to a user's query and declare a winner.
    ### User Query:
    {query}
    ### Response A (Gemini):
    {gemini_response}
    ### Response B (Groq - model: {groq_model_name}):
    {groq_response}
    Instructions:
    1. Begin with "Winner: Gemini" or "Winner: Groq".
    2. Explain your reasoning clearly, comparing accuracy, clarity, and completeness.
    3. **Evaluate the responses purely on their merit for the given query. Do not show bias towards any model provider. Your judgment must be neutral and unbiased.**
    """
    
    print("---JUDGE: Calling Mistral for evaluation---")
    judgment = query_mistral_judge(judge_prompt, mistral_api_key)
    
    match = re.search(r"winner\s*:\s*(gemini|groq)", judgment, re.IGNORECASE)
    winner_name = match.group(1).capitalize() if match else "Evaluation"
    
    chosen_answer = ""
    chosen_model_name = ""
    loser_response = ""
    loser_model_name = ""
    loser_name = ""

    if winner_name == "Gemini":
        chosen_answer = gemini_response
        chosen_model_name = gemini_model_name
        loser_response = groq_response
        loser_model_name = groq_model_name
        loser_name = "Groq"
    elif winner_name == "Groq":
        chosen_answer = groq_response
        chosen_model_name = groq_model_name
        loser_response = gemini_response
        loser_model_name = gemini_model_name
        loser_name = "Gemini"
    else:
        chosen_answer = gemini_response
        chosen_model_name = gemini_model_name
        loser_response = groq_response
        loser_model_name = groq_model_name
        loser_name = "Groq"

    final_output = f"### 🏆 Judged Best Answer ({winner_name})\n"
    final_output += f"#### Model: {chosen_model_name}\n\n{chosen_answer}\n\n"
    final_output += f"### 🧠 Judge's Evaluation (from Mistral)\n{judgment}\n\n---\n\n"
    final_output += f"### Other Response ({loser_name})\n\n"
    final_output += f"#### Model: {loser_model_name}\n\n{loser_response}"
    
    return final_output

# ===================================================================
# TOOL 2: IMAGE GENERATION TOOL (Unchanged)
# ===================================================================
def image_generation_tool(prompt: str, google_api_key: str, pollinations_token: str) -> dict:
    """Use this tool when the user asks to create, draw, or generate an image."""
    logging.info(f"---TOOL: Generating Image for prompt: '{prompt}'---")
    try:
        enhancer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
        enhancer_prompt = f"Rewrite this short prompt into a detailed, vibrant, and artistic image generation description: {prompt}"
        final_prompt = enhancer_llm.invoke(enhancer_prompt).content.strip()
        
        encoded_prompt = quote_plus(final_prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?token={pollinations_token}"
        response = requests.get(url, timeout=120)
        response.raise_for_status() 
        img_bytes = response.content
        img = Image.open(BytesIO(img_bytes))
        
        return {"image": img, "caption": f"Your prompt: '{prompt}'"}

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err} - Response: {response.text}")
        return {"error": f"The image generation service returned an error: {http_err}"}
    except requests.exceptions.ReadTimeout as timeout_err:
        logging.error(f"Image generation timed out: {timeout_err}")
        return {"error": "The image generation service timed out (took longer than 120s). It might be very busy. Please try again in a moment."}
    except Exception as e:
        logging.error(f"An unexpected error occurred in image generation: {e}")
        return {"error": f"Failed to generate image: {e}"}

# ===================================================================
# TOOL 3: FILE ANALYSIS TOOL (Unchanged)
# ===================================================================
def file_analysis_tool(question: str, file_content_as_text: str, google_api_key: str):
    """
    Use this tool when the user has uploaded a file and is asking a question about it.
    This tool is now empowered to use its own expertise to provide a comprehensive analysis.
    """
    print("---TOOL: Executing Empowered File Analysis---")
    streaming_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, streaming=True) 
    
    prompt = f"""
    **Your Persona:** You are a highly intelligent AI assistant and a multi-disciplinary expert... (rest of prompt unchanged)
    **User's Question:**
    {question}
    **Provided File Content:**
    ---
    {file_content_as_text[:40000]} 
    ---
    **Your Comprehensive Analysis:**
    """
    return streaming_llm.stream([HumanMessage(content=prompt)])

# ===================================================================
# TOOL 4: WEB SEARCH & REAL-TIME DATA ANALYSIS (Unchanged)
# ===================================================================
def web_search_tool(query: str, tavily_api_key: str, google_api_key: str) -> str:
    """
    Use this tool to get real-time information, answer questions about current events,
    or for any query that requires up-to-date knowledge from the internet.
    """
    print("---TOOL: Executing Web Search and Analysis---")
    try:
        tavily = TavilyClient(api_key=tavily_api_key)
        search_results = tavily.search(query=query, search_depth="advanced", max_results=5)
        search_content = "\n".join([result["content"] for result in search_results["results"]])
        
        analyzer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
        analysis_prompt = f"""
        You are an expert research analyst... (rest of prompt unchanged)
        ### User Query:
        {query}
        ### Web Search Results:
        ---
        {search_content}
        ---
        Your Answer:
        """
        final_answer = analyzer_llm.invoke(analysis_prompt).content
        return final_answer
        
    except Exception as e:
        return f"⚠️ Web search failed: {e}"
    
# ===================================================================
# THE AGENT: A "WORKSHOP MANAGER" THAT CHOOSES THE RIGHT TOOL
# ===================================================================

# --- MODIFIED: AgentState is now stateful ---
class AgentState(TypedDict):
    query: str
    route: str
    history: list[BaseMessage]
    # This will accumulate responses from each tool step
    intermediate_responses: list[Any] 

# --- MODIFIED: All tool-calling nodes now append to intermediate_responses ---
def call_comparison_tool(state: AgentState, google_api_key: str, groq_api_key: str, mistral_api_key: str):
    print("---AGENT: Calling Comparison Tool---")
    # --- NEW: Get query from the *last* text response if available, else use original query
    # This allows chaining, e.g., "search for X" -> "now write an essay about X"
    query_to_use = state['query']
    if state['history']:
        # Get the last AI text message
        for msg in reversed(state['history']):
            if msg.type == "ai" and isinstance(msg.content, str):
                query_to_use = msg.content # Use the output of the last step as input
                break
    
    response = comparison_and_evaluation_tool(
        query_to_use, # Use the potentially modified query
        google_api_key, 
        groq_api_key,
        mistral_api_key
    )
    current_steps = state.get('intermediate_responses', [])
    return {"intermediate_responses": current_steps + [response]}

def call_image_tool(state: AgentState, google_api_key: str, pollinations_token: str):
    print("---AGENT: Calling Image Tool---")
    # --- NEW: Get prompt from the *last* text response if available
    prompt_to_use = state['query']
    if state['history']:
        for msg in reversed(state['history']):
            if msg.type == "ai" and isinstance(msg.content, str):
                prompt_to_use = msg.content # Use the output of the last step as prompt
                print(f"---AGENT: Using chained prompt for image: '{prompt_to_use[:50]}...'---")
                break
            elif msg.type == "human":
                prompt_to_use = msg.content # Or the original user query
                break
                
    response = image_generation_tool(prompt_to_use, google_api_key, pollinations_token)
    current_steps = state.get('intermediate_responses', [])
    return {"intermediate_responses": current_steps + [response]}

def call_web_search_tool(state: AgentState, tavily_api_key: str, google_api_key: str):
    print("---AGENT: Calling Web Search Tool---")
    response = web_search_tool(state['query'], tavily_api_key, google_api_key)
    current_steps = state.get('intermediate_responses', [])
    return {"intermediate_responses": current_steps + [response]}

# --- MODIFIED: The Router is now much smarter and decides when to END ---
def router(state: AgentState, google_api_key: str):
    """
    The brain of the agent. Decides which tool to use next, or to end the chain.
    """
    print("---AGENT: Routing query---")
    router_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
    
    query = state['query']
    history = state.get('history', [])
    steps = state.get('intermediate_responses', [])
    
    # --- NEW: This prompt is completely rewritten for chaining ---
    router_prompt = f"""
    You are a master routing agent. Your goal is to break down a complex user query into a series of steps, executed one by one.
    You will be given the user's original query and the history of steps already taken.
    Your job is to decide the *next* step.

    You have four choices:
    1.  `comparison_tool`: Use for complex questions, coding problems, analysis, text generation, or any task that requires a detailed, evaluated answer.
    2.  `image_generation_tool`: Use ONLY if the user explicitly asks to create, draw, or generate an image.
    3.  `web_search_tool`: Use this for any query that requires real-time, up-to-date information, news, or current events.
    4.  `END`: Use this ONLY when you are 100% certain the user's original query has been fully and completely answered by the previous steps. If the user asked for two things (e.g., "search for X and draw it") and you have only done one, DO NOT end.

    ### User's Original Query:
    "{query}"

    ### Chat History (for context):
    {history}

    ### Steps Taken So Far:
    {steps if steps else "None. This is the first step."}

    Based on the query and the steps already taken, what is the *next* tool to use?
    If the last step was an image, the task is probably finished, so choose END.
    If the last step was a web search, and the user *also* wanted an image, choose `image_generation_tool`.
    
    Return ONLY the tool name (`comparison_tool`, `image_generation_tool`, `web_search_tool`) or `END`.
    """
    
    response = router_llm.invoke(router_prompt).content.strip()
    
    if "web_search_tool" in response:
        print("---AGENT: Decision -> Web Search Tool---")
        return {"route": "web_search"}
    elif "image_generation_tool" in response:
        print("---AGENT: Decision -> Image Generation Tool---")
        return {"route": "image_generator"}
    elif "comparison_tool" in response:
        print("---AGENT: Decision -> Comparison & Evaluation Tool---")
        return {"route": "comparison_chat"}
    else:
        print("---AGENT: Decision -> END---")
        return {"route": "END"}

# --- MODIFIED: Define the Agentic Graph with new looping logic ---
def build_agent(google_api_key: str, groq_api_key: str, pollinations_token: str, tavily_api_key: str, mistral_api_key: str):
    """
    Builds the agent graph.
    """
    workflow = StateGraph(AgentState)

    router_with_keys = partial(router, google_api_key=google_api_key)
    
    comparison_node = partial(
        call_comparison_tool, 
        google_api_key=google_api_key, 
        groq_api_key=groq_api_key,
        mistral_api_key=mistral_api_key
    )
    image_node = partial(call_image_tool, google_api_key=google_api_key, pollinations_token=pollinations_token)
    web_search_node = partial(call_web_search_tool, tavily_api_key=tavily_api_key, google_api_key=google_api_key)

    workflow.add_node("router", router_with_keys)
    workflow.add_node("comparison_chat", comparison_node)
    workflow.add_node("image_generator", image_node)
    workflow.add_node("web_search", web_search_node)

    workflow.set_entry_point("router")
    
    # --- MODIFIED: Conditional Edges now include the END route ---
    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "comparison_chat": "comparison_chat", 
            "image_generator": "image_generator", 
            "web_search": "web_search",
            "END": END  # <-- This is the new exit point
        }
    )
    
    # --- MODIFIED: All tool nodes now loop back to the router ---
    workflow.add_edge("comparison_chat", "router")
    workflow.add_edge("image_generator", "router")
    workflow.add_edge("web_search", "router")
    
    return workflow.compile()
