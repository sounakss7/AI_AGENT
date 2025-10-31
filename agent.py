import os
import re
import requests
from io import BytesIO
from PIL import Image
from typing import TypedDict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, END
import concurrent.futures
from functools import partial
from tavily import TavilyClient
from urllib.parse import quote_plus # --- NEW IMPORT ---
import logging

# =======================================================================================
# TOOL 1: THE COMPARISON & EVALUATION WORKFLOW
# =======================================================================================

def choose_groq_model(prompt: str):
    p = prompt.lower()
    if any(x in p for x in ["python", "code", "algorithm", "bug", "function", "script"]):
        return "openai/gpt-oss-20b"
    else:
        return "llama-3.1-8b-instant"

def query_groq(prompt: str, groq_api_key: str):
    model = choose_groq_model(prompt)
    headers = {"Authorization": f"Bearer {groq_api_key}", "Content-Type": "application/json"}
    data = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048}
    try:
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers)
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            return f"**Model:** {model}\n\n{content}"
        return f"❌ Groq API Error: {resp.text}"
    except Exception as e:
        return f"⚠️ Groq Error: {e}"

def comparison_and_evaluation_tool(query: str, google_api_key: str, groq_api_key: str) -> str:
    """
    Runs a query through Gemini and Groq, has an AI judge evaluate the best response,
    and formats everything into a comprehensive answer.
    """
    print("---TOOL: Executing the Comparison & Evaluation Workflow---")
    
    # Use the fast model for the head-to-head comparison
    fast_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
    # Use the powerful model for the critical task of judging
    judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_gemini = executor.submit(lambda: fast_llm.invoke(query).content)
        future_groq = executor.submit(query_groq, query, groq_api_key)
        gemini_response = future_gemini.result()
        groq_response = future_groq.result()

    judge_prompt = f"""
    You are an impartial AI evaluator. Compare two responses to a user's query and declare a winner.

    ### User Query:
    {query}
    ### Response A (Gemini):
    {gemini_response}
    ### Response B (Groq):
    {groq_response}

    Instructions:
    1. Begin with "Winner: Gemini" or "Winner: Groq".
    2. Explain your reasoning clearly, comparing accuracy, clarity, and completeness.
    """
    judgment = judge_llm.invoke(judge_prompt).content
    
    match = re.search(r"winner\s*:\s*(gemini|groq)", judgment, re.IGNORECASE)
    winner = match.group(1).capitalize() if match else "Evaluation"
    
    chosen_answer = gemini_response if winner == "Gemini" else groq_response
    
    final_output = f"## 🏆 Judged Best Answer ({winner})\n"
    final_output += f"{chosen_answer}\n\n"
    final_output += f"### 🧠 Judge's Evaluation\n{judgment}\n\n---\n\n"
    final_output += f"### Other Responses\n\n"
    final_output += f"**🤖 Gemini's Full Response:**\n{gemini_response}\n\n"
    final_output += f"**⚡ Groq's Full Response:**\n{groq_response}"
    
    return final_output

# ===================================================================
# TOOL 2: IMAGE GENERATION TOOL
# ===================================================================
def image_generation_tool(prompt: str, google_api_key: str, pollinations_token: str) -> dict:
    """Use this tool when the user asks to create, draw, or generate an image."""
    logging.info(f"---TOOL: Generating Image for prompt: '{prompt}'---")
    try:
        enhancer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
        enhancer_prompt = f"Rewrite this short prompt into a detailed, vibrant, and artistic image generation description: {prompt}"
        final_prompt = enhancer_llm.invoke(enhancer_prompt).content.strip()
        
        # --- FIX 1: URL Encode the prompt to handle special characters ---
        encoded_prompt = quote_plus(final_prompt)
        
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?token={pollinations_token}"
        
        # --- FIX 2: MODIFIED - Increased timeout to 180 seconds ---
        response = requests.get(url, timeout=180)
        
        # --- FIX 3: Check if the request was successful before processing ---
        response.raise_for_status()  # This will raise an error for bad status codes (4xx or 5xx)
        
        # If the above line passes, we know we have a valid response
        img_bytes = response.content
        img = Image.open(BytesIO(img_bytes))
        
        return {"image": img, "caption": f"Your prompt: '{prompt}'"}

    except requests.exceptions.HTTPError as http_err:
        logging.error(f"HTTP error occurred: {http_err} - Response: {response.text}")
        return {"error": f"The image generation service returned an error: {http_err}"}
    
    # --- NEW: Catch ReadTimeout specifically to give a better error message ---
    except requests.exceptions.ReadTimeout as timeout_err:
        logging.error(f"Image generation timed out: {timeout_err}")
        return {"error": "The image generation service timed out (took longer than 180s). It might be very busy. Please try again in a moment."}
    
    except Exception as e:
        logging.error(f"An unexpected error occurred in image generation: {e}")
        return {"error": f"Failed to generate image: {e}"}
# ===================================================================
# TOOL 3: FILE ANALYSIS TOOL (Streams output)
# ===================================================================

def file_analysis_tool(question: str, file_content_as_text: str, google_api_key: str):
    """
    Use this tool when the user has uploaded a file and is asking a question about it.
    This tool is now empowered to use its own expertise to provide a comprehensive analysis.
    """
    print("---TOOL: Executing Empowered File Analysis---")
    # Using a more powerful model for high-quality analysis and evaluation
    streaming_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, streaming=True) 
    
    # This new, "freer" prompt gives the agent permission to be a true expert.
    prompt = f"""
    **Your Persona:** You are a highly intelligent AI assistant and a multi-disciplinary expert. Your goal is to provide the most helpful and insightful analysis possible, combining the provided file content with your own vast knowledge.

    **The Task:** A user has uploaded a file and asked a question. Use the file content as the primary source of truth and context, but you are encouraged to enrich your answer with your own expertise, especially when asked for an evaluation, opinion, or a subjective score.

    **How to Behave Based on File Content:**
    * **If the file appears to be CODE (e.g., Python, JavaScript, etc.):** Act as a senior software engineer. Analyze its structure, logic, efficiency, and style. If the user asks for a score or review, provide a thoughtful evaluation with justifications and suggestions for improvement.
    * **If the file appears to be TEXT (e.g., an article, report, essay):** Act as a research analyst. Summarize key points, extract specific information, and answer the user's questions. You can add relevant context from your own knowledge if it enhances the answer (e.g., providing historical context for an article).
    * **For all other file types:** Do your best to interpret the text content and provide a helpful, intelligent response to the user's question.

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
# THE AGENT: A "WORKSHOP MANAGER" THAT CHOOSES THE RIGHT TOOL
# ===================================================================

# ===================================================================
# NEW TOOL 3: WEB SEARCH & REAL-TIME DATA ANALYSIS
# ===================================================================
def web_search_tool(query: str, tavily_api_key: str, google_api_key: str) -> str:
    """
    Use this tool to get real-time information, answer questions about current events,
    or for any query that requires up-to-date knowledge from the internet.
    """
    print("---TOOL: Executing Web Search and Analysis---")
    try:
        tavily = TavilyClient(api_key=tavily_api_key)
        # Perform a search and get the most relevant results
        search_results = tavily.search(query=query, search_depth="advanced", max_results=5)
        
        # Extract the content from the search results
        search_content = "\n".join([result["content"] for result in search_results["results"]])
        
        # Use a powerful LLM to analyze the search results and answer the user's query
        analyzer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
        analysis_prompt = f"""
        You are an expert research analyst. You have been given a user's query and the results from a web search.
        Your task is to provide a clear, concise, and comprehensive answer to the user's query based *only* on the provided search results.
        Cite your sources using the information available in the search results if possible.

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
    
class AgentState(TypedDict):
    query: str
    route: str  # Add this key to store the router's decision
    final_response: Optional[any]
# These are wrapper functions for the nodes to handle passing state and API keys
def call_comparison_tool(state: AgentState, google_api_key: str, groq_api_key: str):
    response = comparison_and_evaluation_tool(state['query'], google_api_key, groq_api_key)
    return {"final_response": response}

def call_image_tool(state: AgentState, google_api_key: str, pollinations_token: str):
    response = image_generation_tool(state['query'], google_api_key, pollinations_token)
    return {"final_response": response}
def call_web_search_tool(state: AgentState, tavily_api_key: str, google_api_key: str):
    response = web_search_tool(state['query'], tavily_api_key, google_api_key)
    return {"final_response": response}

def router(state: AgentState, google_api_key: str):
    """The brain of the agent. Decides which tool to use and updates the 'route' state key."""
    print("---AGENT: Routing query---")
    router_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
    query = state['query']
    
    router_prompt = f"""
    You are a master routing agent. Determine the user's primary intent and select the appropriate tool. You have two choices:
    1.  `comparison_tool`: Use for complex questions, coding problems, analysis, or any text-based query needing a detailed, evaluated answer.
    2.  `image_generation_tool`: Use ONLY if the user explicitly asks to create, draw, or generate an image.
    3.  `web_search_tool`: Use this for any query that requires real-time, up-to-date information. This includes questions about current events, news, weather, recent scientific discoveries, or topics created after 2023.

    User Query: "{query}"
    Return ONLY the tool name (`comparison_tool` or `image_generation_tool` or `web_search_tool`).
    """
    response = router_llm.invoke(router_prompt).content.strip()
    
    if "web_search_tool" in response:
        print("---AGENT: Decision -> Web Search Tool---")
        return {"route": "web_search"}
    elif "image_generation_tool" in response:
        print("---AGENT: Decision -> Image Generation Tool---")
        return {"route": "image_generator"}
    else:
        print("---AGENT: Decision -> Comparison & Evaluation Tool---")
        return {"route": "comparison_chat"}
# --- Define the Agentic Graph ---
def build_agent(google_api_key: str, groq_api_key: str, pollinations_token: str , tavily_api_key: str ):
    workflow = StateGraph(AgentState)

    router_with_keys = partial(router, google_api_key=google_api_key)
    comparison_node = partial(call_comparison_tool, google_api_key=google_api_key, groq_api_key=groq_api_key)
    image_node = partial(call_image_tool, google_api_key=google_api_key, pollinations_token=pollinations_token)
    web_search_node = partial(call_web_search_tool, tavily_api_key=tavily_api_key, google_api_key=google_api_key)

    workflow.add_node("router", router_with_keys)
    workflow.add_node("comparison_chat", comparison_node)
    workflow.add_node("image_generator", image_node)
    workflow.add_node("web_search", web_search_node)

    workflow.set_entry_point("router")
    
    # The conditional edge function now simply reads the 'route' from the state
    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {"comparison_chat": "comparison_chat", "image_generator": "image_generator" , "web_search": "web_search"}
    )
    
    workflow.add_edge("comparison_chat", END)
    workflow.add_edge("image_generator", END)
    workflow.add_edge("web_search", END)
    return workflow.compile() 
