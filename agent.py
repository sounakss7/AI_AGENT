import os
import re
import requests
from io import BytesIO
from PIL import Image
from typing import TypedDict, Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langgraph.graph import StateGraph, END
import concurrent.futures
from functools import partial
from tavily import TavilyClient
import logging
import json

# =======================================================================================
# This section remains unchanged. Your tools are the "skills" the planner can use.
# =======================================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    logging.info("---TOOL: Executing the Comparison & Evaluation Workflow---")
    fast_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
    judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_gemini = executor.submit(lambda: fast_llm.invoke(query).content)
        future_groq = executor.submit(query_groq, query, groq_api_key)
        gemini_response, groq_response = future_gemini.result(), future_groq.result()
    judge_prompt = f"""You are an impartial AI evaluator... (prompt content is unchanged)"""
    judgment = judge_llm.invoke(judge_prompt).content
    match = re.search(r"winner\s*:\s*(gemini|groq)", judgment, re.IGNORECASE)
    winner = match.group(1).capitalize() if match else "Evaluation"
    chosen_answer = gemini_response if winner == "Gemini" else groq_response
    final_output = f"## 🏆 Judged Best Answer ({winner})\n{chosen_answer}\n\n### 🧠 Judge's Evaluation\n{judgment}\n\n---\n\n### Other Responses\n\n**🤖 Gemini's Full Response:**\n{gemini_response}\n\n**⚡ Groq's Full Response:**\n{groq_response}"
    return final_output

def image_generation_tool(prompt: str, google_api_key: str, pollinations_token: str) -> dict:
    logging.info("---TOOL: Generating Image---")
    try:
        enhancer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
        enhancer_prompt = f"Rewrite this short prompt into a detailed, vibrant, and artistic image generation description: {prompt}"
        final_prompt = enhancer_llm.invoke(enhancer_prompt).content.strip()
        url = f"https://image.pollinations.ai/prompt/{final_prompt}?token={pollinations_token}"
        img_bytes = requests.get(url, timeout=30).content
        img = Image.open(BytesIO(img_bytes))
        return {"image": img, "caption": f"Your prompt: '{prompt}'"}
    except Exception as e:
        return {"error": f"Failed to generate image: {e}"}

def file_analysis_tool(question: str, file_content_as_text: str, google_api_key: str):
    logging.info("---TOOL: Executing Empowered File Analysis---")
    streaming_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, streaming=True)
    prompt = f"""**Your Persona:** You are a highly intelligent AI assistant... (prompt content is unchanged)"""
    return streaming_llm.stream([HumanMessage(content=prompt)])

def web_search_tool(query: str, tavily_api_key: str, google_api_key: str) -> str:
    logging.info("---TOOL: Executing Web Search and Analysis---")
    try:
        tavily = TavilyClient(api_key=tavily_api_key)
        search_results = tavily.search(query=query, search_depth="advanced", max_results=5)
        search_content = "\n".join([result["content"] for result in search_results["results"]])
        analyzer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
        analysis_prompt = f"""You are an expert research analyst... (prompt content is unchanged)"""
        final_answer = analyzer_llm.invoke(analysis_prompt).content
        return final_answer
    except Exception as e:
        return f"⚠️ Web search failed: {e}"


# ===================================================================
# --- MODIFIED: This section is now the Plan-and-Execute Agent ---
# ===================================================================

# Define the state for the new agent
class PlanExecuteState(TypedDict):
    query: str
    plan: List[str]
    step_results: List[str]
    final_response: Optional[any]

# Define the nodes for the new graph
def planner_node(state: PlanExecuteState, google_api_key: str):
    logging.info("---AGENT: Generating a plan---")
    planner_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
    
    planner_prompt = f"""
    You are a master planner. Your job is to create a step-by-step plan to answer the user's query.
    You must decompose the query into a series of concrete steps that use my available tools.
    
    ## Available Tools:
    - `web_search`: Use this for any query that requires real-time, up-to-date information from the internet.
    - `image_generator`: Use this ONLY if the user explicitly asks to create, draw, or generate an image.
    - `comparison_tool`: Use this for complex questions, coding problems, or analysis that could benefit from two models and a judge.
    
    ## Instructions:
    - The output must be a JSON list of strings.
    - Each string in the list is a single step in the plan.
    - Each step MUST start with the name of the tool to use, followed by a colon, followed by the detailed query for that tool.
    - If a single tool call is sufficient, create a plan with just one step.
    - Example: ["web_search: latest news about ISRO's Gaganyaan mission", "image_generator: A futuristic Indian astronaut looking at Earth from space"]
    
    ## User's Query:
    "{state['query']}"
    
    ## Your Plan (as a JSON list of strings):
    """
    
    response = planner_llm.invoke(planner_prompt).content.strip()
    plan = json.loads(response)
    logging.info(f"---AGENT: Generated Plan -> {plan}---")
    
    return {"plan": plan}

def tool_executor_node(state: PlanExecuteState, google_api_key: str, groq_api_key: str, pollinations_token: str, tavily_api_key: str):
    plan = state['plan']
    step = plan.pop(0)
    
    tool_name, query_for_tool = step.split(":", 1)
    tool_name = tool_name.strip()
    query_for_tool = query_for_tool.strip()
    
    logging.info(f"---AGENT: Executing step -> Tool: {tool_name}, Query: {query_for_tool}---")
    
    result = ""
    if tool_name == "web_search":
        result = web_search_tool(query_for_tool, tavily_api_key, google_api_key)
    elif tool_name == "image_generator":
        result = image_generation_tool(query_for_tool, google_api_key, pollinations_token)
    elif tool_name == "comparison_tool":
        result = comparison_and_evaluation_tool(query_for_tool, google_api_key, groq_api_key)
        
    current_results = state.get("step_results", [])
    current_results.append(result)
    
    return {"plan": plan, "step_results": current_results}

def final_response_node(state: PlanExecuteState, google_api_key: str):
    logging.info("---AGENT: Generating final response---")
    
    last_result = state["step_results"][-1]
    if isinstance(last_result, dict) and "image" in last_result:
        return {"final_response": last_result}

    summarizer_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
    
    summarizer_prompt = f"""
    You are a helpful AI assistant. You have just completed a plan to answer a user's query.
    Your task is to synthesize the results from the executed steps into a single, clear, and comprehensive final answer.

    ## User's Original Query:
    {state['query']}
    
    ## Results from Executed Steps:
    {state['step_results']}
    
    ## Your Final Answer:
    """
    
    final_answer = summarizer_llm.invoke(summarizer_prompt).content.strip()
    return {"final_response": final_answer}

def should_continue(state: PlanExecuteState):
    if state["plan"]:
        return "continue"
    else:
        return "end"

# --- This function now builds the new graph ---
def build_agent(google_api_key: str, groq_api_key: str, pollinations_token: str, tavily_api_key: str):
    workflow = StateGraph(PlanExecuteState)

    workflow.add_node("planner", partial(planner_node, google_api_key=google_api_key))
    workflow.add_node("executor", partial(tool_executor_node, google_api_key=google_api_key, groq_api_key=groq_api_key, pollinations_token=pollinations_token, tavily_api_key=tavily_api_key))
    workflow.add_node("responder", partial(final_response_node, google_api_key=google_api_key))

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "executor")
    
    workflow.add_conditional_edges(
        "executor",
        should_continue,
        {
            "continue": "executor",
            "end": "responder"
        }
    )
    
    workflow.add_edge("responder", END)

    return workflow.compile()