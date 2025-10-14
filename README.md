# 🧠 AI Agent Workshop

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced multi-tool AI agent built with Streamlit and LangGraph. This interactive web application serves as a workshop to demonstrate and test the capabilities of a modern AI agent that can search the web, generate images, perform comparative analysis between models, and analyze user-uploaded files.

## ✨ Features

-   **🤖 Agentic Router:** A central LangGraph agent that intelligently routes user queries to the most appropriate tool.
-   **🌐 Real-Time Web Search:** Utilizes the Tavily API to fetch current information from the internet and provides summarized, relevant answers.
-   **🎨 AI Image Generation:** Integrates with the Pollinations AI API to generate images from user prompts, first enhancing the prompts with Gemini for more artistic results.
-   **⚖️ Dual-Model Comparison & Evaluation:** A unique tool that runs the same query on both **Google's Gemini 2.5 Flash** and a model via the **Groq API** (Llama 3.1 or GPT-OSS). A "Judge" LLM then evaluates both responses to determine the winner.
-   **📂 Advanced File Analysis:**
    -   Supports various file types (`.pdf`, `.txt`, `.py`, `.js`, etc.).
    -   Features an **OCR fallback** for PDFs without a text layer.
    -   Uses a powerful Gemini model with an "Expert Persona" to provide in-depth analysis, code reviews, or document summaries.
-   **📊 Live Performance Dashboard:** The sidebar provides real-time metrics on agent performance, including total requests, average latency, tool usage distribution, and user feedback accuracy.

## ⚙️ Architecture & Workflow

The application operates on two primary workflows, determined by whether a file is uploaded.

1.  **File Analysis Path:** A direct, non-agentic path where uploaded file content is processed and sent directly to the `file_analysis_tool`.
2.  **Core Agent Path:** When no file is present, the query is handled by the LangGraph agent, which routes it to the appropriate tool (Web Search, Image Gen, or Comparison).

![Agent Workflow Diagram](https://mermaid.ink/img/pako:eNqdVU1v2zAQ_ivFzwJIEtgh6LoN2m2DFg3aD4W2HAmJbSUkUiTZcfAY_t9HSVK0S9fBgJ3gQSKfO-Qe3WOpvV1zD6YQJ2_kXb48zMljmC-p5eRtv-17r4c2S6l07JjCjY59G5R341Tq_xX_M2OqH4I9WjSgG_w05bQe-x-G_Z4WpC9nQpP-f3Vw4j26k6P406z96u8-7U-47_qV-FfF-d7h8-e6_aT-O6n0uO1W9f_w8M7x-n92lO9kXQ5b5h8E5M7F0y6t7E-f4o5434dM4xG5-dO_6-u7p4yLllgU6BqOa51pT8y2eE0w66jXQy6X4t_VfC3u1JqP7tOqL4aQz114_C92z_V8d5T09o8e_d0K3r7n8iP1H1-H188-f5Xf798p_5d9XnQ-T39O3683f8u26t_4U-1-3xG3f-Pj-8-5p-r37s-5l3bY8yT9i_084Hh87b_03sT4U-PzF66j_tW-XqO3t0O5_t-z6O7U505y_Nf21f7V7_r_eW84T8_xR_n1-1X8uP869_v1P3iPj_Nn6_v3x-eT6h7_Xn8n-s5P43P1f_F8e5P3L2563sT-1_pT2fL9Vf8f903x6-tX9T3-nL9UfzT9P38uX-R-27P5X-n-H1-e7v8-396v9j86P2T-L-Xf--5X2x_d_f19-v8b-8t_2H8h_p5_rW9_L_uW8b_Xf-t8uT7Ff-b9Ue7f6O83_p5f-r-aXk_rT_n3f8-8q-f5P-d_Jv-fT-t3_T6-lH-9fJ_qf-9-tL-W_-T_i_97-T_fFv_q_aL_7P-H38e7f-7_d3-X_r35P6b_r_Wf1n_9P1b-fX-1-f1r_7_F_9b-b-2f3b8t_l_J--f4t_S_m_9v_Z-d_m39n-3f1v9-_a_mf339n-f_p_6f9v-z_l3_J_6b_T_W_9__J_0r-L-X-L_j_7-2v9n3W_5P-L_S_4_5f6f6P-r-j_k-m_ov9L_f_V_i39W_S_1f-r_rf6-0X_H_-T-n-r_zH9X-L-x_S_qP9T_S_z_9H-r9R_oP5L_S_3_wP538f9b9f9o-o_rf6n_z-X_5f-T_J_qP6P-D_V_1X_f_l_i_-n_lfk_8n_J-2P6_9L-9-z_S__n-k_U_yf9X_a_S_S_l_yP_7-L-k_T_Wv2P6n_K_if5X6n_b_W_N_S_1v9J_a_W_V_d_2v9r-J_a_S_Xf8r9r_L_Q_4f-L_3v4P-l_S_k_4v8r_J_6_9r-l_Z_lf9X-z_T_2P87-r_lf2v6b9R_r-lf1H_L_p_Tf8f-z_b_mf639D_J_0v9f-2_W-Z_k_7v9H_F_1f73_N_if-v_Z_qf93_H_w_9v-J_j_6v-F_G_J_8v-1fW_1P-X-L-6_z_yf2H_l_b_ov8v-v_p_0P_n_r-i_o_Qv-f_b-2v_n_Jf_f_Z-x_8P_D_4_8f-j_e_y_9n-r-l_V_2_qf-X9j_lfzP-3_h_3P-n_F_6-u__AmlWw0U?version=0.1)

## 🛠️ Tech Stack

-   **Frontend:** [Streamlit](https://streamlit.io/)
-   **AI Orchestration:** [LangChain](https://www.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/)
-   **Language:** Python
-   **Core Libraries:** Pandas, Pillow, PyPDF2, PyMuPDF (fitz), Pytesseract
-   **APIs & Services:**
    -   **LLMs:** Google Gemini, Groq API (Llama, OpenAI models)
    -   **Web Search:** Tavily AI
    -   **Image Generation:** Pollinations AI

## 🚀 Getting Started

Follow these instructions to set up and run the project locally.

### 1. Prerequisites

-   Python 3.9 or higher
-   Git
-   **Tesseract OCR Engine:** This is a system-level dependency required for the file analysis tool's OCR fallback.
    -   Follow the official installation guide for your operating system: [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
    -   Make sure to add the Tesseract executable to your system's PATH.

### 2. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_PROJECT_DIRECTORY>
```


### 3. Set Up a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```
### 4. Install Dependencies

Create a `requirements.txt` file in your project root with the following content:

```text
streamlit
pandas
Pillow
PyPDF2
PyMuPDF
pytesseract
requests
langchain-google-genai
langchain
langgraph
tavily-python
pip install -r requirements.txt
```
