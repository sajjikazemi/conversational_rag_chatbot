# LLM & AI Application Development Examples

This repository serves as a portfolio showcasing a variety of projects built using Large Language Models (LLMs) and related AI technologies. It demonstrates practical applications ranging from conversational AI and specialized prediction tasks to complex agent systems and multi-modal interactions.

## Overview

The projects included explore several key areas of modern AI development:

1.  **Retrieval-Augmented Generation (RAG):** Building systems that leverage external knowledge bases to enhance LLM responses, implemented using LangChain and custom agent approaches.
2.  **LLM Fine-tuning:** Adapting pre-trained LLMs (both closed-source via API and open-source like Llama 3) for specific tasks like price estimation, utilizing techniques like PEFT/QLoRA for efficiency.
3.  **Multi-Agent Systems:** Designing collaborative systems where multiple specialized agents work together, combining outputs using ensemble methods.
4.  **Multi-modal AI:** Creating applications that understand and generate multiple types of data, including text, images (DALL-E 3), and audio (Whisper ASR, TTS).
5.  **LLM Tool Use / Function Calling:** Extending LLM capabilities by allowing them to interact with external tools and APIs.
6.  **Local LLM Inference:** Running powerful open-source LLMs (like quantized Llama 3.1) directly on local hardware for tasks like summarization.

## Projects Included

*(Organize this list based on your repository structure, e.g., subdirectories)*

* **Conversational RAG Chatbots:**
    * Uses LangChain, OpenAI, Gradio.
    * Demonstrates use of ChromaDB and FAISS vector stores.
* **LLM Fine-tuning for Price Estimation:**
    * Uses OpenAI Fine-tuning API (`gpt-4o-mini`).
    * Uses Hugging Face `transformers`, `peft`, `trl` for QLoRA fine-tuning of Llama 3.
    * Includes evaluation scripts and WandB integration.
* **Multi-Agent System for Price Estimation:**
    * Features collaborating agents (`FrontierAgent` using RAG, `EnsembleAgent` using `scikit-learn`).
    * Integrates RAG, potentially fine-tuned models, and traditional ML.
* **Multi-modal Airline Assistant (FlightAI):**
    * Showcases OpenAI Function Calling for data retrieval.
    * Integrates DALL-E 3 for image generation and OpenAI TTS for speech output.
    * Uses Gradio `Blocks` for a custom UI.
* **Meeting Minutes Generation (ASR + Local LLM):**
    * Pipeline using OpenAI Whisper API for audio transcription.
    * Generates structured markdown minutes using a locally run, quantized Llama 3.1 8B Instruct model via Hugging Face `transformers`.

## Key Skills Demonstrated

This collection of projects highlights proficiency across the AI development lifecycle:

* **LLM Techniques:** RAG, Fine-tuning (API & PEFT/QLoRA), Tool Use/Function Calling, Prompt Engineering, Local Inference, Quantization.
* **AI Architectures:** Chatbots, Multi-Agent Systems, Multi-modal Apps, Processing Pipelines.
* **AI Modalities:** Text, Image, Audio (ASR & TTS).
* **ML Concepts:** Ensemble Methods, Regression, Evaluation Metrics.
* **Frameworks/Libraries:** LangChain, Hugging Face (`transformers`, `datasets`, `peft`, `trl`, `hub`), Sentence Transformers, Scikit-learn, Joblib, Pandas, Gradio, OpenAI SDK, PIL, Pydub.
* **Supporting Tech:** Vector DBs (ChromaDB, FAISS), Python (OOP, APIs, Data Handling), API Integration, MLOps Concepts (Data Prep, Tracking, Resource Mgmt).

## Setup & Usage (General Guidance)

*(This section MUST be customized based on your specific project structure, dependencies, and execution methods)*

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/sajjikazemi/llm_practice.git
    cd llm_practice
    ```
2.  **Environment Setup:**
    * Create and activate Python virtual environments (e.g., `python -m venv venv && source venv/bin/activate`). Consider separate environments if dependency conflicts arise between projects.
3.  **Install Dependencies:**
    * Install required packages using `pip install -r requirements.txt`. You may need multiple `requirements.txt` files (e.g., `requirements-rag.txt`, `requirements-finetune.txt`, `requirements-local-llm.txt`) depending on your project organization. Ensure all necessary libraries listed under "Frameworks/Libraries" above are included.
    * **Note:** For audio playback (`pydub`), you might need `ffmpeg`. For local LLM inference (`transformers` with quantization), ensure you have compatible hardware (GPU strongly recommended) and necessary drivers (e.g., CUDA).
4.  **API Keys & Configuration:**
    * Create a `.env` file in the root or relevant project directories.
    * Add your API keys: `OPENAI_API_KEY`, `HF_TOKEN` (for Hugging Face Hub login/downloads), `WANDB_API_KEY` (if using Weights & Biases).
5.  **Data & Models:**
    * Place knowledge base documents, training data (`.pkl`, datasets), audio files (`.mp3`), or pre-trained models (`.pkl`) in the expected locations for each project.
    * Ensure vector database setup (e.g., ChromaDB persistence directory) is handled correctly.
6.  **Run Projects:**
    * Execute the main Python script for the desired project (e.g., `python run_rag_project.py`, `python run_local_summarizer.py`).
    * Refer to comments within the code or specific documentation you add for detailed instructions for each project.
