# 🧠 Python Code Documentation Assistant

This project is a multi-LLM-powered Python code documentation assistant that analyzes Python code and generates high-quality docstrings and inline comments. It supports multiple large language models (LLMs) from OpenAI (GPT), Anthropic (Claude), and Google (Gemini).

## ✨ Features

- Generates Google-style docstrings and meaningful inline comments.
- Supports streaming responses from:
  - OpenAI GPT (`gpt-4o`)
  - Claude (`claude-3-5-sonnet-20240620`)
  - Gemini (`gemini-2.0-flash-exp`)
- Simple Gradio-based web interface for code input and output.
- Follows PEP 8, PEP 257, and PEP 484 standards.

## 🚀 Installation

### 1. Clone the repository

### 2. Install dependencies

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

### 3. Set your API keys
Create a .env file in the root directory with the following variables:

OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-gemini-api-key