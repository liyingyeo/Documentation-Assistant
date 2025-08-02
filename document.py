
import os
import io
import sys
import json
import requests
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai
import anthropic
from IPython.display import Markdown, display, update_display
import gradio as gr
import subprocess

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')
os.environ['ANTHROPIC_API_KEY'] = os.getenv('ANTHROPIC_API_KEY', 'your-key-if-not-using-env')
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN', 'your-key-if-not-using-env')
os.environ['GEMINI_API_KEY'] = os.getenv('GOOGLE_API_KEY', 'your-key-if-not-using-env')


openai = OpenAI()
claude = anthropic.Anthropic()
google.generativeai.configure(api_key=os.getenv('GEMINI_API_KEY'))

OPENAI_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
GEMINI_MODEL = "gemini-2.0-flash-exp"
LLAMA_MODEL = "llama3-8b-8192"

def system_message():
    system_message = "You are a Python code documentation assistant. Your task is to analyze Python code and generate high-quality, concise comments and docstrings that enhance code readability and maintainability. Follow these guidelines:"
    system_message += "\n\nDocstrings:"
    system_message += "\n- Add comprehensive docstrings for every function, class, and module using Google docstring format"
    system_message += "\n- Include purpose, parameters (with types), return values (with types), and exceptions raised where applicable"
    system_message += "\n- For classes, document the class purpose and key attributes"
    system_message += "\n- Keep descriptions concise but informative, focusing on 'what' and 'why' rather than 'how'"
    system_message += "\n- Use proper type hints in conjunction with docstrings"

    system_message += "\n\nInline Comments:"
    system_message += "\n- Add strategic inline comments for complex algorithms, business logic, or non-intuitive operations"
    system_message += "\n- Explain the reasoning behind important decisions or workarounds"
    system_message += "\n- Clarify magic numbers, complex regex patterns, or unusual data transformations"
    system_message += "\n- Avoid commenting on self-evident operations unless they serve a larger purpose"
    system_message += "\n- Place comments above the relevant code block or at the end of the line for brief clarifications"

    system_message += "\n\nCode Quality:"
    system_message += "\n- Ensure all original functionality remains unchanged"
    system_message += "\n- Add type hints where missing to improve code clarity"
    system_message += "\n- Maintain consistent formatting and style throughout"
    system_message += "\n- Follow PEP 8, PEP 257, and PEP 484 standards"

    system_message += "\n\nOutput Requirements:"
    system_message += "\n- Return only the enhanced Python code with added documentation"
    system_message += "\n- Do not include explanations or summaries outside the code"
    system_message += "\n- Maintain the original code structure and logic exactly"
    system_message += "\n- Target intermediate Python developers for documentation clarity"

    return system_message

def user_prompt_for(python):
    user_prompt = "Analyze the following Python code and enhance it by adding high-quality, concise docstrings and comments. "
    user_prompt += "Ensure all functions, classes, and modules have appropriate docstrings describing their purpose, parameters, and return values. "
    user_prompt += "Add inline comments only for complex or non-obvious parts of the code. "
    user_prompt += "Follow Python's PEP 257 and PEP 8 standards for documentation and formatting. "
    user_prompt += "Do not modify the code itself; only add annotations.\n\n"
    user_prompt += python
    return user_prompt

def messages_for(python):
    return [
        {"role": "system", "content": system_message()},
        {"role": "user", "content": user_prompt_for(python)}
    ]

def stream_output(input_code: str) -> None:
    # Remove markdown code blocks and common explanatory text in one pass
    unwanted_patterns = [
        "```python",
        "```",
        "Here's the enhanced Python code with added docstrings and comments:",
        "Here's the Python code with added docstrings and comments:",
        "Enhanced Python code:"
    ]
    
    cleaned_code = input_code
    for pattern in unwanted_patterns:
        cleaned_code = cleaned_code.replace(pattern, "")
    
    # Strip leading/trailing whitespace and normalize line endings
    cleaned_code = cleaned_code.strip()  
    return cleaned_code


pi = """
import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.time()
result = calculate(100_000_000, 4, 1) * 4
end_time = time.time()

print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
"""


gemini = google.generativeai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    system_instruction=system_message()
)

def stream_gpt(python):    
    stream = openai.chat.completions.create(model=OPENAI_MODEL, messages=messages_for(python), stream=True)
    reply = ""
    for chunk in stream:
        fragment = chunk.choices[0].delta.content or ""
        reply += fragment
        yield stream_output(reply)

def stream_claude(python):
    result = claude.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        system=system_message(),
        messages=[{"role": "user", "content": user_prompt_for(python)}],
    )
    reply = ""
    with result as stream:
        for text in stream.text_stream:
            reply += text
            yield stream_output(reply)

def stream_gemini(python):
    result = gemini.generate_content(
        user_prompt_for(python),
        generation_config={
            "max_output_tokens": 2000,
        },
        stream=True,
    )
    reply = "" 

    for chunk in result:
            text = chunk.text
            reply += text
            yield stream_output(reply)
   

def document(python, model):
    if model=="GPT":
        result = stream_gpt(python)
    elif model=="Claude":
        result = stream_claude(python)
    elif model=="Gemini":
        result = stream_gemini(python)
    else:
        raise ValueError("Unknown model")
    for stream_so_far in result:
        yield stream_so_far    

def main():
    with gr.Blocks() as ui:
        with gr.Row():
            python = gr.Textbox(label="Python code:", lines=20, value=pi)
            cpp = gr.Textbox(label="Documented code:", lines=20)
        with gr.Row():
            model = gr.Dropdown(["GPT", "Claude", "Gemini"], label="Select model", value="GPT")
            convert = gr.Button("Document code")

        convert.click(document, inputs=[python, model], outputs=[cpp])

    ui.launch(inbrowser=True)

if __name__ == "__main__":
    main()