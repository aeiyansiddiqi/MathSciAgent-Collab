from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TokenTextSplitter
from PyPDF2 import PdfReader
import copy
import re
from IPython.display import display, Markdown
import time
import os
import json


import os
import json
import time
from langchain.text_splitter import TokenTextSplitter
import ollama

# === CONFIGURATION ===
INPUT_FOLDER = "C:/Users/Admin/Desktop/Samaneh_Proj/converted_pdf_01"
OUTPUT_FOLDER = "C:/Users/Admin/Desktop/Samaneh_Proj/distillated_chunks_02"
chunk_size = 850
chunk_overlap = 120

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === HELPER FUNCTIONS ===
def call_mistral(chunk):
    system_prompt = """You are a scientific assistant helping to create a structured knowledge graph.
Your outputs must be self-contained, complete, and rich in technical content.
You must never reference specific names, figures, plots, tables, or citations.
Focus purely on abstract scientific facts and reasoning.
Output only a JSON object.
"""
    user_prompt = f"""
Given this scientific text:
\"\"\"{chunk}\"\"\"

Perform the following tasks and return a single JSON object:

1. Write a comprehensive scientific summary of approximately 150–200 words.
   - Ensure it includes essential background, methodology, and findings.
   - Express logical and technical reasoning clearly.
   - The summary must be self-contained and readable without external references.

2. Extract a list of bullet points based on the summary:
   - Each bullet must express a **single, specific, and self-contained** scientific idea.
   - Use precise terminology, and explain technical relationships when relevant.
   - Include **as many bullets as necessary** to cover the core content thoroughly.

3. Generate a brief, standalone scientific title that reflects the core concept or contribution of the text.

Return the result in the following JSON format **without any extra text, markdown, or comments**:

{{
  "title": "<your generated title>",
  "summary": "<your rewritten scientific summary>",
  "bullet_points": [
    "<bullet point 1>",
    "<bullet point 2>",
    "... (continue as needed)"
  ]
}}
"""
    response = ollama.chat(model='mistral-openorca', messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    return response['message']['content'].strip()

def safe_json_loads(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[6:].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        text = text.replace('\n', ' ').replace('\r', '').strip()
        return json.loads(text)

# === MAIN DISTILLATION LOOP ===
for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".mmd"):
        paper_id = os.path.splitext(filename)[0]
        print(f"\n\n--- Processing: {filename} ---")

        with open(os.path.join(INPUT_FOLDER, filename), "r", encoding="utf-8") as file:
            raw_text = file.read()

        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_text(raw_text)

        paper_output_path = os.path.join(OUTPUT_FOLDER, f"{paper_id}.json")
        distillated_chunks = []

        for i, chunk in enumerate(chunks):
            try:
                print(f"Processing chunk {i+1}/{len(chunks)}...")
                start_time = time.time()
                result_str = call_mistral(chunk)
                result_json = safe_json_loads(result_str)

                output = {
                    "chunk_index": i + 1,
                    "title": result_json.get("title", "").strip(),
                    "summary": result_json.get("summary", "").strip(),
                    "bullet_points": result_json.get("bullet_points", []),
                }

                distillated_chunks.append(output)

                duration = round(time.time() - start_time, 2)
                print(f"✓ Done (took {duration}s) — Title: {output['title']}")

            except Exception as e:
                print(f"✗ Error processing chunk {i+1}: {e}")
                with open("error_chunks.txt", "a", encoding="utf-8") as error_file:
                    error_file.write(f"\n\n{paper_id} - Chunk {i+1}:\n{chunk}\n---\n")

        with open(paper_output_path, "w", encoding="utf-8") as f_out:
            json.dump({"paper_id": paper_id, "chunks": distillated_chunks}, f_out, ensure_ascii=False, indent=2)

#######################################################################################
#####################################################################################################
####################################################################################################
#first Code

# # Initialize the splitter
# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=chunk_size,
#     chunk_overlap=chunk_overlap,
#     length_function=len,
#     is_separator_regex=False,
# )

# # Read the .mmd file into the txt variable (example path)
# with open("C:/Users/Admin/Desktop/Samaneh_Proj/converted_pdf/LPSF.mmd", "r") as file:
#     txt = file.read()

# # Split the text into chunks
# pages = splitter.split_text(txt)

# # Print the number of chunks
# print("Number of chunks = ", len(pages))

# # Optionally, display the first chunk
# from IPython.display import display, Markdown
# display(Markdown(pages[0]))



# ############################################################################
# # Assuming you have the Mistral-7B-OpenOrca model set up
# import ollama

# def call_model(prompt):
#     response = ollama.chat(model='mistral-openorca', messages=[
#         {"role": "system", "content": prompt[0]},
#         {"role": "user", "content": prompt[1]}
#     ])
#     return response['message']['content']


# def generate_summary_and_insights(chunk):
#     # Prompts
#     summary_system = "You respond with a concise scientific summary, you never use name or references."
#     summary_user = f"In a matter of fact voice, rewrite this \"{chunk}\". The writing must stand on its own and provide all background needed and include details. Do not include names, figures, plots or citations in your response, only facts."
    
#     summary = call_model((summary_system, summary_user))

#     bullet_points_user = f"Provide a bullet point list of the key facts and reasoning in \"{summary}\". The writing must stand on its own and provide all background needed, and include details. Do not include figures, plots, or citations in your response. Think step by step."
#     bullet_points = call_model((summary_system, bullet_points_user))

#     title_system = "You are a scientist who writes a scientific paper. You never use names or citations."
#     title_user = f"Provide a one-sentence title of this text: \"{summary}\". Make sure the title can be understood fully without any other context. Do not use the word 'title', just provide the answer."
#     title = call_model((title_system, title_user))

#     return summary, bullet_points, title

# output_path = "C:/Users/Admin/Desktop/Samaneh_Proj/distillated_chunks/output.txt"

# # Open the file in write mode with UTF-8 encoding
# with open(output_path, "w", encoding="utf-8") as file:
#     for i, chunk in enumerate(pages):
#         summary, bullet_points, title = generate_summary_and_insights(chunk)

#         # Format the output
#         output = f"Chunk {i+1}\n"
#         output += f"Title: {title}\n"
#         output += f"Summary:\n{summary}\n"
#         output += f"Bullet Points:\n{bullet_points}\n"
#         output += "-" * 80 + "\n"

#         # Print to console
#         print(output)

#         # Write to file
#         file.write(output)