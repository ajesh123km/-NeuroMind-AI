import PyPDF2
import json
import os
import re
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# --- Gemini API Configuration ---
api_key_from_env = os.environ.get("GEMINI_API_KEY", "AIzaSyBHoiIctFOUYCgKeWRSUa43pmRtiO90Qzs")
if not api_key_from_env:
    print("CRITICAL ERROR: GEMINI_API_KEY environment variable not set.")
    print("Please set it to your actual Gemini API Key and restart your terminal/IDE.")
    exit()

genai.configure(api_key=api_key_from_env)

def call_gemini_api(prompt, response_schema=None, file_part=None):
    """Calls Gemini for content generation, optionally with file input (PDF)."""
    model_name = 'gemini-2.0-flash'
    model = genai.GenerativeModel(model_name)

    contents = []
    if file_part:
        contents.append(file_part)  # file_part is a dict as explained below
    contents.append({"text": prompt})

    generation_config_args = {}
    if response_schema:
        generation_config_args["response_mime_type"] = "application/json"
        generation_config_args["response_schema"] = response_schema
        generation_config_args["temperature"] = 0.3
        generation_config_args["top_p"] = 0.9
        generation_config_args["top_k"] = 32

    generation_config = genai.GenerationConfig(**generation_config_args) if generation_config_args else None

    try:
        response = model.generate_content(
            contents,
            generation_config=generation_config,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        if response_schema:
            try:
                parsed_json = json.loads(response.text)
                return parsed_json
            except Exception:
                print(f"Error: Could not parse JSON response from API: {response.text[:200]}...")
                return None
        else:
            return {"candidates": [{"content": {"parts": [{"text": response.text}]}}]}
    except Exception as e:
        print(f"An error occurred during Gemini API call: {e}")
        if hasattr(e, 'response') and hasattr(e.response, 'text'):
            print(f"API Error Response: {e.response.text}")
        return None

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF using PyPDF2."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or ""
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return ""
    except Exception as e:
        print(f"An error occurred while extracting text from PDF: {e}")
        return ""
    return text

def classify_student(marks):
    """Classifies student performance based on marks."""
    if marks < 50:
        return "Needs Improvement"
    elif 50 <= marks < 70:
        return "Average"
    elif 70 <= marks < 85:
        return "Good"
    elif 85 <= marks < 95:
        return "Excellent"
    else:
        return "Advanced Learner"

def get_pdf_content_parts(pdf_path):
    """Loads PDF file as bytes and prepares it as Gemini input (dict)."""
    try:
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        return {"mime_type": "application/pdf", "data": pdf_bytes}
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return None
    except Exception as e:
        print(f"An error occurred while preparing PDF for Gemini: {e}")
        return None

def extract_titles_from_pdf_multimodal(pdf_file_part):
    """Extracts main section titles/headings from PDF via Gemini multimodal."""
    print("\nExtracting main titles/sections from the document using Gemini's multimodal PDF understanding...")
    prompt = (
        "Analyze the provided PDF document. Identify and list all the main section titles or prominent headings that logically structure the content, similar to a Table of Contents. "
        "Exclude page numbers, header/footer text, and very short phrases that are not true section headings. "
        "Return the output as a JSON array of strings. Ensure titles are exactly as they appear in the document."
    )
    response_schema = {
        "type": "ARRAY",
        "items": {
            "type": "STRING",
        }
    }
    result_json = call_gemini_api(prompt, response_schema=response_schema, file_part=pdf_file_part)
    if result_json and isinstance(result_json, list):
        return [t.strip() for t in result_json if isinstance(t, str) and len(t.strip()) > 5]
    else:
        print("Error: No titles extracted or unexpected JSON structure from multimodal call.")
        return []

def get_text_for_topic(full_text, selected_topic, all_titles):
    """Finds text section within full_text corresponding to selected_topic, using all_titles as possible boundaries."""
    if not selected_topic or not all_titles or not full_text:
        return full_text
    # Try to match start (case-insensitive, allows some whitespace)
    start_pattern = r'(?:^|\n|\r\n)\s*' + re.escape(selected_topic.strip()) + r'[\s\.]*'
    start_match = re.search(start_pattern, full_text, re.IGNORECASE | re.MULTILINE)
    if not start_match:
        print(f"Warning: Could not find distinct start of '{selected_topic}' in the text. Summarizing entire document.")
        return full_text
    start_idx_content = start_match.end()
    # Find the next title in all_titles AFTER selected_topic
    end_idx_content = len(full_text)
    current_topic_found = False
    for i, title in enumerate(all_titles):
        if title.strip().lower() == selected_topic.strip().lower():
            current_topic_found = True
        elif current_topic_found:
            # Find this (next title) in the text after current topic
            next_topic_pattern = r'(?:^|\n|\r\n)\s*' + re.escape(title.strip()) + r'[\s\.]*'
            next_topic_match = re.search(next_topic_pattern, full_text[start_idx_content:], re.IGNORECASE | re.MULTILINE)
            if next_topic_match:
                end_idx_content = start_idx_content + next_topic_match.start()
                break
    extracted_section = full_text[start_idx_content:end_idx_content].strip()
    if len(extracted_section) < 50 and len(full_text) > 100:
        print(f"Warning: Extracted section for '{selected_topic}' is very short ({len(extracted_section)} chars). Falling back to summarizing entire document.")
        return full_text
    return extracted_section

def summarize_text_with_gemini(text_to_summarize, summary_type="medium", student_classification=None):
    """Summarizes the text with Gemini, possibly customized for student level."""
    if not text_to_summarize.strip():
        return "Error: No text provided for summarization."
    if summary_type.lower() == "quick":
        base_prompt_instruction = "Summarize the following text very briefly, highlighting only the main points."
    elif summary_type.lower() == "detailed":
        base_prompt_instruction = "Provide a detailed summary of the following text, covering all key aspects and supporting details."
    else:
        base_prompt_instruction = "Summarize the following text concisely and accurately."
    prompt_instruction = base_prompt_instruction
    if student_classification:
        if student_classification == "Needs Improvement":
            prompt_instruction = f"{base_prompt_instruction} Make sure the language is very simple, easy to understand, and avoid complex jargon. Focus on the core concepts."
        elif student_classification == "Average":
            prompt_instruction = f"{base_prompt_instruction} Use clear and straightforward language. Explain any potentially difficult terms briefly."
        elif student_classification == "Good":
            prompt_instruction = f"{base_prompt_instruction} Use standard academic language, providing good detail and clarity."
        elif student_classification == "Excellent":
            prompt_instruction = f"{base_prompt_instruction} Use precise and possibly technical terminology where appropriate. Assume a strong understanding of the subject matter."
        elif student_classification == "Advanced Learner":
            prompt_instruction = f"{base_prompt_instruction} Provide a highly detailed and nuanced summary, using advanced terminology and potentially exploring subtle implications or connections. Assume a deep understanding of the subject."
    prompt = f"{prompt_instruction}\n\n{text_to_summarize}"
    result_dict = call_gemini_api(prompt)
    if result_dict and result_dict.get("candidates") and len(result_dict["candidates"]) > 0 and \
       result_dict["candidates"][0].get("content") and result_dict["candidates"][0]["content"].get("parts") and \
       len(result_dict["candidates"][0]["content"]["parts"]) > 0:
        summary = result_dict["candidates"][0]["content"]["parts"][0].get("text", "No summary text found.")
        return summary
    else:
        return "Error: No summary found in the API response. Please try again or provide more text."

if __name__ == "__main__":
    # ------- Set PDF path here -------
    pdf_file_path = r"C:\Users\jazil\Desktop\yourfile2.pdf"
    print(f"--- PDF Study Assistant ---")
    print(f"Attempting to extract text from: {pdf_file_path}")

    extracted_text = extract_text_from_pdf(pdf_file_path)
    pdf_file_part = get_pdf_content_parts(pdf_file_path)

    if not extracted_text:
        print("Failed to extract text from PDF. Exiting.")
        exit()

    if not pdf_file_part:
        print("Warning: Could not prepare PDF for multimodal analysis (direct PDF upload). Heading extraction might be less accurate.")
        extracted_titles = []
    else:
        extracted_titles = extract_titles_from_pdf_multimodal(pdf_file_part)

    summarize_for_text = extracted_text

    if extracted_titles:
        print("\n--- Document Sections ---")
        print("0. Summarize Entire Document")
        for i, title in enumerate(extracted_titles):
            print(f"{i+1}. {title}")
        print("\n--------------------------")
        while True:
            topic_choice = input("Enter the number of the topic to summarize, or '0' for the entire document: ").strip()
            try:
                topic_idx = int(topic_choice)
                if topic_idx == 0:
                    print("You chose to summarize the entire document.")
                    break
                elif 1 <= topic_idx <= len(extracted_titles):
                    selected_topic_title = extracted_titles[topic_idx - 1]
                    print(f"You chose to summarize: '{selected_topic_title}'")
                    temp_text = get_text_for_topic(extracted_text, selected_topic_title, extracted_titles)
                    if temp_text and len(temp_text.strip()) > 50:
                        summarize_for_text = temp_text
                    else:
                        print(f"Could not isolate text for '{selected_topic_title}' effectively. Summarizing the entire document instead.")
                        summarize_for_text = extracted_text
                    break
                else:
                    print("Invalid choice. Please enter a valid number from the list.")
            except ValueError:
                print("Invalid input. Please enter a number.")
    else:
        print("Could not extract distinct titles from the PDF. This might happen with scanned PDFs or complex layouts. Will proceed with summarizing the entire document.")

    # --- Choose summary depth and personalization ---
    selected_summary_type = None
    while selected_summary_type not in ["quick", "medium", "detailed"]:
        selected_summary_type = input("Enter desired summary depth (quick, medium, detailed): ").lower().strip()
        if selected_summary_type not in ["quick", "medium", "detailed"]:
            print("Invalid input. Please choose 'quick', 'medium', or 'detailed'.")

    student_classification_for_summary = None
    while True:
        personalization_choice = input("Do you want a personalized summary based on marks (for learning)? (yes/no/default): ").lower().strip()
        if personalization_choice == "yes":
            while True:
                try:
                    student_marks_str = input("Enter your marks (e.g., 75): ").strip()
                    student_marks = float(student_marks_str)
                    if 0 <= student_marks <= 100:
                        student_classification_for_summary = classify_student(student_marks)
                        print(f"Summary will be tailored for a '{student_classification_for_summary}' learner.")
                        break
                    else:
                        print("Marks should be between 0 and 100.")
                except ValueError:
                    print("Invalid input. Please enter a number for marks.")
            break
        elif personalization_choice == "no":
            print("Generating a general summary (not personalized).")
            break
        elif personalization_choice == "default":
            print("Generating a default summary (not personalized).")
            break
        else:
            print("Invalid choice. Please enter 'yes', 'no', or 'default'.")

    print(f"\nSending text to Gemini for {selected_summary_type} summarization...")
    summary_result = summarize_text_with_gemini(
        summarize_for_text,
        summary_type=selected_summary_type,
        student_classification=student_classification_for_summary
    )

    print("\n--- Generated Summary ---")
    print(summary_result)
    print("\n--------------------------")
    print("\n--- Program Finished ---")
