import PyPDF2
import requests
import json
import os

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The extracted text from the PDF, or an empty string if an error occurs.
    """
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() or "" # Add extracted text, handle None case
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
        return ""
    except Exception as e:
        print(f"An error occurred while extracting text from PDF: {e}")
        return ""
    return text

def summarize_text_with_gemini(text_to_summarize, summary_type="medium"):
    """
    Summarizes the given text using the Gemini 2.0 Flash API,
    with a specified summary depth.

    Args:
        text_to_summarize (str): The text to be summarized.
        summary_type (str): Desired summary depth ('quick', 'medium', 'detailed').

    Returns:
        str: The summarized text, or an error message if the API call fails.
    """
    if not text_to_summarize.strip():
        return "Error: No text provided for summarization."

    # The API key is automatically provided by the Canvas environment when left as an empty string.
    # In a real backend environment, you would load this from an environment variable or a secure config.
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyBHoiIctFOUYCgKeWRSUa43pmRtiO90Qzs") # Use environment variable for API key
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    # Adjust prompt based on summary type
    if summary_type.lower() == "quick":
        prompt_instruction = "Summarize the following text very briefly, highlighting only the main points:"
    elif summary_type.lower() == "detailed":
        prompt_instruction = "Provide a detailed summary of the following text, covering all key aspects and supporting details:"
    else: # Default to medium
        prompt_instruction = "Summarize the following text concisely and accurately:"

    prompt = f"{prompt_instruction}\n\n{text_to_summarize}"
    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]

    payload = {"contents": chat_history}

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        result = response.json()

        if result.get("candidates") and len(result["candidates"]) > 0 and \
           result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts") and \
           len(result["candidates"][0]["content"]["parts"]) > 0:
            summary = result["candidates"][0]["content"]["parts"][0].get("text", "No summary text found.")
            return summary
        else:
            return "Error: No summary found in the API response. Please try again or provide more text."

    except requests.exceptions.RequestException as e:
        return f"Error connecting to Gemini API: {e}"
    except json.JSONDecodeError:
        return "Error: Could not parse JSON response from Gemini API."
    except Exception as e:
        return f"An unexpected error occurred during summarization: {e}"

if __name__ == "__main__":
    # Example usage:
    # Replace 'path/to/your/document.pdf' with the actual path to your PDF file
    pdf_file_path = r"C:/Impact-a-thon/backend4/today.pdf" # IMPORTANT: Change this to your PDF file path

    print(f"Attempting to extract text from: {pdf_file_path}")
    extracted_text = extract_text_from_pdf(pdf_file_path)

    if extracted_text:
        while True:
            user_summary_type = input("Enter desired summary type (quick, medium, detailed): ").lower().strip()
            if user_summary_type in ["quick", "medium", "detailed"]:
                break
            else:
                print("Invalid input. Please choose 'quick', 'medium', or 'detailed'.")

        print(f"\nText extracted successfully. Sending to Gemini for {user_summary_type} summarization...")
        summary = summarize_text_with_gemini(extracted_text, user_summary_type)
        print("\n--- Summary ---")
        print(summary)
    else:
        print("Could not extract text from the PDF. Summary generation aborted.")
