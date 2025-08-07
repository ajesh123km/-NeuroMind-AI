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

def call_gemini_api(prompt, response_schema=None):
    """
    Helper function to call the Gemini 2.0 Flash API.

    Args:
        prompt (str): The prompt to send to the model.
        response_schema (dict, optional): A JSON schema for structured responses. Defaults to None.

    Returns:
        dict: The JSON response from the API, or None if an error occurs.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyDP8jweZagbjeU73wU87F7aZ59lUGCRKcI")
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    chat_history = [{"role": "user", "parts": [{"text": prompt}]}]
    payload = {"contents": chat_history}

    if response_schema:
        payload["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }

    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        return result
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Gemini API: {e}")
        return None
    except json.JSONDecodeError:
        print("Error: Could not parse JSON response from Gemini API.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during API call: {e}")
        return None

def generate_questions(text, num_questions=3):
    """
    Generates questions from the given text using the Gemini API.

    Args:
        text (str): The text from which to generate questions.
        num_questions (int): The number of questions to generate.

    Returns:
        list: A list of generated questions (strings), or an empty list if an error occurs.
    """
    print(f"Generating {num_questions} questions from the text...")
    prompt = f"From the following text, generate {num_questions} factual questions that can be answered directly from the text. Provide the questions as a JSON array of strings. Ensure the questions are clear and unambiguous.\n\nText:\n{text}"

    # Define the JSON schema for the expected response
    response_schema = {
        "type": "ARRAY",
        "items": {
            "type": "STRING"
        }
    }

    result = call_gemini_api(prompt, response_schema=response_schema)

    if result and result.get("candidates") and len(result["candidates"]) > 0 and \
       result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts") and \
       len(result["candidates"][0]["content"]["parts"]) > 0:
        try:
            # The API returns a string that needs to be parsed as JSON
            questions_json_str = result["candidates"][0]["content"]["parts"][0].get("text", "[]")
            questions = json.loads(questions_json_str)
            if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                return questions
            else:
                print("Warning: Generated questions are not in the expected list of strings format.")
                return []
        except json.JSONDecodeError:
            print("Error: Could not parse generated questions as JSON.")
            return []
    else:
        print("Error: No questions generated or unexpected API response structure.")
        return []

def evaluate_answer(original_text, question, user_answer):
    """
    Evaluates a user's answer against the original text using the Gemini API.

    Args:
        original_text (str): The full original text from the PDF.
        question (str): The question asked.
        user_answer (str): The user's provided answer.

    Returns:
        dict: A dictionary containing 'score' (0-100), 'feedback' (textual), and 'correct_answer' (brief correct answer),
              or None if an error occurs.
    """
    print(f"Evaluating answer for question: '{question}'...")
    prompt = f"""
    You are an AI assistant evaluating a user's answer to a question based on a provided text.

    Original Text:
    {original_text}

    Question:
    {question}

    User's Answer:
    {user_answer}

    Evaluate the user's answer for accuracy and completeness based *only* on the information in the Original Text.
    Provide your evaluation as a JSON object with the following keys:
    - "score": An integer from 0 to 100, indicating how accurate and complete the user's answer is.
    - "feedback": A concise textual feedback explaining why the score was given, pointing out strengths and weaknesses, and suggesting improvements.
    - "correct_answer": A brief, accurate answer to the question based on the Original Text.
    """

    # Define the JSON schema for the expected response
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "score": {"type": "INTEGER"},
            "feedback": {"type": "STRING"},
            "correct_answer": {"type": "STRING"}
        },
        "required": ["score", "feedback", "correct_answer"]
    }

    result = call_gemini_api(prompt, response_schema=response_schema)

    if result and result.get("candidates") and len(result["candidates"]) > 0 and \
       result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts") and \
       len(result["candidates"][0]["content"]["parts"]) > 0:
        try:
            # The API returns a string that needs to be parsed as JSON
            feedback_json_str = result["candidates"][0]["content"]["parts"][0].get("text", "{}")
            feedback_data = json.loads(feedback_json_str)
            if all(k in feedback_data for k in ["score", "feedback", "correct_answer"]):
                return feedback_data
            else:
                print("Warning: Generated feedback is not in the expected JSON format.")
                return None
        except json.JSONDecodeError:
            print("Error: Could not parse generated feedback as JSON.")
            return None
    else:
        print("Error: No evaluation feedback generated or unexpected API response structure.")
        return None

if __name__ == "__main__":
    # IMPORTANT: Replace 'path/to/your/document.pdf' with the actual path to your PDF file
    # For Windows paths, use a raw string (r"...") or double backslashes ("\\")
    pdf_file_path = "C:/Impact-a-thon/backend4/today.pdf"

    print(f"--- PDF Question-Answer Evaluator ---")
    print(f"Attempting to extract text from: {pdf_file_path}")

    extracted_text = extract_text_from_pdf(pdf_file_path)

    if not extracted_text:
        print("Failed to extract text from PDF. Exiting.")
    else:
        print("\nText extracted successfully. Generating questions...")
        questions = generate_questions(extracted_text, num_questions=3) # You can change num_questions

        if not questions:
            print("Failed to generate questions. Exiting.")
        else:
            print("\nQuestions generated. Let's start the quiz!")
            for i, question in enumerate(questions):
                print(f"\n--- Question {i+1} ---")
                print(f"Q: {question}")
                user_answer = input("Your Answer: ").strip()

                if not user_answer:
                    print("You didn't provide an answer. Skipping evaluation for this question.")
                    continue

                feedback = evaluate_answer(extracted_text, question, user_answer)

                if feedback:
                    print("\n--- Feedback ---")
                    print(f"Score: {feedback.get('score', 'N/A')}/100")
                    print(f"Feedback: {feedback.get('feedback', 'No feedback provided.')}")
                    print(f"Correct Answer: {feedback.get('correct_answer', 'Not available.')}")
                else:
                    print("Could not get evaluation feedback for this answer.")
            print("\n--- Quiz Finished ---")
