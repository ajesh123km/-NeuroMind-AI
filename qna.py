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
    api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyBHoiIctFOUYCgKeWRSUa43pmRtiO90Qzs")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        print("Please set your Gemini API key before running the script.")
        return None

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
        print(f"Response content: {response.text if 'response' in locals() else 'N/A'}")
        return None
    except json.JSONDecodeError:
        print("Error: Could not parse JSON response from Gemini API.")
        print(f"Raw response: {response.text if 'response' in locals() else 'N/A'}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during API call: {e}")
        return None

def generate_questions(text, num_questions=3, question_type="open-ended"):
    """
    Generates questions from the given text using the Gemini API, with specified types.

    Args:
        text (str): The text from which to generate questions.
        num_questions (int): The number of questions to generate.
        question_type (str): The type of questions to generate (mcq, true/false, fill-in-the-blanks, open-ended, mixed).

    Returns:
        list: A list of generated questions (dictionaries for structured types, strings for open-ended),
              or an empty list if an error occurs.
    """
    print(f"Generating {num_questions} {question_type} questions from the text...")

    prompt = ""
    response_schema = None

    if question_type == "mcq":
        prompt = f"""
        From the following text, generate {num_questions} Multiple Choice Questions (MCQ). Each question should have 4 options (A, B, C, D) and indicate the correct answer.
        Provide the output as a JSON array of objects, where each object has "question", "options" (an array of strings), and "correct_answer" (the letter A, B, C, or D).

        Text:
        {text}
        """
        response_schema = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "question": {"type": "STRING"},
                    "options": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "minItems": 4,
                        "maxItems": 4
                    },
                    "correct_answer": {"type": "STRING"}
                },
                "required": ["question", "options", "correct_answer"]
            }
        }
    elif question_type == "true/false":
        prompt = f"""
        From the following text, generate {num_questions} True/False questions.
        Provide the output as a JSON array of objects, where each object has "statement" and "correct_answer" (boolean: true or false).

        Text:
        {text}
        """
        response_schema = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "statement": {"type": "STRING"},
                    "correct_answer": {"type": "BOOLEAN"}
                },
                "required": ["statement", "correct_answer"]
            }
        }
    elif question_type == "fill-in-the-blanks":
        prompt = f"""
        From the following text, generate {num_questions} fill-in-the-blanks questions. For each question, remove a key word or phrase and replace it with an underscore '__'. Provide the correct answer for the blank.
        Provide the output as a JSON array of objects, where each object has "question" (with the blank) and "correct_answer".

        Text:
        {text}
        """
        response_schema = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "question": {"type": "STRING"},
                    "correct_answer": {"type": "STRING"}
                },
                "required": ["question", "correct_answer"]
            }
        }
    elif question_type == "open-ended":
        prompt = f"""
        From the following text, generate {num_questions} factual, open-ended questions that can be answered directly from the text.
        Provide the output as a JSON array of strings.

        Text:
        {text}
        """
        response_schema = {
            "type": "ARRAY",
            "items": {"type": "STRING"}
        }
    elif question_type == "mixed":
        prompt = f"""
        From the following text, generate a mix of {num_questions} factual questions. Include Multiple Choice Questions (MCQ), True/False questions, Fill-in-the-blanks, and Open-ended questions.
        For MCQs, provide "question", "options" (array of 4 strings), and "correct_answer" (letter A, B, C, or D).
        For True/False, provide "statement" and "correct_answer" (boolean).
        For Fill-in-the-blanks, provide "question" (with blank '__') and "correct_answer".
        For Open-ended, just provide the question as a string.
        Each question object should also have a "type" field indicating the question type (e.g., "mcq", "true/false", "fill-in-the-blanks", "open-ended").
        Provide the output as a JSON array of these question objects.

        Text:
        {text}
        """
        # For mixed, the schema needs to be flexible, or we handle parsing more loosely.
        # A union type for items is complex in basic JSON schema, so we'll often parse and validate in code.
        # For a prompt that asks for various structured types, it's often easier to parse and then validate.
        # However, for the sake of demonstrating structured output, let's try a less strict schema for mixed
        # and rely on the model to follow the instructions within the prompt for each type.
        # A more robust solution for mixed would involve more complex schema or post-processing validation.
        response_schema = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "type": {"type": "STRING"}, # e.g., "mcq", "true/false", etc.
                    "question": {"type": "STRING"},
                    "options": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "correct_answer": {"type": ["STRING", "BOOLEAN"]},
                    "statement": {"type": "STRING"} # For true/false
                },
                "required": ["type"] # We'll check other fields based on 'type'
            }
        }
    else:
        print("Invalid question type specified.")
        return []

    result = call_gemini_api(prompt, response_schema=response_schema)

    if result and result.get("candidates") and len(result["candidates"]) > 0 and \
       result["candidates"][0].get("content") and result["candidates"][0]["content"].get("parts") and \
       len(result["candidates"][0]["content"]["parts"]) > 0:
        try:
            questions_json_str = result["candidates"][0]["content"]["parts"][0].get("text", "[]")
            questions = json.loads(questions_json_str)

            # Basic validation based on selected type
            if isinstance(questions, list):
                if question_type == "open-ended":
                    if all(isinstance(q, str) for q in questions):
                        return questions
                    else:
                        print(f"Warning: Generated open-ended questions are not in the expected list of strings format. Raw: {questions_json_str}")
                        return []
                elif question_type == "mcq":
                    if all(isinstance(q, dict) and "question" in q and "options" in q and "correct_answer" in q for q in questions):
                        return questions
                    else:
                        print(f"Warning: Generated MCQ questions are not in the expected format. Raw: {questions_json_str}")
                        return []
                elif question_type == "true/false":
                    if all(isinstance(q, dict) and "statement" in q and "correct_answer" in q for q in questions):
                        return questions
                    else:
                        print(f"Warning: Generated True/False questions are not in the expected format. Raw: {questions_json_str}")
                        return []
                elif question_type == "fill-in-the-blanks":
                    if all(isinstance(q, dict) and "question" in q and "correct_answer" in q for q in questions):
                        return questions
                    else:
                        print(f"Warning: Generated Fill-in-the-blanks questions are not in the expected format. Raw: {questions_json_str}")
                        return []
                elif question_type == "mixed":
                    # For mixed, we'll return as is and rely on the display logic to handle different types
                    return questions
            else:
                print(f"Warning: Generated questions are not in a list format. Raw: {questions_json_str}")
                return []
        except json.JSONDecodeError:
            print(f"Error: Could not parse generated questions as JSON. Raw: {questions_json_str}")
            return []
    else:
        print("Error: No questions generated or unexpected API response structure.")
        print(f"API response: {result}")
        return []

def evaluate_answer(original_text, question_data, user_answer, question_type):
    """
    Evaluates a user's answer against the original text using the Gemini API.

    Args:
        original_text (str): The full original text from the PDF.
        question_data (str or dict): The question asked (string for open-ended, dict for structured types).
        user_answer (str): The user's provided answer.
        question_type (str): The type of question being evaluated.

    Returns:
        dict: A dictionary containing 'score' (0-100), 'feedback' (textual), and 'correct_answer' (brief correct answer),
              or None if an error occurs.
    """
    print(f"Evaluating answer for question: '{question_data}'...")

    if isinstance(question_data, dict):
        # For structured questions, we might want to display the full question for clarity in the prompt
        if question_type == "mcq":
            question_text = f"{question_data['question']}\nOptions: {', '.join(question_data['options'])}"
        elif question_type == "true/false":
            question_text = f"Statement: {question_data['statement']}"
        elif question_type == "fill-in-the-blanks":
            question_text = f"Fill in the blank: {question_data['question']}"
        else: # mixed or other unexpected structured type
            question_text = str(question_data)
    else: # Open-ended or if 'mixed' generates a simple string
        question_text = question_data

    prompt = f"""
    You are an AI assistant evaluating a user's answer to a question based on a provided text.

    Original Text:
    {original_text}

    Question:
    {question_text}

    User's Answer:
    {user_answer}

    Evaluate the user's answer for accuracy and completeness based *only* on the information in the Original Text.
    Provide your evaluation as a JSON object with the following keys:
    - "score": An integer from 0 to 100, indicating how accurate and complete the user's answer is.
    - "feedback": A concise textual feedback explaining why the score was given, pointing out strengths and weaknesses, and suggesting improvements.
    - "correct_answer": A brief, accurate answer to the question based on the Original Text. For True/False questions, state "True" or "False". For Fill-in-the-blanks, state the missing word/phrase. For MCQ, state the correct option letter and content.
    """

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
            feedback_json_str = result["candidates"][0]["content"]["parts"][0].get("text", "{}")
            feedback_data = json.loads(feedback_json_str)
            if all(k in feedback_data for k in ["score", "feedback", "correct_answer"]):
                return feedback_data
            else:
                print(f"Warning: Generated feedback is not in the expected JSON format. Raw: {feedback_json_str}")
                return None
        except json.JSONDecodeError:
            print(f"Error: Could not parse generated feedback as JSON. Raw: {feedback_json_str}")
            return None
    else:
        print("Error: No evaluation feedback generated or unexpected API response structure.")
        print(f"API response: {result}")
        return None

if __name__ == "__main__":
    # IMPORTANT: Replace 'path/to/your/document.pdf' with the actual path to your PDF file
    # For Windows paths, use a raw string (r"...") or double backslashes ("\\")
    pdf_file_path = r"C:\Users\jazil\Desktop\yourfile2.pdf" # Replace with your PDF path

    print(f"--- PDF Question-Answer Evaluator ---")
    print(f"Attempting to extract text from: {pdf_file_path}")

    extracted_text = extract_text_from_pdf(pdf_file_path)

    if not extracted_text:
        print("Failed to extract text from PDF. Exiting.")
    else:
        print("\nText extracted successfully. Ready to generate questions.")

        while True:
            print("\nSelect question type:")
            print("1. Open-ended")
            print("2. Multiple Choice Questions (MCQ)")
            print("3. True/False")
            print("4. Fill-in-the-blanks")
            print("5. Mixed (A combination of types)")
            q_type_choice = input("Enter your choice (1-5): ").strip()

            question_type_map = {
                "1": "open-ended",
                "2": "mcq",
                "3": "true/false",
                "4": "fill-in-the-blanks",
                "5": "mixed"
            }

            selected_question_type = question_type_map.get(q_type_choice)

            if not selected_question_type:
                print("Invalid choice. Please enter a number between 1 and 5.")
                continue

            try:
                num_questions = int(input(f"How many {selected_question_type} questions do you want to generate? ").strip())
                if num_questions <= 0:
                    print("Please enter a positive number of questions.")
                    continue
            except ValueError:
                print("Invalid input. Please enter a number.")
                continue

            questions = generate_questions(extracted_text, num_questions=num_questions, question_type=selected_question_type)

            if not questions:
                print("Failed to generate questions. Please try again or with a different document/parameters.")
            else:
                print(f"\n{len(questions)} {selected_question_type} questions generated. Let's start the quiz!")
                for i, q_data in enumerate(questions):
                    print(f"\n--- Question {i+1} ---")

                    current_question_type = selected_question_type # Assume type unless mixed
                    if selected_question_type == "mixed":
                        current_question_type = q_data.get("type", "open-ended") # Default to open-ended if type is missing

                    if current_question_type == "mcq":
                        print(f"Q: {q_data['question']}")
                        for j, option in enumerate(q_data['options']):
                            print(f"  {chr(65+j)}. {option}")
                    elif current_question_type == "true/false":
                        print(f"Q: True or False: {q_data['statement']}")
                    elif current_question_type == "fill-in-the-blanks":
                        print(f"Q: Fill in the blank: {q_data['question']}")
                    else: # open-ended or mixed where type is not explicitly found
                        print(f"Q: {q_data if isinstance(q_data, str) else q_data.get('question', 'N/A')}")

                    user_answer = input("Your Answer: ").strip()

                    if not user_answer:
                        print("You didn't provide an answer. Skipping evaluation for this question.")
                        continue

                    feedback = evaluate_answer(extracted_text, q_data, user_answer, current_question_type)

                    if feedback:
                        print("\n--- Feedback ---")
                        print(f"Score: {feedback.get('score', 'N/A')}/100")
                        print(f"Feedback: {feedback.get('feedback', 'No feedback provided.')}")
                        print(f"Correct Answer: {feedback.get('correct_answer', 'Not available.')}")
                    else:
                        print("Could not get evaluation feedback for this answer.")
                print("\n--- Quiz Finished ---")
            
            another_round = input("\nDo you want to generate more questions? (yes/no): ").strip().lower()
            if another_round != 'yes':
                break