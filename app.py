import streamlit as st
import PyPDF2
import json
import os
import re
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from gtts import gTTS
import tempfile

# --- Page Configuration ---
# Use st.set_page_config() only once at the beginning of the script.
st.set_page_config(page_title="NeuroMind AI", layout="wide", page_icon="🧠")


# --- Gemini API Configuration ---
# It's recommended to use st.secrets for API keys in deployed apps.
# For local development, you can set it as an environment variable or hardcode it.
# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or "YOUR_API_KEY"
# The user provided a key, so we will use that.
GEMINI_API_KEY = "AIzaSyDLCj1L3IOmEXwbN27NuvwxZu3mh7kvtjI"

# Configure the genai library
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Gemini API. Please check your API key. Error: {e}")


# --- Backend Functions ---

def call_gemini_api(prompt, response_schema=None, file_part=None):
    """
    Calls the Gemini API for content generation, optionally with a file input.
    This function is the single point of contact for all Gemini API calls.
    """
    if not GEMINI_API_KEY or "YOUR_API_KEY" in GEMINI_API_KEY:
        st.error("Gemini API key is not configured. Please add it to proceed.")
        return None

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        contents = []
        if file_part:
            contents.append(file_part)
        # The genai library expects the prompt string to be part of a list/dict structure
        contents.append(prompt)

        generation_config = None
        if response_schema:
            generation_config = genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=response_schema,
                temperature=0.3
            )

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
        # Handle both schema and non-schema responses
        if response_schema:
            return json.loads(response.text)
        else:
            # Standardize the non-schema response for consistency
            return {"candidates": [{"content": {"parts": [{"text": response.text}]}}]}
    except Exception as e:
        st.error(f"An error occurred during the Gemini API call: {e}")
        return None

def extract_text_from_pdf(pdf_file_obj):
    """Extracts text from an uploaded PDF file object."""
    text = ""
    try:
        # The user needs to seek(0) before passing the file object
        pdf_file_obj.seek(0)
        reader = PyPDF2.PdfReader(pdf_file_obj)
        for page in reader.pages:
            text += (page.extract_text() or "")
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text

def get_pdf_file_part(uploaded_file):
    """Prepares the uploaded PDF file for the Gemini API multimodal call."""
    uploaded_file.seek(0)
    pdf_bytes = uploaded_file.read()
    return {"mime_type": "application/pdf", "data": pdf_bytes}

def extract_titles_from_pdf_multimodal(pdf_file_part):
    """Extracts main section titles from a PDF using Gemini's multimodal capabilities."""
    prompt = "Analyze the provided PDF document. Identify and list all the main section titles or prominent headings that structure the content, like a Table of Contents. Exclude page numbers and minor subheadings. Return the output as a JSON array of strings."
    response_schema = {"type": "ARRAY", "items": {"type": "STRING"}}
    result_json = call_gemini_api(prompt, response_schema=response_schema, file_part=pdf_file_part)
    if result_json and isinstance(result_json, list):
        # Filter and clean titles
        return [t.strip() for t in result_json if isinstance(t, str) and len(t.strip()) > 3]
    return []

def get_text_for_topic(full_text, selected_topic, all_titles):
    """Isolates the text content for a selected topic/section from the full text."""
    if not selected_topic or "entire document" in selected_topic.lower():
        return full_text
    try:
        start_index = full_text.lower().index(selected_topic.lower())
        end_index = len(full_text)
        # Find the next topic in the list to determine the end of the current section
        current_topic_index_in_titles = all_titles.index(selected_topic)
        if current_topic_index_in_titles + 1 < len(all_titles):
            next_topic = all_titles[current_topic_index_in_titles + 1]
            try:
                # Search for the next topic starting from after the current topic's start
                end_index = full_text.lower().index(next_topic.lower(), start_index + 1)
            except ValueError:
                pass # Next topic not found, so we'll go to the end of the document.
        return full_text[start_index:end_index]
    except (ValueError, IndexError):
        return full_text # Fallback to full document if topic not found

def classify_student(marks):
    """Classifies student performance based on marks."""
    if marks < 50: return "Needs Improvement"
    elif 50 <= marks < 70: return "Average"
    elif 70 <= marks < 85: return "Good"
    elif 85 <= marks < 95: return "Excellent"
    else: return "Advanced Learner"

def summarize_text_with_gemini(text_to_summarize, summary_type="medium", student_classification=None):
    """Generates a standard or personalized summary."""
    if not text_to_summarize.strip(): return "Error: No text was provided for summarization."

    prompts = {
        "quick": "Summarize the following text very briefly, highlighting only the main points.",
        "medium": "Summarize the following text concisely and accurately.",
        "detailed": "Provide a detailed summary of the following text, covering all key aspects and supporting details."
    }
    base_prompt = prompts.get(summary_type.lower(), prompts["medium"])

    personalization_prompts = {
        "Needs Improvement": " Make the language very simple and easy to understand. Avoid complex jargon.",
        "Average": " Use clear and straightforward language. Briefly explain difficult terms.",
        "Good": " Use standard academic language with good detail.",
        "Excellent": " Use precise and technical terminology where appropriate.",
        "Advanced Learner": " Provide a highly detailed and nuanced summary, using advanced terminology and exploring subtle implications."
    }
    if student_classification in personalization_prompts:
        base_prompt += personalization_prompts[student_classification]

    full_prompt = f"{base_prompt}\n\nText to summarize:\n{text_to_summarize}"
    result = call_gemini_api(full_prompt)
    return result["candidates"][0]["content"]["parts"][0].get("text", "Error: No summary found.") if result else "Error generating summary."

def generate_questions(text, num_questions=3, question_type="open-ended"):
    """Generates quiz questions of a specified type using the Gemini API."""
    prompt, schema = "", None
    if question_type == "mcq":
        prompt = f'From the provided text, generate exactly {num_questions} Multiple Choice Questions. Each question must have a "question" field, an "options" field containing 4 strings, and a "correct_answer" field which is one of the provided options.\n\nText:\n{text}'
        schema = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"question": {"type": "STRING"}, "options": {"type": "ARRAY", "items": {"type": "STRING"}}, "correct_answer": {"type": "STRING"}}, "required": ["question", "options", "correct_answer"]}}
    elif question_type == "true/false":
        prompt = f'From the provided text, generate exactly {num_questions} True/False questions. Each question must have a "statement" field and a "correct_answer" field which must be a boolean (true or false).\n\nText:\n{text}'
        schema = {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {"statement": {"type": "STRING"}, "correct_answer": {"type": "BOOLEAN"}}, "required": ["statement", "correct_answer"]}}
    else: # open-ended
        prompt = f'From the provided text, generate exactly {num_questions} open-ended questions that encourage critical thinking.\n\nText:\n{text}'
        schema = {"type": "ARRAY", "items": {"type": "STRING"}}
    return call_gemini_api(prompt, response_schema=schema)

def get_chatbot_response(user_input, chat_history, context_text):
    """Generates a chatbot response based on conversation history and document context."""
    # Constructing a prompt that includes history and context
    history_prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    full_prompt = (
        f"You are a helpful assistant. Your knowledge is based on the following document text. "
        f"Answer the user's question based on this text and the conversation history.\n\n"
        f"--- Document Context ---\n{context_text}\n\n"
        f"--- Conversation History ---\n{history_prompt}\n\n"
        f"User: {user_input}\nAI:"
    )
    result = call_gemini_api(full_prompt)
    return result["candidates"][0]["content"]["parts"][0].get("text", "Sorry, I had trouble responding.") if result else "Connection error."

def assign_priority_and_duration(score):
    """Assigns study priority and duration based on a score."""
    if score >= 80:
        return "Low", 30  # Low priority for high scores
    elif score >= 60:
        return "Medium", 60
    else:
        return "High", 90 # High priority for low scores

# --- Streamlit Frontend ---

st.title("🧠 NeuroMind AI")
st.markdown("Because every student matters")

# --- Session State Initialization ---
# This ensures that variables persist across user interactions.
for key in ['pdf_text', 'questions', 'chat_history', 'extracted_titles', 'pdf_file_part', 'current_question_index', 'user_score']:
    if key not in st.session_state:
        if key in ['questions', 'extracted_titles']:
            st.session_state[key] = []
        elif key == 'chat_history':
            st.session_state[key] = []
        elif key in ['current_question_index', 'user_score']:
            st.session_state[key] = 0
        else:
            st.session_state[key] = None

# --- PDF Uploader and Processor (Sidebar) ---
st.sidebar.header("1. Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload your PDF document", type="pdf", key="pdf_uploader")

# Process the file only once when a new file is uploaded
if uploaded_file and (uploaded_file.name != st.session_state.get('processed_file_name')):
    with st.spinner('Analyzing document... This may take a moment.'):
        st.session_state.pdf_text = extract_text_from_pdf(uploaded_file)
        st.session_state.pdf_file_part = get_pdf_file_part(uploaded_file)
        if st.session_state.pdf_file_part:
            st.session_state.extracted_titles = extract_titles_from_pdf_multimodal(st.session_state.pdf_file_part)
        st.session_state.processed_file_name = uploaded_file.name # Track the processed file
        # Reset other states when a new PDF is uploaded
        st.session_state.questions = []
        st.session_state.chat_history = []
        st.session_state.current_question_index = 0
        st.session_state.user_score = 0


if st.session_state.pdf_text:
    st.sidebar.success(f"Successfully processed '{st.session_state.processed_file_name}'.")
    if st.session_state.extracted_titles:
        st.sidebar.info(f"Detected {len(st.session_state.extracted_titles)} sections.")
else:
    st.info("Please upload a PDF document using the sidebar to activate the features.")
    st.stop() # Stop execution if no PDF is processed

# --- Main Interface Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📝 Personalized Summarizer", "❓ Quiz Yourself", "💬 Chat with Document", "📅 Study Scheduler"])


# --- Tab 1: Personalized Summarizer ---
with tab1:
    st.header("Personalized Document Summarizer")
    st.markdown("Generate a summary of your document, tailored to your needs.")

    # Step 1: Topic Selection
    st.subheader("1. Select Content to Summarize")
    topic_options = ["Summarize Entire Document"] + st.session_state.extracted_titles
    selected_topic_summary = st.selectbox("Choose a section or the whole document:", topic_options, key="summarizer_topic")

    # Step 2: Personalization
    st.subheader("2. Personalize Your Summary (Optional)")
    is_personalized = st.toggle("Tailor summary for a student?", key="personalize_toggle")
    student_classification = None
    if is_personalized:
        marks = st.slider("Select student's performance level (0-100):", 0, 100, 75, key="student_marks")
        student_classification = classify_student(marks)
        st.info(f"Summary will be adapted for a **'{student_classification}'** learner.")

    # Step 3: Summary Depth
    st.subheader("3. Choose Summary Depth")
    summary_type = st.selectbox("Select summary detail level:", ["Quick", "Medium", "Detailed"], index=1, key="summary_depth")

    # Step 4: Generate
    if st.button("✨ Generate Summary", key="generate_summary_btn"):
        with st.spinner("Crafting your summary..."):
            text_to_summarize = get_text_for_topic(st.session_state.pdf_text, selected_topic_summary, st.session_state.extracted_titles)
            if not text_to_summarize or not text_to_summarize.strip():
                 st.error("Could not extract text for the selected topic. Please try another section or the entire document.")
            else:
                summary = summarize_text_with_gemini(text_to_summarize, summary_type, student_classification)
                st.markdown("---")
                st.markdown("### Your Generated Summary")
                st.success(summary)
                try:
                    with st.spinner("Generating audio version..."):
                        tts = gTTS(text=summary, lang='en')
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tts_file:
                            tts.save(tts_file.name)
                            audio_bytes = open(tts_file.name, 'rb').read()
                            st.audio(audio_bytes, format='audio/mp3')
                            os.remove(tts_file.name)
                except Exception as e:
                    st.warning(f"Could not generate audio for the summary. Error: {e}")


# --- Tab 2: Quiz Generator ---
with tab2:
    st.header("Test Your Knowledge")
    st.markdown("Generate a quiz from your document to check your understanding.")

    quiz_col1, quiz_col2, quiz_col3 = st.columns(3)
    with quiz_col1:
        num_questions = st.number_input("Number of questions:", min_value=1, max_value=10, value=3)
    with quiz_col2:
        question_type = st.selectbox("Question type:", ["mcq", "true/false", "open-ended"])
    with quiz_col3:
        # Let user select a topic for the quiz
        quiz_topic_options = ["Entire Document"] + st.session_state.extracted_titles
        selected_topic_quiz = st.selectbox("Quiz on which section?", quiz_topic_options, key="quiz_topic")

    if st.button("🚀 Generate Quiz", key="generate_quiz_btn"):
        st.session_state.questions = [] # Reset previous quiz
        st.session_state.current_question_index = 0
        st.session_state.user_score = 0
        with st.spinner("Generating your quiz..."):
            text_for_quiz = get_text_for_topic(st.session_state.pdf_text, selected_topic_quiz, st.session_state.extracted_titles)
            if not text_for_quiz or not text_for_quiz.strip():
                st.error("Could not extract text for the selected topic. Cannot generate quiz.")
            else:
                questions = generate_questions(text_for_quiz, num_questions, question_type)
                if questions and isinstance(questions, list):
                    st.session_state.questions = questions
                    st.success(f"Quiz generated with {len(questions)} questions!")
                else:
                    st.error("Failed to generate quiz. The API might have returned an unexpected format. Please try again.")

    if st.session_state.questions:
        st.markdown("---")
        idx = st.session_state.current_question_index
        if idx < len(st.session_state.questions):
            q = st.session_state.questions[idx]
            with st.form(key=f"quiz_form_{idx}"):
                if question_type == "mcq":
                    st.subheader(f"Question {idx + 1}: {q['question']}")
                    user_answer = st.radio("Choose your answer:", q['options'], index=None)
                elif question_type == "true/false":
                    st.subheader(f"Question {idx + 1}: {q['statement']}")
                    user_answer = st.radio("Is this statement True or False?", ["True", "False"], index=None)
                else: # open-ended
                    st.subheader(f"Question {idx + 1}: {q}")
                    user_answer = st.text_area("Your answer:")

                submitted = st.form_submit_button("Submit Answer")

                if submitted:
                    if user_answer is not None:
                        is_correct = False
                        if question_type == "mcq":
                            if user_answer == q['correct_answer']:
                                is_correct = True
                        elif question_type == "true/false":
                            if str(user_answer).lower() == str(q['correct_answer']).lower():
                                is_correct = True
                        
                        if is_correct:
                            st.success("Correct!")
                            st.session_state.user_score += 1
                        else:
                            if question_type != "open-ended":
                                st.error(f"Incorrect. The correct answer is: {q['correct_answer']}")
                            else:
                                st.info("Answer submitted for open-ended question.")
                        
                        st.session_state.current_question_index += 1
                        st.rerun()
                    else:
                        st.warning("Please select an answer.")
        else:
            st.balloons()
            st.success(f"Quiz complete! Your score: {st.session_state.user_score}/{len(st.session_state.questions)}")
            if st.button("🔄 Retake Quiz"):
                st.session_state.current_question_index = 0
                st.session_state.user_score = 0
                st.rerun()


# --- Tab 3: Chat with Document ---
with tab3:
    st.header("Chat with Your Document")
    st.markdown("Ask questions and get answers based on the content of your uploaded PDF.")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use the entire document as context for the chatbot
                response = get_chatbot_response(prompt, st.session_state.chat_history, st.session_state.pdf_text)
                st.markdown(response)
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})


# --- Tab 4: Study Scheduler ---
with tab4:
    st.header("📅 Smart Weekly Study Scheduler")
    st.markdown("Enter your subjects and recent scores to get a personalized weekly study plan. You can use the quiz results from the 'Quiz Yourself' tab!")

    # Use session state to store topics for persistence
    if 'scheduler_topics' not in st.session_state:
        st.session_state.scheduler_topics = [{"name": "", "score": 75} for _ in range(3)]

    num_topics = st.number_input("How many subjects do you want to schedule?", min_value=1, max_value=20, value=len(st.session_state.scheduler_topics), key="num_subjects")

    # Adjust the list of topics in session state if the number changes
    if num_topics > len(st.session_state.scheduler_topics):
        for _ in range(num_topics - len(st.session_state.scheduler_topics)):
            st.session_state.scheduler_topics.append({"name": "", "score": 75})
    elif num_topics < len(st.session_state.scheduler_topics):
        st.session_state.scheduler_topics = st.session_state.scheduler_topics[:num_topics]


    with st.form("subject_form"):
        st.subheader("Enter Your Subjects and Scores")
        for i in range(num_topics):
            cols = st.columns([3, 1])
            st.session_state.scheduler_topics[i]['name'] = cols[0].text_input(f"Subject {i+1} Name", value=st.session_state.scheduler_topics[i]['name'], key=f"name_{i}")
            st.session_state.scheduler_topics[i]['score'] = cols[1].number_input(f"Score (0-100)", min_value=0, max_value=100, value=st.session_state.scheduler_topics[i]['score'], key=f"score_{i}")

        submitted = st.form_submit_button("Generate Schedule")

    if submitted:
        # Filter out topics without a name
        valid_topics = [topic for topic in st.session_state.scheduler_topics if topic["name"].strip()]

        if not valid_topics:
            st.warning("Please enter at least one subject name.")
        else:
            week_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            schedule = {day: [] for day in week_days}
            
            # Sort topics by score (lowest first) to prioritize them
            sorted_topics = sorted(valid_topics, key=lambda x: x['score'])
            
            i = 0
            for topic in sorted_topics:
                priority, duration = assign_priority_and_duration(topic["score"])
                # Distribute tasks across the week
                day = week_days[i % len(week_days)]
                schedule[day].append({
                    "topic": topic["name"],
                    "priority": priority,
                    "recommended_duration": f"{duration} min"
                })
                i += 1

            st.markdown("---")
            st.markdown("## 🗓️ Your Weekly Smart Study Schedule")
            
            # Create a more visual schedule
            for day in week_days:
                st.markdown(f"#### {day}")
                if schedule[day]:
                    for task in schedule[day]:
                        if task['priority'] == 'High':
                            st.error(f"**{task['topic']}** ({task['priority']} Priority): Study for **{task['recommended_duration']}**")
                        elif task['priority'] == 'Medium':
                            st.warning(f"**{task['topic']}** ({task['priority']} Priority): Study for **{task['recommended_duration']}**")
                        else:
                            st.info(f"**{task['topic']}** ({task['priority']} Priority): Review for **{task['recommended_duration']}**")
                else:
                    st.success("🎉 Rest Day!")
