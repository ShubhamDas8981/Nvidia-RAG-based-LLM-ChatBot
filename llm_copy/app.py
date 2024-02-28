import streamlit as st
import os
import googletrans
import speech_recognition as sr
import gtts
from io import BytesIO
import pygame
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS
import pickle
from langdetect import detect
import langid

#from PIL import Image
#from pytesseract import TesseractNotFoundError  # Import the specific exception
#import pytesseract 


st.set_page_config(layout="wide")

# Component #1 - Document Loader

with st.sidebar:
    DOCS_DIR = os.path.abspath("./uploaded_docs")
    

    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)
    st.subheader("Add to the Knowledge Base (must be English)")
    with st.form("my-form", clear_on_submit=True):
        uploaded_files = st.file_uploader("Upload a file to the Knowledge Base:", accept_multiple_files=True)
        submitted = st.form_submit_button("Upload!")

    if uploaded_files and submitted:
        for uploaded_file in uploaded_files:
            st.success(f"File {uploaded_file.name} uploaded successfully!")
            with open(os.path.join(DOCS_DIR, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.read())
    # Display the names of existing uploaded documents with delete option
    st.subheader("Existing Uploaded Documents:")
    existing_documents = os.listdir(DOCS_DIR)
    if existing_documents:
        for doc_name in existing_documents:
            delete_checkbox = st.checkbox(f"Delete {doc_name}")
            if delete_checkbox:
                doc_path = os.path.join(DOCS_DIR, doc_name)
                os.remove(doc_path)
                st.success(f"Document {doc_name} deleted successfully!")

    else:
        st.text("No documents uploaded yet.")


# Component #2 - Embedding Model and LLM

from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

llm = ChatNVIDIA(model="mixtral_8x7b")
document_embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="passage")
query_embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="query")

# Component #3 - Vector Database Store

with st.sidebar:
    use_existing_vector_store = st.radio("Use existing vector store if available", ["Yes", "No"], horizontal=True)

# Path to the vector store file
vector_store_path = "vectorstore.pkl"

# Load raw documents from the directory
raw_documents = DirectoryLoader(DOCS_DIR).load()

vectorstore = None
if use_existing_vector_store == "Yes" and os.path.exists(vector_store_path):
    with open(vector_store_path, "rb") as f:
        vectorstore = pickle.load(f)
    with st.sidebar:
        st.success("Existing vector store loaded successfully.")
else:
    with st.sidebar:
        if raw_documents:
            with st.spinner("Splitting documents into chunks..."):
                text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                documents = text_splitter.split_documents(raw_documents)

            with st.spinner("Adding document chunks to vector database..."):
                vectorstore = FAISS.from_documents(documents, document_embedder)

            with st.spinner("Saving vector store"):
                with open(vector_store_path, "wb") as f:
                    pickle.dump(vectorstore, f)
            st.success("Vector store created and saved.")
        else:
            st.warning("No documents available to process!", icon="⚠️")

# Component #4 - LLM Response Generation and Chat

st.subheader("পিডিএফ সহকারী (PDF Sahokari) - PDF Assistant")

st.subheader("Chat with your AI Assistant!")
st.subheader("আপনার সহায়কের সাথে চ্যাট করুন !")
st.subheader("अपने सहायक के साथ चैट करें !")


chat_output_slot = st.empty()  # Create an empty space for dynamic chat updates


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import pyttsx3
import speech_recognition as sr

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Initialize the chain variable
prompt_template = ChatPromptTemplate.from_messages(
    [("system", "You are a helpful AI assistant named Envie. You will reply to questions only based on the context that you are provided. If something is out of context, you will refrain from replying and politely decline to respond to the user."), ("user", "{input}")]
)
llm = ChatNVIDIA(model="mixtral_8x7b")
chain = prompt_template | llm | StrOutputParser()

# Initialize the translator
translator = googletrans.Translator()



# Function to capture and play English voice input
def capture_and_play_english_voice_input():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.info("Listening for English voice input...")
        audio = recognizer.listen(source, timeout=10)  # Adjust timeout as needed

    try:
        st.success("Voice input captured successfully!")

        # Detect the language of the spoken input
        detected_language = detect(recognizer.recognize_google(audio))

        if detected_language != "en":
            st.warning(f"Detected language: {detected_language}. Expected English. Please speak in English.")
            return ""

        text = recognizer.recognize_google(audio, language="en")  # English language code
        st.text(f"English Voice Input: {text}")

        converted_audio = gtts.gTTS(text, lang="en")

        # Save audio to BytesIO object instead of a file
        audio_bytes = BytesIO()
        converted_audio.write_to_fp(audio_bytes)
        audio_bytes.seek(0)

        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(audio_bytes)
        pygame.mixer.music.play()

        # Wait until the audio is finished playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        return text
    except sr.UnknownValueError:
        st.warning("Could not understand the spoken input.")
        return ""
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return ""

# Function to capture and translate voice input for Hindi
def capture_and_translate_voice_input_hindi():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.info("Listening for Hindi voice input...")
        audio = recognizer.listen(source, timeout=10)  # Adjust timeout as needed

    try:
        st.success("Hindi voice input captured successfully!")
        text = recognizer.recognize_google(audio, language="hi-IN")  # Hindi language code
        st.text(f"Hindi Voice Input: {text}")

        # Check detected language
        detected_language, _ = langid.classify(text)
        if detected_language != 'hi':
            st.warning(f"Detected language: {detected_language}. Expected Hindi. Please speak in Hindi.")
            return "", "", ""

        translation = translator.translate(text, dest="en")
        st.text(f"Translated: {translation.text}")

        converted_audio = gtts.gTTS(translation.text, lang="hi")

        # Save audio to BytesIO object instead of a file
        audio_bytes = BytesIO()
        converted_audio.write_to_fp(audio_bytes)
        audio_bytes.seek(0)

        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(audio_bytes)
        pygame.mixer.music.play()

        # Wait until the audio is finished playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        return text, translation.text, "hi"
    except sr.UnknownValueError:
        st.warning("Could not understand Hindi audio.")
        return "", "", ""
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return "", "", ""

# Function to capture and translate voice input for Bengali
def capture_and_translate_voice_input_bengali():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.info("Listening for Bengali voice input...")
        audio = recognizer.listen(source, timeout=10)  # Adjust timeout as needed

    try:
        st.success("Bengali voice input captured successfully!")
        text = recognizer.recognize_google(audio, language="bn-IN")  # Bengali language code
        st.text(f"Bengali Voice Input: {text}")

        # Check detected language
        detected_language, _ = langid.classify(text)
        if detected_language != 'bn':
            st.warning(f"Detected language: {detected_language}. Expected Bengali. Please speak in Bengali.")
            return "", "", ""

        translation = translator.translate(text, dest="en")
        st.text(f"Translated: {translation.text}")

        converted_audio = gtts.gTTS(translation.text, lang="bn")

        # Save audio to BytesIO object instead of a file
        audio_bytes = BytesIO()
        converted_audio.write_to_fp(audio_bytes)
        audio_bytes.seek(0)

        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(audio_bytes)
        pygame.mixer.music.play()

        # Wait until the audio is finished playing
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

        return text, translation.text, "bn"
    except sr.UnknownValueError:
        st.warning("Could not understand Bengali audio.")
        return "", "", ""
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return "", "", ""

# Function to process user input
def process_user_input(user_input, input_language):
    st.session_state.messages.append({"role": "user", "content": user_input})
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    context = ""
    for doc in docs:
        context += doc.page_content + "\n\n"

    augmented_user_input = "Context: " + context + "\n\nQuestion: " + user_input + "\n"

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for response in chain.stream({"input": augmented_user_input}):
            full_response += response
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

    # Translate the full response to a regional language if input language is not English
    if input_language != "en":
        if input_language == "hi":
            regional_translation = translator.translate(full_response, dest="hi")
        elif input_language == "bn":
            regional_translation = translator.translate(full_response, dest="bn")
        else:
            regional_translation = full_response

        st.text(f"Regional Language Output: {regional_translation.text}")
        st.session_state.messages.append({"role": "assistant", "content": regional_translation.text})

        # Convert the translated text to voice
        converted_audio = gtts.gTTS(regional_translation.text, lang=input_language)
    else:
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # Convert the original English text to voice
        converted_audio = gtts.gTTS(full_response, lang="en")

    # Save audio to BytesIO object instead of a file
    audio_bytes = BytesIO()
    converted_audio.write_to_fp(audio_bytes)
    audio_bytes.seek(0)

    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(audio_bytes)
    pygame.mixer.music.play()

    # Wait until the audio is finished playing
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(0.5)
    cont_button = st.button("Click to Continue")

    if cont_button:
        pygame.mixer.Channel(0).stop()

    # Update chat output dynamically
    chat_output_slot.markdown(f'<div style="height: 200px;"></div>', unsafe_allow_html=True)  # Empty space
    chat_output_slot.markdown('<style>div{transition: height 0.5s ease;}</style>', unsafe_allow_html=True)  # Smooth transition
# Capture voice input buttons
col1, col2, col3 = st.columns(3)

# Capture English voice input button
if col1.button("Speak in English"):
    voice_input = capture_and_play_english_voice_input()
    if voice_input:
        #st.text(f"Original English Text: {voice_input}")
        process_user_input(voice_input, "en")

# Capture Hindi voice input button
if col2.button("हिंदी मे बोलो"):
    voice_input, translated_text, language = capture_and_translate_voice_input_hindi()
    if voice_input:
        #st.text(f"Hindi Voice Input: {voice_input}")
        if language != "":  # Check if translation is needed
            #st.text(f"Translated Text: {translated_text}")
            process_user_input(translated_text, language)
        else:
            process_user_input(voice_input, "hi")

# Capture Bengali voice input button
if col3.button("বাংলায় কথা বলুন"):
    voice_input, translated_text, language = capture_and_translate_voice_input_bengali()
    if voice_input:
        #st.text(f"Bengali Voice Input: {voice_input}")
        if language != "":  # Check if translation is needed
            #st.text(f"Translated Text: {translated_text}")
            process_user_input(translated_text, language)
        else:
            process_user_input(voice_input, "bn")


# Chat input box
user_input = st.chat_input("Type your question:")
if user_input and vectorstore is not None:
    process_user_input(user_input, "en")
