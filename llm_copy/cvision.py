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

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import pyttsx3
import speech_recognition as sr
import time

st.set_page_config(layout="wide")

DOCS_DIR = os.path.abspath("./uploaded_docs")

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

# Text to be converted to audio
text = "Welcome ! Chat with your AI Assistant!"

converted_audio = gtts.gTTS(text, lang="en")

# Save audio to BytesIO object instead of a file
audio_bytes = BytesIO()
converted_audio.write_to_fp(audio_bytes)
audio_bytes.seek(0)

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(audio_bytes)
pygame.mixer.music.play()
time.sleep(3)

text1 = "আপনার সহায়কের সাথে খোশগল্প করুন !"

converted_audio1 = gtts.gTTS(text1, lang="bn")

# Save audio to BytesIO object instead of a file
audio_bytes1 = BytesIO()
converted_audio1.write_to_fp(audio_bytes1)
audio_bytes1.seek(0)

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(audio_bytes1)
pygame.mixer.music.play()
time.sleep(3)

text2 = "अपने सहायक के साथ बात करें !"

converted_audio2 = gtts.gTTS(text2, lang="hi")

# Save audio to BytesIO object instead of a file
audio_bytes2 = BytesIO()
converted_audio2.write_to_fp(audio_bytes2)
audio_bytes2.seek(0)

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(audio_bytes2)
pygame.mixer.music.play()
time.sleep(3)



def capture_and_play_english_voice_input():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.info("Listening for English voice input...")
        audio = recognizer.listen(source, timeout=10)  # Adjust timeout as needed

    try:
        st.success("English voice input captured successfully!")
        text = recognizer.recognize_google(audio,language="en")  # English language code
        st.text(f"English Voice Input: {text}")
        return text
    except sr.UnknownValueError:
        st.warning("Could not understand English audio.")
        return ""
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return ""
    
def capture_and_translate_voice_input_hindi():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.info("Listening for Hindi voice input...")
        audio = recognizer.listen(source, timeout=10)  # Adjust timeout as needed

    try:
        st.success("Hindi voice input captured successfully!")
        text = recognizer.recognize_google(audio, language="hi-IN")  # Hindi language code
        st.text(f"Hindi Voice Input: {text}")

        translation = translator.translate(text, dest="en")
        st.text(f"Translated: {translation.text}")
        return text, translation.text, "hi"
    except sr.UnknownValueError:
        st.warning("Could not understand Hindi audio.")
        return "", "", ""
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return "", "", ""

def capture_and_translate_voice_input_bengali():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        st.info("Listening for Bengali voice input...")
        audio = recognizer.listen(source, timeout=10)  # Adjust timeout as needed

    try:
        st.success("Bengali voice input captured successfully!")
        text = recognizer.recognize_google(audio, language="bn-IN")  # Bengali language code
        st.text(f"Bengali Voice Input: {text}")

        translation = translator.translate(text, dest="en")
        st.text(f"Translated: {translation.text}")
        return text, translation.text, "bn"
    except sr.UnknownValueError:
        st.warning("Could not understand Bengali audio.")
        return "", "", ""
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Speech Recognition service; {e}")
        return "", "", ""

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
    elif input_language == "hi":  # Hindi input, provide answer in Hindi
        st.text("Regional Language Output: (Original Hindi)")
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    elif input_language == "bn":  # Bengali input, provide answer in Bengali
        st.text("Regional Language Output: (Original Bengali)")
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        #st.text("Regional Language Output: (Original English)")
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Convert the translated text to voice
    converted_audio = gtts.gTTS(regional_translation.text, lang=input_language)
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

    # Update chat output dynamically
    chat_output_slot.markdown(f'<div style="height: 200px;"></div>', unsafe_allow_html=True)  # Empty space
    chat_output_slot.markdown('<style>div{transition: height 0.5s ease;}</style>', unsafe_allow_html=True)  # Smooth transition

while(True):
    text3 = "For English Speak English हिंदी बोलने के लिए Hindi  এবং বাংলার জন্য Bangla Speak Quit for Quit"

    converted_audio3 = gtts.gTTS(text3)

    # Save audio to BytesIO object instead of a file
    audio_bytes3 = BytesIO()
    converted_audio3.write_to_fp(audio_bytes3)
    audio_bytes3.seek(0)

    pygame.init()
    pygame.mixer.init()
    pygame.mixer.music.load(audio_bytes3)
    pygame.mixer.music.play()

    time.sleep(6)
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening for  voice input...")
        audio = recognizer.listen(source, timeout=5)  # Adjust timeout as needed

    try:
        st.success("voice input captured successfully!")
        text = recognizer.recognize_google(audio)  # English language code
        st.text(f"Voice Input: {text}")
    except sr.UnknownValueError:
        st.warning("Could not understand audio.")

    if text== 'English':
        text4 = "Ask your Query in English"

        converted_audio4 = gtts.gTTS(text4,lang="en")

        # Save audio to BytesIO object instead of a file
        audio_bytes4 = BytesIO()
        converted_audio4.write_to_fp(audio_bytes4)
        audio_bytes4.seek(0)

        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(audio_bytes4)
        pygame.mixer.music.play()
        time.sleep(3)
        voice_input = capture_and_play_english_voice_input()
        if voice_input:
            process_user_input(voice_input, "en")

    if text== 'Hindi':
        text5 = "हिंदी में प्रश्न पूछें"

        converted_audio5 = gtts.gTTS(text5,lang="hi")

        # Save audio to BytesIO object instead of a file
        audio_bytes5 = BytesIO()
        converted_audio5.write_to_fp(audio_bytes5)
        audio_bytes5.seek(0)

        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(audio_bytes5)
        pygame.mixer.music.play()
        time.sleep(3)
        voice_input, translated_text, language = capture_and_translate_voice_input_hindi()
        if voice_input:
            #st.text(f"Hindi Voice Input: {voice_input}")
            if language != "":  # Check if translation is needed
                #st.text(f"Translated Text: {translated_text}")
                process_user_input(translated_text, language)
            else:
                process_user_input(voice_input, "hi")

    if text== 'Bangla':
        text6 = "বাংলায় প্রশ্ন জিজ্ঞাসা করুন"

        converted_audio6 = gtts.gTTS(text6,lang="bn")

        # Save audio to BytesIO object instead of a file
        audio_bytes6 = BytesIO()
        converted_audio6.write_to_fp(audio_bytes6)
        audio_bytes6.seek(0)

        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(audio_bytes6)
        pygame.mixer.music.play()
        time.sleep(3)
        voice_input, translated_text, language = capture_and_translate_voice_input_bengali()
        if voice_input:
            #st.text(f"Hindi Voice Input: {voice_input}")
            if language != "":  # Check if translation is needed
                #st.text(f"Translated Text: {translated_text}")
                process_user_input(translated_text, language)
            else:
                process_user_input(voice_input, "bn")
    if text=='quit':
        st.subheader("Thanks for using us")
        break

#def capture_and_play_english_voice_input():

# Function to process user input

# Capture voice input buttons
#col1, col2, col3 = st.columns(3)

# Capture English voice input button
#if col1.button("Capture English Voice Input"):
#    voice_input = capture_and_play_english_voice_input()
#    if voice_input:
#        #st.text(f"Original English Text: {voice_input}")
#        process_user_input(voice_input, "en")


