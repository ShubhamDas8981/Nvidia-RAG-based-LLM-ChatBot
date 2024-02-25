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
import hydralit_components as hc
import time
import cv2
import mediapipe as mp
import subprocess
import pyttsx3
import speech_recognition as sr

# for 1 (index=5) from the standard loader group
with hc.HyLoader('PDF Shokari',hc.Loaders.standard_loaders,index=2):
    time.sleep(5)


text1="Namaskar Before Completion of this loading  Please Choose your Thumb Gesture Thumbs Up for Blind or Thumbs Down for Normal Person"
text2="অনুগ্রহ করে ক্যামেরার সামনে বুড়ো আঙুলের ইশারা বেছে নিন, অন্ধ ব্যক্তির জন্য আপনার বুড়ো আঙুলটি উপরে নিয়ে যান বা সাধারণ ব্যক্তির জন্য আপনার বুড়ো আঙুলটি নিচে নামান"
text3="कृपया कैमरे के सामने अंगूठे की उंगली का इशारा चुनें, अंधे व्यक्ति के लिए अपने अंगूठे को ऊपर की ओर ले जाएं या सामान्य व्यक्ति के लिए अपने अंगूठे को नीचे की ओर ले जाएं"
converted_audio1 = gtts.gTTS(text1,lang='en')

# Save audio to BytesIO object instead of a file
audio_bytes1 = BytesIO()
converted_audio1.write_to_fp(audio_bytes1)
audio_bytes1.seek(0)

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(audio_bytes1)
pygame.mixer.music.play()
time.sleep(7)
converted_audio2 = gtts.gTTS(text2,lang='bn')

# Save audio to BytesIO object instead of a file
audio_bytes2 = BytesIO()
converted_audio2.write_to_fp(audio_bytes2)
audio_bytes2.seek(0)

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(audio_bytes2)
pygame.mixer.music.play()
time.sleep(14)
converted_audio3 = gtts.gTTS(text3,lang='hi')

# Save audio to BytesIO object instead of a file
audio_bytes3 = BytesIO()
converted_audio3.write_to_fp(audio_bytes3)
audio_bytes3.seek(0)

pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(audio_bytes3)
pygame.mixer.music.play()

time.sleep(2)

progress_text = "Your Camera is Loading. Please Wait..!"
my_bar = st.progress(0, text=progress_text)
for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1, text=progress_text)

# Function to detect and count thumbs up and thumbs down gestures with brackets around thumbs
def detect_and_count_gestures():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

    cap = cv2.VideoCapture(0)

    #thumbs_up_count = 0
    #thumbs_down_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmark coordinates for thumb
                thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]
                thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0]

                # Draw square brackets around the thumb
                thumb_size = 30
                cv2.rectangle(frame, (int(thumb_x - thumb_size), int(thumb_y - thumb_size)),
                              (int(thumb_x + thumb_size), int(thumb_y + thumb_size)), (0, 255, 0), 2)
                # Detect thumbs up or thumbs down gesture based on thumb position
                if thumb_y < frame.shape[0] // 2:  # Assuming thumbs up if thumb is above the center of the frame
                    #thumbs_up_count += 1
                    #break
                    cap.release()
                    cv2.destroyAllWindows()
                    command = ["streamlit", "run", "cvision.py"]  # Replace "app.py" with the name of your Streamlit app file
                    # Run the command
                    subprocess.run(command)
                    #print("hello")
                    exit
                else:
                    #thumbs_down_count += 1
                    #break
                    cap.release()
                    cv2.destroyAllWindows()
                    # Define the command to run the Streamlit app
                    command = ["streamlit", "run", "app.py"]  # Replace "app.py" with the name of your Streamlit app file
                    # Run the command
                    subprocess.run(command)
                    exit
                    


        # Display the count of thumbs up and thumbs down gestures
        #cv2.putText(frame, f"Thumbs Up: {thumbs_up_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        #cv2.putText(frame, f"Thumbs Down: {thumbs_down_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        
        # Display the frame
        cv2.imshow("Hand Gestures", frame)
        # Check if 'q' key is pressed to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    

# Call the function to detect and count gestures
detect_and_count_gestures()

