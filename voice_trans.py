'''
import googletrans
import speech_recognition
import gtts
import playsound
recognizer=speech_recognition.Recognizer()
with speech_recognition.Microphone() as source:
    print("Speak Now")
    voice=recognizer.listen(source)
    text=recognizer.recognize_google(voice,language="bn")
    print(text)
#print(googletrans.LANGUAGES)
# hi: hindi,bn: bengali, en:english
translator=googletrans.Translator()
translaton=translator.translate(text,dest="en")
print(translaton.text)

converted_audio=gtts.gTTS(translaton.text,lang="en")
playsound.playsound(converted_audio)
'''
import googletrans
import speech_recognition
import gtts
import pygame
from io import BytesIO
print(googletrans.LANGUAGES)
recognizer = speech_recognition.Recognizer()
#lang_input="bn or hi" 
with speech_recognition.Microphone() as source:
    print("Speak Now")
    voice = recognizer.listen(source)
    text = recognizer.recognize_google(voice, language="bn")
    print(text)

translator = googletrans.Translator()
translation = translator.translate(text, dest="en")
print(translation.text)

converted_audio = gtts.gTTS(translation.text, lang="en")

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
