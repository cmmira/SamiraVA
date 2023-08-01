from gtts import gTTS
import speech_recognition as sr
from playsound import playsound
import random
import datetime
hour = datetime.datetime.now().strftime('%H:%M')
date = datetime.date.today().strftime('%d/%B/%Y')
date = date.split('/')

import webbrowser as wb
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from Mods import ComRes, Agenda
commands = ComRes.commands
responses = ComRes.responses 

va_name = 'Samira '

chrome_path = 'open -a /Applications/Google\ Chrome.app %s'
def search(sentence):
    wb.get(chrome_path).open('https://www.google.com/search?q=' + sentence)

MODEL_TYPES = ['EMOTION']
def load_model_by_name(model_type):
    if model_type == MODEL_TYPES[0]:
        model = tf.keras.models.load_model('Model/speech_emotion_recognition.hdf5')
        model_dict = list(['calm', 'happy', 'fear', 'nervous', 'neutral', 'disgust', 'surprise', 'sad'])
        SAMPLE_RATE = 48000
    return model, model_dict, SAMPLE_RATE
model_type = 'EMOTION'
loaded_model = load_model_by_name(model_type)

def predict_sound(AUDIO, SAMPLE_RATE, plot=True):
    results= []
    wav_data, sample_rate = librosa.load(AUDIO, sr = SAMPLE_RATE)
    clip, index = librosa.effects.trim(wav_data, top_db=60, frame_length=512, hop_length=64)
    splitted_audio_data = tf.signal.frame(clip, sample_rate, sample_rate, pad_end = True, pad_value=0)
    for i, data in enumerate(splitted_audio_data.numpy()):
        mfccs_features = librosa.feature.mfcc(y = data, sr = sample_rate, n_mfcc = 40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis = 0)
        mfccs_scaled_features = mfccs_scaled_features.reshape(1, -1)
        mfccs_scaled_features = mfccs_scaled_features[:, :, np.newaxis]
        predictions = loaded_model[0].predict(mfccs_scaled_features)
        if plot:
            plt.figure(figsize=(len(splitted_audio_data), 5))
            plt.barh(loaded_model[1], predictions[0])
            plt.tight_layout()
            plt.show()

        predictions = predictions.argmax(axis = 1)
        predictions = predictions.astype(int).flatten()
        predictions = loaded_model[1][predictions[0]]
        results.append(predictions)
        result_str = 'PART' + str(i) + ': ' + str(predictions).upper()
    count_results = [[results.count(x), x] for x in set(results)]
    return max(count_results)

def play_music_youtube(emotion):
    play = False 
    if emotion == 'sad' or emotion == 'fear':
        wb.get(chrome_path).open('https://www.youtube.com/watch?v=C0EYKxF1oTI')
        play = True
    if emotion == 'nervous' or emotion == 'surprise':
        wb.get(chrome_path).open('https://www.youtube.com/watch?v=BywDOO99Ia0')
        play = True
    return play

def speak(text):
    tts = gTTS(text, lang='en')
    tts.save('speak.mp3')
    playsound('speak.mp3')

def listen_microphone():
    microphone = sr.Recognizer()
    with sr.Microphone() as source:
        microphone.adjust_for_ambient_noise(source, duration=0.8)
        print('Listening: ')
        audio = microphone.listen(source)
        with open('Recordings/speech.wav', 'wb') as f:
            f.write(audio.get_wav_data())
        try:
            sentence = microphone.recognize_google(audio, language = 'en-US')
            print('You said: ' + sentence)
        except sr.UnknownValueError:
            sentence = ''
            print('Not understood')
        return sentence

def test_models():
    audio_source = 'Recordings/speech.wav'
    prediction = predict_sound(audio_source, loaded_model[2], plot = False)
    return prediction

playing = False
mode_control = False
print('[INFO] Ready to start!')
playsound('n1.mp3')

while(1):
    result = listen_microphone()

    if va_name in result:
        act = result.split(" ",1)
        if len(act) == 1: continue
        result = act[1]
        result = result.lower()

        if result in commands[0]:
            playsound('n2.mp3')
            speak('These are my functionalities: ' + responses[0])

        if result in commands[3]:
            playsound('n2.mp3')
            speak('It is now ' + datetime.datetime.now().strftime('%H:%M'))

        if result in commands[4]:
            playsound('n2.mp3')
            speak('Today is ' + date[0] + ' of ' + date[1])

        if result in commands[1]:
            playsound('n2.mp3')
            speak('Please, tell me the activity!')
            result = listen_microphone()
            annotation = open('annotation.txt', mode='a+', encoding='utf-8')
            annotation.write(result + '\n')
            annotation.close()
            speak(''.join(random.sample(responses[1],k=1)))
            speak('Should I read the notes?')
            result = listen_microphone()
            if result == 'yes' or result == 'sure':
                with open('annotation.txt') as file_source:
                    lines = file_source.readlines()
                    for line in lines:
                        speak(line)
            else:
                speak('Ok!')
        
        if result in commands[2]:
            playsound('n2.mp3')
            speak(''.join(random.sample(responses[2], k=1)))
            result = listen_microphone()
            search(result)

        if result in commands[6]:
            playsound('n2.mp3')
            if Agenda.load_agenda():
                speak('These are the events for today:')
                for i in range(len(Agenda.load_agenda()[1])):
                    speak(Agenda.load_agenda()[1][i] + ' ' + Agenda.load_agenda()[0][i] + ' schedule ' + str(Agenda.load_agenda()[2][i]))
            else:
                speak('There are no events for today considering the current time!')

        if result in commands[5]:
            mode_control = True
            playsound('n1.mp3')
            speak('Emotion analysis mode has been activated!')

        if mode_control:
            analyze = test_models()
            print(f'I heard {analyze} in your voice!')
            if not playing:
                playing = play_music_youtube(analyze[1])

        if result == 'turn off':
            playsound('n2.mp3')
            speak(''.join(random.sample(responses[4],k=1)))
            break

    else:
        playsound('n3.mp3')
