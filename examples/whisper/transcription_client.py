import sys
import pyaudio
import numpy as np
import wave
import useful_transformers as rkut
import requests



CHANNELS = 1
FORMAT = pyaudio.paInt16
RATE = 16000
RECORD_SECONDS = 5
# CHUNK_SIZE = RATE * RECORD_SECONDS
CHUNK_SIZE = 1024

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK_SIZE)
print("recording...")
frames = []
fulldata = np.empty(RATE*RECORD_SECONDS*CHANNELS)

for i in range(0, int(RATE / CHUNK_SIZE * RECORD_SECONDS)):
    data = stream.read(CHUNK_SIZE)
    numpydata = np.frombuffer(data, dtype='int16')
    frames.append(numpydata)
print("finished recording")

combin_array = np.concatenate(frames).tobytes()
url = "http://127.0.0.1:5000/api/transcription"
response = requests.post(url, data=combin_array)

# Check the response
if response.status_code == 200:
    result = response.json()
    print(result)
else:
    print('Error:', response.status_code)

stream.stop_stream()
stream.close()
p.terminate()

