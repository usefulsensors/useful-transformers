from flask import Flask, request, jsonify
import numpy as np
from .whisper import decode_pcm

app = Flask(__name__)

@app.route('/api/transcription', methods=['POST'])
def consume_buffer():
    # Receive the audio buffer as NumPy array from the request
    array_data = np.frombuffer(request.data, dtype='int16')

    # Decode the buffer 
    result = buffer_decode(array_data)

    # Return the result as a JSON response
    return jsonify(result=result)

def buffer_decode(array):
    text = decode_pcm(array, "tiny.en")
    return {'message': text}

if __name__ == '__main__':
    app.run()
