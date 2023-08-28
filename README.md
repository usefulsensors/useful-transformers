# Useful Transformers
Useful Transformers is a library for efficient inference of Transformer models. The focus is on low cost, low energy processors to run inference at the edge. The initial implementation is aimed at running OpenAI's [Whisper](https://github.com/openai/whisper) speech-to-text model efficiently on the [RK3588](https://www.rock-chips.com/a/en/products/RK35_Series/2022/0926/1660.html) processors' based single-board computers. The tiny.en Whisper model runs transcribes speech at 30x real-time speeds, and 2x better than best [known](https://github.com/guillaumekln/faster-whisper) implementation.

## Getting started

The easiest way to try out Whisper transcription is to install the [release](https://github.com/usefulsensors/useful-transformers/releases/download/0.1_rk3588/useful_transformers-0.1-cp310-cp310-linux_aarch64.whl) wheel package.

    # Preferably inside a virtual environment
    $ python -m pip install https://github.com/usefulsensors/useful-transformers/releases/download/0.1_rk3588/useful_transformers-0.1-cp310-cp310-linux_aarch64.whl

 Try transcribing a wav file.

    $ taskset -c 4-7 python -m useful_transformers.transcribe_wav <wav_file>

If you don't have a wav file handy, running the above command will transcribe an example provided in the package.

