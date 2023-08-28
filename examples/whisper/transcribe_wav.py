import os
import sys

from .whisper import decode_wav_file


def main():
    if len(sys.argv) < 2:
        wav_file = os.path.join(os.path.dirname(__file__), 'assets', 'ever_tried.wav')
    else:
        wav_file = sys.argv[1]
    text = decode_wav_file(wav_file)
    print(text)


if __name__ == '__main__':
    main()
