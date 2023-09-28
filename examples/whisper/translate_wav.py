import os
import sys

from .whisper import decode_wav_file


def main():
    if len(sys.argv) < 2:
        wav_file = os.path.join(os.path.dirname(__file__), 'assets', 'chinese.wav')
        src_lang = 'zh'
    else:
        wav_file = sys.argv[1]
        src_lang = sys.argv[2]
    text = decode_wav_file(wav_file, model='tiny', task='translate', src_lang=src_lang)
    print(text)


if __name__ == '__main__':
    main()
