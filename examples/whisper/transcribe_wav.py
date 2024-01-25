import os
import sys
import argparse
from .whisper import decode_wav_file

def main():
    parser = argparse.ArgumentParser(description="Transcribe WAV file to text")
    parser.add_argument('-i', '--input', type=str, help='Path to the input WAV file', default=os.path.join(os.path.dirname(__file__), 'assets', 'ever_tried.wav'))
    parser.add_argument('-o', '--output', type=str, help='Path to the output text file')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"The file {args.input} does not exist.")
        sys.exit(1)

    text = decode_wav_file(args.input)

    if args.output:
        with open(args.output, 'w') as file:
            file.write(text)
    else:
        print(text)

if __name__ == '__main__':
    main()
