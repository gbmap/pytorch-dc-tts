import os
import sys
import speech_recognition as sr
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extracts text from a directory of audios into a .csv.')
    parser.add_argument('--dir', required=True, help='Directory containing audio files.')
    parser.add_argument('--filename', required=True, help='Sheet filename')
    parser.add_argument('--separator', required=False, help='.csv separator (default=|)', default='|')
    parser.add_argument('--overwrite', required=False, default=False, type=bool, help='Overwrite sheet if it already exists')
    args = parser.parse_args()

    os.chdir(args.dir)

    files = [f for f in os.listdir() if os.path.isfile(f)]

    print('Files found:')
    [print('\t{0}'.format(f)) for f in files]

    sheet_exists = os.path.isfile(args.filename) 
    mode = 'w' if not sheet_exists or args.overwrite else 'a+'
    print('Loading sheet {1} with mode ({0})'.format(mode, args.filename))

    # If appending, skip to last file in sheet
    if mode == 'a+':
        with open(args.filename, 'r') as f:
            last_line = f.read().splitlines()[-1] # cover your eyes

            print('Last line in sheet: {0}'.format(last_line))
            last_file = last_line.split(args.separator)[0]
            last_file_index = files.index(last_file+'.wav')

            files = files[last_file_index+1:]
            print('Skipping to file {0}'.format(files[0]))

    print('Starting extraction...')
    with open('metadata.csv', mode) as f:
        for file in files:
            name, ext = os.path.splitext(file)
            if ext != '.wav':
                continue

            r = sr.Recognizer()
            with sr.AudioFile(os.path.join(args.dir, file)) as source:
                audio_data = r.record(source)
                try:
                    text = r.recognize_google(audio_data)
                    if len(text) <= 5:
                        continue

                    output = '{0}{1}{2}{1}{2}\n'.format(name, args.separator, text)
                    print (output)
                    f.write(output)
                except sr.UnknownValueError:
                    continue






