
import os
import argparse
from pydub import AudioSegment

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch converts files to specified format.')
    parser.add_argument('--dir', required=True, help='Directory containing audio files.')
    parser.add_argument('--to', required=True, help='To extension')
    args = parser.parse_args()

    os.chdir(args.dir)
    files = os.listdir()

    os.mkdir('audios')

    print('Files found:')
    [print('\t{0}'.format(f)) for f in files]

    for file in files:
        name, ext = os.path.splitext(file)
        print('Converting {0}'.format(name))

        sound = AudioSegment.from_file(file, ext[1:])
        sound.export('audios'+os.sep+'{0}.{1}'.format(name, args.to), format=args.to)