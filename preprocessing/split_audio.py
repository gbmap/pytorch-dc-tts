
import os
import os.path
import argparse
from pydub import AudioSegment
from pydub.silence import split_on_silence

"""
SOURCE: https://stackoverflow.com/questions/45526996/split-audio-files-using-silence-detection
"""

# Define a function to normalize a chunk to a target amplitude.
def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)

def file_split_silence(file, chunk_name, silence_length, silence_threshold, chunk_index=0):
    print ('Loading audio {0}'.format(file))
    song = AudioSegment.from_mp3(file)

    print ('Splitting audio on silence...')
    chunks = split_on_silence (
        # Use the loaded audio.
        song, 
        # Specify that a silent chunk must be at least 2 seconds or 2000 ms long.
        min_silence_len = silence_length,
        # Consider a chunk silent if it's quieter than -16 dBFS.
        # (You may want to adjust this parameter.)
        silence_thresh = silence_threshold
    )
    print ('{0} chunks found.'.format(len(chunks)))

    chunk_dir = file[:file.rfind(os.sep)+1]
    chunk_filename = chunk_dir + chunk_name + '-{0}'.format(str(chunk_index).zfill(3)) + '-{0}.mp3'

    # Process each chunk with your parameters
    for i, chunk in enumerate(chunks):
        # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
        silence_chunk = AudioSegment.silent(duration=500)

        # Add the padding chunk to beginning and end of the entire chunk.
        audio_chunk = silence_chunk + chunk + silence_chunk

        # Normalize the entire chunk.
        normalized_chunk = match_target_amplitude(audio_chunk, -20.0)

        filename = chunk_filename.format(str(i).zfill(4)) 

        # Export the audio chunk with new bitrate.
        print("Exporting {0}".format(filename))
        normalized_chunk.export(
            filename,
            bitrate = "192k",
            format = "mp3"
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cuts audio into chunks based on silence.')
    parser.add_argument('--file', required=False, help='Audio file to be cut.')
    parser.add_argument('--targetdir', required=False, help='Target directory for chunks exported')
    parser.add_argument('--dir', required=False, help='Alternative to --file, processes whole directory')
    parser.add_argument('--silence', required=False, default=900, type=int, help='Time in ms of silence that determines a stop.')
    parser.add_argument('--threshold', required=True, default=-50, type=int, help='Volume of audio that constitutes silence.')
    parser.add_argument('--chunk_name', required=False, default='chunk', help='Name of each chunk')
    parser.add_argument('--chunk_index', required=False, default=0, type=int, help='Index of chunk')
    args = parser.parse_args()

    if args.dir is not None:
        print ('Processing folder {0}'.format(args.dir))

        files = [f for f in os.listdir(args.dir) if os.path.isfile(os.path.join(args.dir, f))]
        print('Files found:')
        [print('\t{0}'.format(f)) for f in files]
        for i, f in enumerate(files):
            file_split_silence(args.dir + os.sep + f, args.chunk_name, args.silence, args.threshold, i)        
    elif args.file is not None:
        file_split_silence(args.file, args.chunk_name, args.silence, args.threshold)
    else:
        print('Please provide either --dir or --file argument.')