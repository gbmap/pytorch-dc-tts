
import os
import re
import codecs
import unicodedata
import numpy as np

from utils import log

from torch.utils.data import Dataset

DEFAULT_DATASET_DIR_NAME = 'LJSpeech-1.1'

vocab = "PE abcdefghijklmnopqrstuvwxyz'.?"  # P: Padding, E: EOS.
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}

def text_normalize(text, vocab=vocab):
    """
    Removes characters that are not in <vocab>. Also lowercases.
    """
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def read_metadata_from_file(filename):
    """
    Loads metadata table from a text file with columns:
    FILENAME | TEXT LENGTH | TEXT
    """
    fnames, text_lengths, texts = [], [], []
    transcript = os.path.join(filename)
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()
    for line in lines:
        fname, _, text = line.strip().split("|")

        fnames.append(fname)

        text = text_normalize(text) + "E"  # E: EOS
        text = [char2idx[char] for char in text]
        text_lengths.append(len(text))
        texts.append(np.array(text, np.long))

    return fnames, text_lengths, texts

def get_test_data(sentences, max_n):
    """

    """
    normalized_sentences = [text_normalize(line).strip() + "E" for line in sentences]  # text normalization, E: EOS
    texts = np.zeros((len(normalized_sentences), max_n + 1), np.long)
    for i, sent in enumerate(normalized_sentences):
        texts[i, :len(sent)] = [char2idx[char] for char in sent]
    return texts

class SpeechDataset(Dataset):
    def __init__(self, keys, dir_name=DEFAULT_DATASET_DIR_NAME):
        self.keys = keys
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dir_name)
        self.fnames, self. text_lengths, self.texts = read_metadata_from_file(os.path.join(self.path, 'metadata.csv'))

    def slice(self, start, end):
        self.fnames = self.fnames[start:end]
        self.text_lengths = self.text_lengths[start:end]
        self.texts = self.texts[start:end]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        data = {}
        if 'texts' in self.keys:
            data['texts'] = self.texts[index]
        if 'mels' in self.keys:
            # (39, 80)
            data['mels'] = np.load(os.path.join(self.path, 'mels', "%s.npy" % self.fnames[index]))
        if 'mags' in self.keys:
            # (39, 80)
            data['mags'] = np.load(os.path.join(self.path, 'mags', "%s.npy" % self.fnames[index]))
        if 'mel_gates' in self.keys:
            data['mel_gates'] = np.ones(data['mels'].shape[0], dtype=np.int)  # TODO: because pre processing!
        if 'mag_gates' in self.keys:
            data['mag_gates'] = np.ones(data['mags'].shape[0], dtype=np.int)  # TODO: because pre processing!
        return data



if __name__ == '__main__':

    ds = SpeechDataset(['texts', 'mels', 'mel_gates'], dir_name=DEFAULT_DATASET_DIR_NAME)
    log(ds[0])
    

    #print(text_normalize('ISSO é UM TEXTO MANEIRO | [porra] AÍ CARLAHO', vocab))