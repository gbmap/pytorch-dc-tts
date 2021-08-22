#!/usr/bin/env python
"""Synthetize sentences into speech."""
__author__ = 'Erdene-Ochir Tuguldur'

import os
import sys
import argparse
from tqdm import *

import numpy as np
import torch

from models                  import Text2Mel, SSRN
from hparams                 import HParams as hp
from audio                   import save_to_wav
from utils                   import get_last_checkpoint_file_name, load_checkpoint, save_to_png
from datasets.speech_dataset import vocab, get_test_data


#
#  Parse arguments and global parameters
#
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", required=True, help='dataset name')
parser.add_argument("--sentences", required=False, help="file containing sentences")
parser.add_argument("--sentence", required=False, help="single sentence")
parser.add_argument("--output_dir", required=False, help="output directory")

args = parser.parse_args()

if args.sentence is not None and len(args.sentence) > 0:
    SENTENCES = [args.sentence]
else:
    with open(args.sentences, 'r') as f:
        SENTENCES = f.readlines()

if args.output_dir is not None and len(args.output_dir) > 0:
    OUTPUT_DIR = args.output_dir
else:
    OUTPUT_DIR = 'samples'
    

torch.set_grad_enabled(False)


#
#   Load Text2Mel checkpoint 
#
text2mel = Text2Mel(vocab).eval()
last_checkpoint_file_name = get_last_checkpoint_file_name(os.path.join(hp.logdir, '%s-text2mel' % args.dataset))
# last_checkpoint_file_name = 'logdir/%s-text2mel/step-020K.pth' % args.dataset
if last_checkpoint_file_name:
    print("loading text2mel checkpoint '%s'..." % last_checkpoint_file_name)
    load_checkpoint(last_checkpoint_file_name, text2mel, None)
else:
    print("text2mel not exits")
    sys.exit(1)



#
#   Load SSRN checkpoint 
#
ssrn = SSRN().eval()
last_checkpoint_file_name = get_last_checkpoint_file_name(os.path.join(hp.logdir, '%s-ssrn' % args.dataset))
if last_checkpoint_file_name:
    print("loading ssrn checkpoint '%s'..." % last_checkpoint_file_name)
    load_checkpoint(last_checkpoint_file_name, ssrn, None)
else:
    print("ssrn not exits")
    sys.exit(1)



# synthetize by one by one because there is a batch processing bug!
for i in range(len(SENTENCES)):
    sentences = [SENTENCES[i]]

    max_N = len(SENTENCES[i])
    L = torch.from_numpy(get_test_data(sentences, max_N)).to(torch.long)
    zeros = torch.from_numpy(np.zeros((1, hp.n_mels, 1), np.float32))
    Y = zeros
    A = None

    for t in tqdm(range(hp.max_T)):
        _, Y_t, A = text2mel(L, Y, monotonic_attention=True)
        Y = torch.cat((zeros, Y_t), -1)
        _, attention = torch.max(A[0, :, -1], 0)
        attention = attention.item()
        if L[0, attention] == vocab.index('E'):  # EOS
            break

    _, Z = ssrn(Y)

    Y = Y.cpu().detach().numpy()
    A = A.cpu().detach().numpy()
    Z = Z.cpu().detach().numpy()

    save_to_png(f'{OUTPUT_DIR}/%d-att.png' % (i + 1), A[0, :, :])
    save_to_png(f'{OUTPUT_DIR}/%d-mel.png' % (i + 1), Y[0, :, :])
    save_to_png(f'{OUTPUT_DIR}/%d-mag.png' % (i + 1), Z[0, :, :])
    save_to_wav(Z[0, :, :].T, f'{OUTPUT_DIR}/%d-wav.wav' % (i + 1))
