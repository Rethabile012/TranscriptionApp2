import numpy as np
from collections import defaultdict
import torch

class CTCBeamSearchDecoder:
    def __init__(self, idx2char, blank=0, beam_width=10):
        self.idx2char = idx2char
        self.blank = blank
        self.beam_width = beam_width

    def decode(self, log_probs):
        

        # Greedy decoding for simplicity
        indices = torch.argmax(log_probs, dim=-1)  # [T]
        pred_chars = []
        prev_idx = None

        for c in indices.cpu().numpy():
            if c == self.blank:
                prev_idx = None
                continue  # skip blank
            if c == prev_idx:
                continue  # collapse repeats
            char = self.idx2char.get(c, "")  # safe get
            pred_chars.append(char)
            prev_idx = c

        return "".join(pred_chars)
