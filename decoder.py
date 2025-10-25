import numpy as np
from collections import defaultdict

class CTCBeamSearchDecoder:
    def __init__(self, idx2char, beam_width=10, blank=0, lm_path=None, alpha=0.5, beta=1.0):

        self.idx2char = idx2char
        self.beam_width = beam_width
        self.blank = blank
        self.alpha = alpha
        self.beta = beta

        

    def decode(self, log_probs):
        # log_probs: (T, C)
        if log_probs.dim() == 3:
            log_probs = log_probs.squeeze(1)  # remove batch dim if present

        T, vocab_size = log_probs.shape
        log_probs = log_probs.cpu().detach().numpy()

        # Beam: seq -> (log_prob_blank, log_prob_non_blank, lm_state)
        beam = {(): (0.0, -np.inf, None)}

        for t in range(T):
            next_beam = defaultdict(lambda: (-np.inf, -np.inf, None))
            for seq, (p_b, p_nb, state) in beam.items():
                for c in range(vocab_size):
                    p = log_probs[t, c]

                    if c == self.blank:
                        nb = next_beam[seq]
                        next_beam[seq] = (
                            np.logaddexp(nb[0], p_b),
                            np.logaddexp(nb[1], p_nb),
                            state
                        )
                    else:
                        new_seq = seq + (c,)

                        # LM score
                        lm_score = 0.0
                        new_state = state
                        if self.lm and len(seq) == 0:
                            lm_score = self.alpha * self.lm.score(self.idx2char[c], bos=True, eos=False)
                        elif self.lm and len(seq) > 0:
                            prev_text = "".join([self.idx2char[i] for i in seq if i != self.blank])
                            candidate_text = prev_text + self.idx2char[c]
                            lm_score = self.alpha * self.lm.score(candidate_text, bos=True, eos=False)

                        # Word insertion bonus
                        if self.idx2char[c] == " ":
                            lm_score += self.beta

                        if len(seq) > 0 and seq[-1] == c:
                            nb = next_beam[new_seq]
                            next_beam[new_seq] = (
                                nb[0],
                                np.logaddexp(nb[1], p_b + p + lm_score),
                                new_state
                            )
                        else:
                            nb = next_beam[new_seq]
                            next_beam[new_seq] = (
                                nb[0],
                                np.logaddexp(nb[1], max(p_b, p_nb) + p + lm_score),
                                new_state
                            )

            # Prune beams
            beam = dict(sorted(
                next_beam.items(),
                key=lambda x: np.logaddexp(x[1][0], x[1][1]),
                reverse=True
            )[:self.beam_width])

        # Pick best sequence
        best_seq, (p_b, p_nb, _) = max(
            beam.items(),
            key=lambda x: np.logaddexp(x[1][0], x[1][1])
        )

        # Convert to characters
        decoded = []
        prev = self.blank
        for c in best_seq:
            if c != prev and c != self.blank:
                decoded.append(int(c))
            prev = c

        return decoded
