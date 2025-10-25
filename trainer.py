import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import Dataset
from model import SpeechModel
from decoder import CTCBeamSearchDecoder
import editdistance
import numpy as np
import os


def cer(pred_text, target_text):
    if len(target_text) == 0:
        return 1.0 if len(pred_text) > 0 else 0.0
    return editdistance.eval(pred_text, target_text) / len(target_text)


def evaluate(model, loader, criterion, dataset, decoder, device, logit_scale=None, idx2char=None):
    model.eval()
    total_loss, total_cer, num_samples = 0.0, 0.0, 0

    with torch.no_grad():
        for features, transcripts, feat_lens, trans_lens in loader:
            features, transcripts = features.to(device), transcripts.to(device)
            feat_lens, trans_lens = feat_lens.to(device), trans_lens.to(device)

            logits = model(features, feat_lens)
            if logit_scale is not None:
                logits = logits * logit_scale

            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)
            loss = criterion(log_probs, transcripts, feat_lens, trans_lens)
            total_loss += loss.item()

            preds = decoder.decode(log_probs)
            start = 0
            for i, length in enumerate(trans_lens):
                target_seq = transcripts[start:start+length].cpu().numpy().tolist()
                start += length
                mapping = idx2char or dataset.idx2char
                target_text = "".join([mapping[c] for c in target_seq if c in mapping])
                pred_text = preds[i] if isinstance(preds[i], str) else "".join([mapping.get(c, '') for c in preds[i]])
                total_cer += cer(pred_text, target_text)
                num_samples += 1

    avg_loss = total_loss / len(loader)
    avg_cer = total_cer / num_samples
    return avg_loss, avg_cer


def collate_fn(batch):
    features, transcripts = zip(*batch)
    feature_lengths = [f.shape[0] for f in features]
    max_len = max(feature_lengths)
    feat_dim = features[0].shape[1]
    features_padded = torch.zeros(len(batch), max_len, feat_dim)
    for i, f in enumerate(features):
        features_padded[i, :f.shape[0], :] = f
    transcript_lengths = [len(t) for t in transcripts]
    transcripts_concat = torch.cat(transcripts)
    return (features_padded,
            transcripts_concat,
            torch.tensor(feature_lengths, dtype=torch.long),
            torch.tensor(transcript_lengths, dtype=torch.long))


def train_ctc(num_epochs=50, batch_size=8, lr=1e-3, hidden_dim=512, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = Dataset()
    train_loader = DataLoader(dataset.get_all_data(), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset.get_validation_data(), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    features, transcripts, feat_lens, trans_lens = next(iter(train_loader))
    input_dim = features.shape[2]
    output_dim = len(dataset.char2idx) + 1
    model = SpeechModel(input_dim, hidden_dim, output_dim).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    logit_scale = nn.Parameter(torch.tensor(5.0, dtype=torch.float32, device=device), requires_grad=True)
    optimizer = torch.optim.Adam(list(model.parameters()) + [logit_scale], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    decoder = CTCBeamSearchDecoder(idx2char=dataset.idx2char, blank=0, beam_width=10)
    best_val_cer = float("inf")

    # Metric logs
    train_losses, val_losses, train_cers, val_cers = [], [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss, total_cer, num_batches, num_samples = 0.0, 0.0, 0, 0

        for features, transcripts, feat_lens, trans_lens in train_loader:
            features, transcripts = features.to(device), transcripts.to(device)
            feat_lens, trans_lens = feat_lens.to(device), trans_lens.to(device)

            optimizer.zero_grad()
            logits = model(features, feat_lens) * logit_scale
            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)
            loss = criterion(log_probs, transcripts, feat_lens, trans_lens)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Compute quick greedy CER per batch for running display
            greedy = log_probs.argmax(dim=-1).transpose(0, 1)
            start = 0
            for i, length in enumerate(trans_lens):
                target_seq = transcripts[start:start+length].cpu().numpy().tolist()
                start += length
                mapping = dataset.idx2char
                target_text = "".join([mapping[c] for c in target_seq if c in mapping])
                pred_text = "".join([mapping.get(c, '') for c in greedy[i].cpu().numpy()])
                total_cer += cer(pred_text, target_text)
                num_samples += 1

        avg_train_loss = total_loss / num_batches
        avg_train_cer = total_cer / num_samples
        avg_val_loss, avg_val_cer = evaluate(model, val_loader, criterion, dataset, decoder, device, logit_scale, dataset.idx2char)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Train CER: {avg_train_cer:.4f} | Val CER: {avg_val_cer:.4f}")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_cers.append(avg_train_cer)
        val_cers.append(avg_val_cer)

        scheduler.step(avg_val_cer)

        # Save best model (lowest validation CER)
        if avg_val_cer < best_val_cer:
            best_val_cer = avg_val_cer
            np.save("best_model.npy", {
                
            })
            print(f"Saved best model at epoch {epoch+1} (Val CER={best_val_cer:.4f})")

        # Save logs after each epoch
        np.save("training_metrics.npy", {
            'train_loss': np.array(train_losses),
            'val_loss': np.array(val_losses),
            'train_cer': np.array(train_cers),
            'val_cer': np.array(val_cers)
        })

    print(f"Training completed. Best Validation CER: {best_val_cer:.4f}")


if __name__ == "__main__":
    train_ctc()
