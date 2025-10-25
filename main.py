import torch
import torchaudio
import tkinter as tk
from tkinter import filedialog
from model import SpeechModel
from features import AudioProcessor  
from decoder import CTCBeamSearchDecoder
from dataset import Dataset  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = torch.load("best_ctc_model.pth", map_location=device)

input_dim = 80
hidden_dim = 512

# Determine output_dim safely
if 'fc.weight' in checkpoint['model_state_dict']:
    output_dim = checkpoint['model_state_dict']['fc.weight'].shape[0]
else:
    output_dim = 29  # fallback

# Initialize model
model = SpeechModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Dataset and processor
dataset = Dataset()
processor = AudioProcessor()
decoder = CTCBeamSearchDecoder(
    dataset.idx2char,  # ensure list of characters
    beam_width=10,
    blank=0,
    lm_path="4-gram.arpa",
    alpha=0.5,
    beta=1.0
)

# Tkinter GUI
root = tk.Tk()
root.title("Speech-to-Text Transcription")

# Add Text widget for transcription display
transcription_textbox = tk.Text(root, height=10, width=60, wrap="word")
transcription_textbox.pack(padx=20, pady=20)

def transcribe_file():
    audio_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
    if not audio_path:
        return

    # Load audio
    waveform, sr = processor.load_audio(audio_path)

    # Extract features (Mel spectrogram)
    mel_spec = processor.compute_mel_spectrogram(waveform)  # shape: (1, n_mels, time_steps)
    print("Original Mel spectrogram shape:", mel_spec.shape)

    # Transpose to (time_steps, n_mels) for LSTM
    mel_spec = mel_spec.squeeze(0).T  # remove channel dim then transpose
    print("Transposed Mel spectrogram shape:", mel_spec.shape)

    # Add batch dimension: (1, time_steps, n_mels)
    mel_spec = mel_spec.unsqueeze(0).to(device)
    print("Final input shape to model:", mel_spec.shape)

    # Run through model
    with torch.no_grad():
        logits = model(mel_spec)
    
    
    print("Logits shape:", logits.shape)
    print("Logits sample:", logits[0, :5, :5])  # print first 5 timesteps & vocab indices\

    log_probs = logits.squeeze(0)
    print("Log_probs shape for decoder:", log_probs.shape)
    # Decode logits
    decoded_indices = decoder.decode(log_probs.cpu())
    print("Raw decoder output:", decoded_indices)

    blank = decoder.blank
    def ctc_decode(indices, blank=blank):
        decoded = []
        prev = None
        for i in indices:
            if isinstance(i, list):  # if decoder returns list of lists
                i = i[0]
            if i != blank and i != prev:
                decoded.append(i)
            prev = i
        return decoded

    indices = ctc_decode(decoded_indices)
    transcription_text = "".join(dataset.idx2char[i] for i in indices)

    print("Final transcription:", transcription_text)
    transcription_textbox.delete("1.0", tk.END)
    transcription_textbox.insert(tk.END, transcription_text)

# Button
btn = tk.Button(root, text="Select & Transcribe Audio", command=transcribe_file)
btn.pack(pady=10)

root.mainloop()
