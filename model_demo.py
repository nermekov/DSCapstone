import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt
import io



class DenoisingAutoencoder(nn.Module):
    def __init__(self, use_lstm=True):
        super().__init__()
        self.use_lstm = use_lstm

        # Encoder 
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1), 
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.ReLU()
        )

        # LSTM bottleneck
        if self.use_lstm:
            self.lstm = nn.LSTM(
                input_size=128,
                hidden_size=128,
                num_layers=2,
                batch_first=True,
                bidirectional=True
            )
            self.lstm_linear = nn.Linear(128 * 2, 128)

        # Decoder 
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),  # T/4
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(64 + 64, 32, kernel_size=4, stride=2, padding=1),  # T/2
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(32 + 32, 1, kernel_size=4, stride=2, padding=1)   # T
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)    
        enc2 = self.enc2(enc1)  
        enc3 = self.enc3(enc2)  
        # Bottleneck
        if self.use_lstm:
            lstm_input = enc3.permute(0, 2, 1)        
            lstm_out, _ = self.lstm(lstm_input)      
            encoded = self.lstm_linear(lstm_out)     
            encoded = encoded.permute(0, 2, 1)       
        else:
            encoded = enc3

        # Decoder with skip connections
        dec3 = self.dec3(encoded)                                
        dec2 = self.dec2(torch.cat([dec3, enc2], dim=1))         
        dec1 = self.dec1(torch.cat([dec2, enc1], dim=1))         

        return dec1


def run_denoising_streamlit(st, mixture_path, checkpoint_path="chosen_checkpoints/dac 4 3 good aug.pth", sample_rate=16000, device="cpu"):
    model = DenoisingAutoencoder()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    st.success("Model loaded.")
    model.eval()
    model = model.to(device)
    mixture, sr_m = torchaudio.load(mixture_path)
    if sr_m != sample_rate:
        mixture = torchaudio.functional.resample(mixture, sr_m, sample_rate)
    mixture = torch.mean(mixture, dim=0, keepdim=True)
    max_samples = sample_rate * 12
    mixture = mixture[:, :max_samples]
    total_samples = mixture.shape[1]
    SEGMENT_SAMPLES = sample_rate * 4
    predicted_segments = []
    with torch.no_grad():
        for start in range(0, total_samples, SEGMENT_SAMPLES):
            end = start + SEGMENT_SAMPLES
            segment = mixture[:, start:end]
            if segment.shape[-1] < SEGMENT_SAMPLES:
                pad_len = SEGMENT_SAMPLES - segment.shape[-1]
                segment = F.pad(segment, (0, pad_len))
            input_tensor = segment.unsqueeze(0).to(device)
            pred_segment = model(input_tensor)
            pred_segment = pred_segment.squeeze().cpu().clamp(-1, 1)
            predicted_segments.append(pred_segment[:segment.shape[-1]])
    pred_music = torch.cat(predicted_segments, dim=-1)
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    t = torch.linspace(0, mixture.shape[1] / sample_rate, mixture.shape[1])
    axs[0].plot(t, mixture.squeeze(), label="Noisy mixture")
    axs[0].set_title("Noisy Mixture")
    axs[1].plot(t[:pred_music.shape[-1]], pred_music, label="Predicted Music", color="green")
    axs[1].set_title("Predicted Clean Music")
    for ax in axs:
        ax.grid()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    st.image(buf, caption="Denoising Results")
    plt.close()
    st.audio(mixture.squeeze().numpy(), sample_rate=sample_rate, format='audio/wav')
    st.audio(pred_music.numpy(), sample_rate=sample_rate, format='audio/wav')