import librosa
import torch
import dac
from pathlib import Path

ROOT = Path(__file__).parent.parent
folder_path = ROOT / "data/test/audio"
output_path = ROOT / "data/latents/test"
output_path.mkdir(parents=True, exist_ok=True)

model = dac.DAC.load(dac.utils.download())
model.eval()
model.cuda()

for wav_path in folder_path.glob("*.wav"):
    y, sr = librosa.load(wav_path, sr=44100, mono=True)
    audio = torch.tensor(y).unsqueeze(0).unsqueeze(0).cuda()

    with torch.no_grad():
        z, codes, latents, _, _ = model.encode(audio)

    torch.save(z.cpu().float(), output_path / (wav_path.stem + ".pt"))

