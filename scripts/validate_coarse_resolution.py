import torch
import torch.nn.functional as F
import dac
import torchaudio
import random
from pathlib import Path
import soundfile as sf


ROOT = Path(__file__).parent.parent

latent_files = list(Path(ROOT / "data/latents/test").glob("*.pt"))

pt_picked = []
pt_picked = random.sample(latent_files, 10)

output_path = ROOT / "data/coarse_validation"
output_path.mkdir(parents=True, exist_ok=True)

factors = [4, 6, 8]

model = dac.DAC.load(dac.utils.download())
model.eval()
model.cuda()


for i, pt_path in enumerate(pt_picked):
    z = torch.load(pt_path).cuda() #shape [1, 1024, 344]: batch, channels, time frames
    with torch.no_grad():
        original = model.decode(z)
    audio_original = original.squeeze().cpu().numpy()
    sf.write(output_path / f"sample_{i:02d}_original.wav", audio_original, 44100)

    for factor in factors:

        #downsample
        z_coarse = F.avg_pool1d(z, kernel_size=factor, stride=factor) #slides window of size (factor size) along time axis, avgs (factor size) frames into 1

        #upsample
        z_upsample = F.interpolate(z_coarse, size=z.shape[-1], mode='linear', align_corners=False)


        #decode
        with torch.no_grad():
            reconstructed = model.decode(z_upsample) #shape: (1,1,samples)

        #export audio
        audio_out = reconstructed.squeeze().cpu().numpy()
        sf.write(output_path / f"sample_{i:02d}_factor_{factor}x.wav", audio_out, 44100) #sample_[00 - 09]_[which factor]x.wav
print(f"Done. {len(pt_picked) * len(factors)} files saved to {output_path}")
