import torch
import librosa
import random
import json
from muq import MuQMuLan
from mutagen.mp3 import MP3
import os
import numpy as np
from huggingface_hub import hf_hub_download
from diffrhythm.model import DiT, CFM


def prepare_model(device):
    # prepare cfm model
    dit_ckpt_path = hf_hub_download(repo_id="ASLP-lab/DiffRhythm-base", filename="cfm_model.pt")
    dit_config_path = "./diffrhythm/config/diffrhythm-1b.json"
    with open(dit_config_path) as f:
        model_config = json.load(f)
    dit_model_cls = DiT
    cfm = CFM(
                transformer=dit_model_cls(**model_config["model"], use_style_prompt=True),
                num_channels=model_config["model"]['mel_dim'],
                use_style_prompt=True
             )
    cfm = cfm.to(device)
    cfm = load_checkpoint(cfm, dit_ckpt_path, device=device, use_ema=False)
    
    # prepare tokenizer
    tokenizer = CNENTokenizer()
    
    # prepare muq
    muq = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")
    muq = muq.to(device).eval()
    
    # prepare vae
    vae_ckpt_path = hf_hub_download(repo_id="ASLP-lab/DiffRhythm-vae", filename="vae_model.pt")
    vae = torch.jit.load(vae_ckpt_path, map_location='cpu').to(device)
    return cfm, tokenizer, muq, vae
    

# for song edit, will be added in the future
def get_reference_latent(device, max_frames):
    return torch.zeros(1, max_frames, 64).to(device)

def get_negative_style_prompt(device):
    file_path = "./src/negative_prompt.npy"
    vocal_stlye = np.load(file_path)
    
    vocal_stlye = torch.from_numpy(vocal_stlye).to(device) # [1, 512]
    vocal_stlye = vocal_stlye.half()
    
    return vocal_stlye

def get_style_prompt(model, wav_path):
    mulan = model
    
    ext = os.path.splitext(wav_path)[-1].lower()
    if ext == '.mp3':
        meta = MP3(wav_path)
        audio_len = meta.info.length
    elif ext in ['.wav', '.flac']:
        audio_len = librosa.get_duration(path=wav_path)
    else:
        raise ValueError("Unsupported file format: {}".format(ext))
    
    assert audio_len >= 1, "Input audio length shorter than 1 second"
    
    if audio_len >= 10:
        mid_time = audio_len // 2
        start_time = mid_time - 5
        wav, _ = librosa.load(wav_path, sr=24000, offset=start_time, duration=10)
    
    else:
        wav, _ = librosa.load(wav_path, sr=24000)
    wav = torch.tensor(wav).unsqueeze(0).to(model.device)
    
    with torch.no_grad():
        audio_emb = mulan(wavs = wav) # [1, 512]
        
    audio_emb = audio_emb
    audio_emb = audio_emb.half()

    return audio_emb

def parse_lyrics(lyrics: str):
    lyrics_with_time = []
    lyrics = lyrics.strip()
    for line in lyrics.split('\n'):
        try:
            time, lyric = line[1:9], line[10:]
            lyric = lyric.strip()
            mins, secs = time.split(':')
            secs = int(mins) * 60 + float(secs)
            lyrics_with_time.append((secs, lyric))
        except:
            continue
    return lyrics_with_time

class CNENTokenizer():
    def __init__(self):
        with open('./diffrhythm/g2p/g2p/vocab.json', 'r', encoding='utf-8') as file:
            self.phone2id:dict = json.load(file)['vocab']
        self.id2phone = {v:k for (k, v) in self.phone2id.items()}
        # from f5_tts.g2p.g2p_generation import chn_eng_g2p
        from diffrhythm.g2p.g2p_generation import chn_eng_g2p
        self.tokenizer = chn_eng_g2p
    def encode(self, text):
        phone, token = self.tokenizer(text)
        token = [x+1 for x in token]
        return token
    def decode(self, token):
        return "|".join([self.id2phone[x-1] for x in token])
    
def get_lrc_token(text, tokenizer, device):

    max_frames = 2048
    lyrics_shift = 0
    sampling_rate = 44100
    downsample_rate = 2048
    max_secs = max_frames / (sampling_rate / downsample_rate)
   
    pad_token_id = 0
    comma_token_id = 1
    period_token_id = 2    

    lrc_with_time = parse_lyrics(text)
    
    modified_lrc_with_time = []
    for i in range(len(lrc_with_time)):
        time, line = lrc_with_time[i]
        line_token = tokenizer.encode(line)
        modified_lrc_with_time.append((time, line_token))
    lrc_with_time = modified_lrc_with_time

    lrc_with_time = [(time_start, line) for (time_start, line) in lrc_with_time if time_start < max_secs]
    lrc_with_time = lrc_with_time[:-1] if len(lrc_with_time) >= 1 else lrc_with_time
    
    normalized_start_time = 0.

    lrc = torch.zeros((max_frames,), dtype=torch.long)

    tokens_count = 0
    last_end_pos = 0
    for time_start, line in lrc_with_time:
        tokens = [token if token != period_token_id else comma_token_id for token in line] + [period_token_id]
        tokens = torch.tensor(tokens, dtype=torch.long)
        num_tokens = tokens.shape[0]

        gt_frame_start = int(time_start * sampling_rate / downsample_rate)
        
        frame_shift = random.randint(int(lyrics_shift), int(lyrics_shift))
        
        frame_start = max(gt_frame_start - frame_shift, last_end_pos)
        frame_len = min(num_tokens, max_frames - frame_start)

        #print(gt_frame_start, frame_shift, frame_start, frame_len, tokens_count, last_end_pos, full_pos_emb.shape)

        lrc[frame_start:frame_start + frame_len] = tokens[:frame_len]

        tokens_count += num_tokens
        last_end_pos = frame_start + frame_len   
        
    lrc_emb = lrc.unsqueeze(0).to(device)
    
    normalized_start_time = torch.tensor(normalized_start_time).unsqueeze(0).to(device)
    normalized_start_time = normalized_start_time.half()
    
    return lrc_emb, normalized_start_time

def load_checkpoint(model, ckpt_path, device, use_ema=True):
    if device == "cuda":
        model = model.half()

    ckpt_type = ckpt_path.split(".")[-1]
    if ckpt_type == "safetensors":
        from safetensors.torch import load_file

        checkpoint = load_file(ckpt_path)
    else:
        checkpoint = torch.load(ckpt_path, weights_only=True)

    if use_ema:
        if ckpt_type == "safetensors":
            checkpoint = {"ema_model_state_dict": checkpoint}
        checkpoint["model_state_dict"] = {
            k.replace("ema_model.", ""): v
            for k, v in checkpoint["ema_model_state_dict"].items()
            if k not in ["initted", "step"]
        }
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    else:
        if ckpt_type == "safetensors":
            checkpoint = {"model_state_dict": checkpoint}
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    return model.to(device)