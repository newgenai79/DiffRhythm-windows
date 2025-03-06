import torch
import torchaudio
from einops import rearrange
import argparse
import json
import os
from tqdm import tqdm
import random
import numpy as np
import time
import io
import pydub

from diffrhythm.infer.infer_utils import (
    get_reference_latent,
    get_lrc_token,
    get_style_prompt,
    prepare_model,
    get_negative_style_prompt
)

def decode_audio(latents, vae_model, chunked=False, overlap=32, chunk_size=128):
    downsampling_ratio = 2048
    io_channels = 2
    if not chunked:
        # default behavior. Decode the entire latent in parallel
        return vae_model.decode_export(latents)
    else:
        # chunked decoding
        hop_size = chunk_size - overlap
        total_size = latents.shape[2]
        batch_size = latents.shape[0]
        chunks = []
        i = 0
        for i in range(0, total_size - chunk_size + 1, hop_size):
            chunk = latents[:,:,i:i+chunk_size]
            chunks.append(chunk)
        if i+chunk_size != total_size:
            # Final chunk
            chunk = latents[:,:,-chunk_size:]
            chunks.append(chunk)
        chunks = torch.stack(chunks)
        num_chunks = chunks.shape[0]
        # samples_per_latent is just the downsampling ratio
        samples_per_latent = downsampling_ratio
        # Create an empty waveform, we will populate it with chunks as decode them
        y_size = total_size * samples_per_latent
        y_final = torch.zeros((batch_size,io_channels,y_size)).to(latents.device)
        for i in range(num_chunks):
            x_chunk = chunks[i,:]
            # decode the chunk
            y_chunk = vae_model.decode_export(x_chunk)
            # figure out where to put the audio along the time domain
            if i == num_chunks-1:
                # final chunk always goes at the end
                t_end = y_size
                t_start = t_end - y_chunk.shape[2]
            else:
                t_start = i * hop_size * samples_per_latent
                t_end = t_start + chunk_size * samples_per_latent
            #  remove the edges of the overlaps
            ol = (overlap//2) * samples_per_latent
            chunk_start = 0
            chunk_end = y_chunk.shape[2]
            if i > 0:
                # no overlap for the start of the first chunk
                t_start += ol
                chunk_start += ol
            if i < num_chunks-1:
                # no overlap for the end of the last chunk
                t_end -= ol
                chunk_end -= ol
            # paste the chunked audio into our y_final output audio
            y_final[:,:,t_start:t_end] = y_chunk[:,:,chunk_start:chunk_end]
        return y_final

def inference(cfm_model, vae_model, cond, text, duration, style_prompt, negative_style_prompt, steps, sway_sampling_coef, start_time, file_type):

    with torch.inference_mode():
        generated, _ = cfm_model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            steps=steps,
            cfg_strength=4.0,
            sway_sampling_coef=sway_sampling_coef,
            start_time=start_time
        )
        
        generated = generated.to(torch.float32)
        latent = generated.transpose(1, 2) # [b d t]
        output = decode_audio(latent, vae_model, chunked=True)

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")
        output_tensor = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).cpu()
        output_np = output_tensor.numpy().T.astype(np.float32)
        
        if file_type == 'wav':
            return (44100, output_np)
        else:
            buffer = io.BytesIO()
            output_np = np.int16(output_np * 2**15)
            song = pydub.AudioSegment(output_np.tobytes(), frame_rate=44100, sample_width=2, channels=2)
            if file_type == 'mp3':
                song.export(buffer, format="mp3", bitrate="320k")
            else:
                song.export(buffer, format="ogg", bitrate="320k")
            return buffer.getvalue()
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lrc-path', type=str, default="example/eg.lrc") # lyrics of target song
    parser.add_argument('--ref-audio-path', type=str, default="example/eg.mp3") # reference audio as style prompt for target song
    parser.add_argument('--audio-length', type=int, default=95) # length of target song
    parser.add_argument('--output-dir', type=str, default="example/output")
    args = parser.parse_args()
    
    device = 'cuda'
    
    audio_length = args.audio_length
    if audio_length == 95:
        max_frames = 2048
    elif audio_length == 285:
        max_frames = 6144
    
    cfm, tokenizer, muq, vae = prepare_model(device)
    
    with open(args.lrc_path, 'r', encoding='utf-8') as f:
        lrc = f.read()
    lrc_prompt, start_time = get_lrc_token(lrc, tokenizer, device)
    
    style_prompt = get_style_prompt(muq, args.ref_audio_path)
    
    negative_style_prompt = get_negative_style_prompt(device)
    
    latent_prompt = get_reference_latent(device, max_frames)
    
    s_t = time.time()
    generated_song = inference(cfm_model=cfm, 
                               vae_model=vae, 
                               cond=latent_prompt, 
                               text=lrc_prompt, 
                               duration=max_frames, 
                               style_prompt=style_prompt,
                               negative_style_prompt=negative_style_prompt,
                               start_time=start_time
                               )
    e_t = time.time() - s_t
    print(f"inference cost {e_t} seconds")
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "output.wav")
    torchaudio.save(output_path, generated_song, sample_rate=44100)
    