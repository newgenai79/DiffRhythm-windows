import gradio as gr
import torch
import torchaudio
from einops import rearrange
import random
import numpy as np
import os
from transformers import AutoTokenizer
import time
from infer.infer_utils import (
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
    prepare_model,
)
# Global variables
MAX_SEED = np.iinfo(np.int32).max
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def inference(
    cfm_model,
    vae_model,
    cond,
    text,
    duration,
    style_prompt,
    negative_style_prompt,
    start_time,
    chunked=True,
):
    with torch.inference_mode():
        generated, _ = cfm_model.sample(
            cond=cond,
            text=text,
            duration=duration,
            style_prompt=style_prompt,
            negative_style_prompt=negative_style_prompt,
            steps=32,
            cfg_strength=4.0,
            start_time=start_time,
        )

        generated = generated.to(torch.float32)
        latent = generated.transpose(1, 2)  # [b d t]

        output = decode_audio(latent, vae_model, chunked=chunked)

        # Rearrange audio batch to a single sequence
        output = rearrange(output, "b d n -> d (b n)")
        # Peak normalize, clip, convert to int16, and save to file
        output = (
            output.to(torch.float32)
            .div(torch.max(torch.abs(output)))
            .clamp(-1, 1)
            .mul(32767)
            .to(torch.int16)
            .cpu()
        )

        return output

def infer_music(lrc, ref_audio_path, text_prompt, current_prompt_type, seed=42, randomize_seed=False, steps=32, cfg_strength=4.0, file_type='wav', odeint_method='euler', Music_Duration='95s'):
    """Main function to generate music from lyrics and prompts."""
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    torch.manual_seed(seed)
    
    # Set up model parameters based on duration
    if Music_Duration == '95s':
        max_frames = 2048
        repo_id = "ASLP-lab/DiffRhythm-base"
    else:  # '285s'
        max_frames = 6144
        repo_id = "ASLP-lab/DiffRhythm-full"
    
    # Prepare models
    cfm, tokenizer, muq, vae = prepare_model(max_frames, device, repo_id=repo_id)

    try:
        # Process lyrics
        lrc_prompt, start_time = get_lrc_token(max_frames, lrc, tokenizer, device)
        
        # Get style prompt based on prompt type
        if current_prompt_type == 'audio':
            style_prompt = get_style_prompt(muq, ref_audio_path)
        else:
            style_prompt = get_style_prompt(muq, prompt=text_prompt)
    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")
    
    # Get negative style prompt and reference latent
    negative_style_prompt = get_negative_style_prompt(device)
    latent_prompt = get_reference_latent(device, max_frames)
    
    # Run inference
    s_t = time.time()
    generated_song = inference(
        cfm_model=cfm,
        vae_model=vae,
        cond=latent_prompt,
        text=lrc_prompt,
        duration=max_frames,
        style_prompt=style_prompt,
        negative_style_prompt=negative_style_prompt,
        start_time=start_time,
        chunked=True,
    )
    e_t = time.time() - s_t
    print(f"Inference completed in {e_t:.2f} seconds")
    
    # Save the generated song to a file
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_filename = f"diffrhythm_{timestamp}.{file_type}"
    output_path = os.path.join(output_dir, output_filename)
    
    torchaudio.save(output_path, generated_song, sample_rate=44100)
    
    return output_path

# CSS styling for the UI
css = """
/* 固定文本域高度并强制滚动条 */
.lyrics-scroll-box textarea {
    height: 405px !important;  /* 固定高度 */
    max-height: 500px !important;  /* 最大高度 */
    overflow-y: auto !important;  /* 垂直滚动 */
    white-space: pre-wrap;  /* 保留换行 */
    line-height: 1.5;  /* 行高优化 */
}

.gr-examples {
    background: transparent !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 8px;
    margin: 1rem 0 !important;
    padding: 1rem !important;
}

"""

# Create the Gradio interface
with gr.Blocks(css=css) as demo:
    gr.HTML(f"""
            <div style="display: flex; align-items: center;">
                <img src='https://raw.githubusercontent.com/ASLP-lab/DiffRhythm/refs/heads/main/src/DiffRhythm_logo.jpg' 
                    style='width: 200px; height: 40%; display: block; margin: 0 auto 20px;'>
            </div>
            
            <div style="flex: 1; text-align: center;">
                <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 5px">
                    Di♪♪Rhythm (谛韵)
                </div>
                <div style="display:flex; justify-content: center; column-gap:4px;">
                    <a href="https://arxiv.org/abs/2503.01183">
                        <img src='https://img.shields.io/badge/Arxiv-Paper-blue'>
                    </a> 
                    <a href="https://github.com/ASLP-lab/DiffRhythm">
                        <img src='https://img.shields.io/badge/GitHub-Repo-green'>
                    </a> 
                    <a href="https://aslp-lab.github.io/DiffRhythm.github.io/">
                        <img src='https://img.shields.io/badge/Project-Page-brown'>
                    </a>
                </div>
            </div> 
            """)
    
    with gr.Tabs() as tabs:
        # Music Generation Tab
        with gr.Tab("Music Generate", id=0):
            with gr.Row():
                with gr.Column():
                    lrc = gr.Textbox(
                        label="Lyrics",
                        placeholder="Input the full lyrics in LRC format",
                        lines=12,
                        max_lines=50,
                        elem_classes="lyrics-scroll-box",
                        value="""[00:04.34]Tell me that I'm special\n[00:06.57]Tell me I look pretty\n[00:08.46]Tell me I'm a little angel\n[00:10.58]Sweetheart of your city\n[00:13.64]Say what I'm dying to hear\n[00:17.35]Cause I'm dying to hear you\n[00:20.86]Tell me I'm that new thing\n[00:22.93]Tell me that I'm relevant\n[00:24.96]Tell me that I got a big heart\n[00:27.04]Then back it up with evidence\n[00:29.94]I need it and I don't know why\n[00:34.28]This late at night\n[00:36.32]Isn't it lonely\n[00:39.24]I'd do anything to make you want me\n[00:43.40]I'd give it all up if you told me\n[00:47.42]That I'd be\n[00:49.43]The number one girl in your eyes\n[00:52.85]Your one and only\n[00:55.74]So what's it gon' take for you to want me\n[00:59.78]I'd give it all up if you told me\n[01:03.89]That I'd be\n[01:05.94]The number one girl in your eyes\n[01:11.34]Tell me I'm going real big places\n[01:14.32]Down to earth so friendly\n[01:16.30]And even through all the phases\n[01:18.46]Tell me you accept me\n[01:21.56]Well that's all I'm dying to hear\n[01:25.30]Yeah I'm dying to hear you\n[01:28.91]Tell me that you need me\n[01:30.85]Tell me that I'm loved\n[01:32.90]Tell me that I'm worth it\n[01:34.95]And that I'm enough\n[01:37.91]I need it and I don't know why\n[01:42.08]This late at night\n[01:44.24]Isn't it lonely\n[01:47.18]I'd do anything to make you want me\n[01:51.30]I'd give it all up if you told me\n[01:55.32]That I'd be\n[01:57.35]The number one girl in your eyes\n[02:00.72]Your one and only\n[02:03.57]So what's it gon' take for you to want me\n[02:07.78]I'd give it all up if you told me\n[02:11.74]That I'd be\n[02:13.86]The number one girl in your eyes\n[02:17.03]The girl in your eyes\n[02:21.05]The girl in your eyes\n[02:26.30]Tell me I'm the number one girl\n[02:28.44]I'm the number one girl in your eyes\n[02:33.49]The girl in your eyes\n[02:37.58]The girl in your eyes\n[02:42.74]Tell me I'm the number one girl\n[02:44.88]I'm the number one girl in your eyes\n[02:49.91]Well isn't it lonely\n[02:53.19]I'd do anything to make you want me\n[02:57.10]I'd give it all up if you told me\n[03:01.15]That I'd be\n[03:03.31]The number one girl in your eyes\n[03:06.57]Your one and only\n[03:09.42]So what's it gon' take for you to want me\n[03:13.50]I'd give it all up if you told me\n[03:17.56]That I'd be\n[03:19.66]The number one girl in your eyes\n[03:25.74]The number one girl in your eyes"""    
                    )
                    
                    current_prompt_type = gr.State(value="audio")
                    with gr.Tabs() as inside_tabs:
                        with gr.Tab("Audio Prompt"):
                            audio_prompt = gr.Audio(label="Audio Prompt", type="filepath", value="./src/prompt/default.wav")
                        with gr.Tab("Text Prompt"):
                            text_prompt = gr.Textbox(
                            label="Text Prompt",
                            placeholder="Enter the Text Prompt, eg: emotional piano pop",
                            )
                    with gr.Accordion("Advanced Settings", open=False):
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=0,
                        )
                        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                        
                        steps = gr.Slider(
                            minimum=10,
                            maximum=100,
                            value=32,
                            step=1,
                            label="Diffusion Steps",
                            interactive=True,
                            elem_id="step_slider"
                        )
                        cfg_strength = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=4.0,
                            step=0.5,
                            label="CFG Strength",
                            interactive=True,
                            elem_id="cfg_slider"
                        )
                        odeint_method = gr.Radio(["euler", "midpoint", "rk4", "implicit_adams"], label="ODE Solver", value="euler")                        
                        file_type = gr.Dropdown(["wav", "mp3", "ogg"], label="Output Format", value="wav")

                        def update_prompt_type(evt: gr.SelectData):
                            return "audio" if evt.index == 0 else "text"

                        inside_tabs.select(
                            fn=update_prompt_type,
                            outputs=current_prompt_type
                        )
                    
                with gr.Column():
                    with gr.Accordion("Best Practices Guide", open=True):
                        gr.Markdown("""
                        1. **Lyrics Format Requirements**
                            - Each line must follow: `[mm:ss.xx]Lyric content`
                            - Example of valid format:
                            ``` 
                            [00:10.00]Moonlight spills through broken blinds
                            [00:13.20]Your shadow dances on the dashboard shrine
                            ```

                        2. **Audio Prompt Requirements**
                            - Reference audio should be ≥ 1 second, audio >10 seconds will be randomly clipped into 10 seconds
                            - For optimal results, the 10-second clips should be carefully selected
                            - Shorter clips may lead to incoherent generation
                        3. **Supported Languages**
                            - **Chinese and English**
                            - More languages comming soon

                        4. **Others** 
                            - If loading audio result is slow, you can select Output Format as mp3 in Advanced Settings.
                        """)
                    
                    Music_Duration = gr.Radio(["95s", "285s"], label="Music Duration", value="95s")
                    
                    lyrics_btn = gr.Button("Generate", variant="primary")
                    audio_output = gr.Audio(label="Audio Result", type="filepath", elem_id="audio_output")
                    

            gr.Examples(
                examples=[
                    ["./src/prompt/pop_cn.wav"], 
                    ["./src/prompt/default.wav"],
                ],
                inputs=[audio_prompt],  
                label="Audio Examples",
                examples_per_page=13,
                elem_id="audio-examples-container" 
            )
            
            gr.Examples(
                examples=[
                    ["Pop Emotional Piano"],
                    ["Electronic Dance Music"],
                    ["Acoustic Folk Guitar"],
                    ["Orchestral Cinematic"],
                ],
                inputs=[text_prompt],  
                label="Text Examples",
                examples_per_page=4,
                elem_id="text-examples-container" 
            )

            gr.Examples(
                examples=[
                    ["""[00:04.34]I'm standing on the edge of tomorrow
[00:08.55]Looking out at a world I don't know
[00:12.67]The path ahead is filled with shadows
[00:16.83]But I know I can't let go"""],
                    ["""[00:02.00]The morning sun breaks through the clouds
[00:06.50]As I walk along the shore
[00:10.75]The waves crash gently at my feet
[00:15.00]I've never felt so sure"""],
                ],
                inputs=[lrc],
                label="Lrc Examples",
                examples_per_page=3,
                elem_id="lrc-examples-container",
            )
    
    # Connect the generate button to the inference function
    lyrics_btn.click(
        fn=infer_music,
        inputs=[lrc, audio_prompt, text_prompt, current_prompt_type, seed, randomize_seed, steps, cfg_strength, file_type, odeint_method, Music_Duration],
        outputs=audio_output
    )

if __name__ == "__main__":
    demo.launch()