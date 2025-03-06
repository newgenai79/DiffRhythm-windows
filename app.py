import gradio as gr
from openai import OpenAI
import requests
import json
# from volcenginesdkarkruntime import Ark
import torch
import torchaudio
from einops import rearrange
import argparse
import json
import os

from tqdm import tqdm
import random
import numpy as np
import sys
import base64
from diffrhythm.infer.infer_utils import (
    get_reference_latent,
    get_lrc_token,
    get_style_prompt,
    prepare_model,
    get_negative_style_prompt
)
from diffrhythm.infer.infer import inference

MAX_SEED = np.iinfo(np.int32).max
device='cuda'
cfm, tokenizer, muq, vae = prepare_model(device)
cfm = torch.compile(cfm)


def infer_music(lrc, ref_audio_path, seed=42, randomize_seed=False, steps=32, file_type='wav', max_frames=2048, device='cuda'):

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    torch.manual_seed(seed)
    sway_sampling_coef = -1 if steps < 32 else None
    lrc_prompt, start_time = get_lrc_token(lrc, tokenizer, device)
    style_prompt = get_style_prompt(muq, ref_audio_path)
    negative_style_prompt = get_negative_style_prompt(device)
    latent_prompt = get_reference_latent(device, max_frames)
    generated_song = inference(cfm_model=cfm, 
                               vae_model=vae, 
                               cond=latent_prompt, 
                               text=lrc_prompt, 
                               duration=max_frames, 
                               style_prompt=style_prompt,
                               negative_style_prompt=negative_style_prompt,
                               steps=steps,
                               sway_sampling_coef=sway_sampling_coef,
                               start_time=start_time,
                               file_type=file_type
                               )
    return generated_song

def R1_infer1(theme, tags_gen, language):
    try:
        client = OpenAI(api_key=os.getenv('HS_DP_API'), base_url = "https://ark.cn-beijing.volces.com/api/v3")

        llm_prompt = """
        请围绕"{theme}"主题生成一首符合"{tags}"风格的语言为{language}的完整歌词。严格遵循以下要求：

        ### **强制格式规则**
        1. **仅输出时间戳和歌词**，禁止任何括号、旁白、段落标记（如副歌、间奏、尾奏等注释）。
        2. 每行格式必须为 `[mm:ss.xx]歌词内容`，时间戳与歌词间无空格，歌词内容需完整连贯。
        3. 时间戳需自然分布，**第一句歌词起始时间不得为 [00:00.00]**，需考虑前奏空白。

        ### **内容与结构要求**
        1. 歌词应富有变化，使情绪递进，整体连贯有层次感。**每行歌词长度应自然变化**，切勿长度一致，导致很格式化。
        2. **时间戳分配应根据歌曲的标签、歌词的情感、节奏来合理推测**，而非机械地按照歌词长度分配。
        3. 间奏/尾奏仅通过时间空白体现（如从 [02:30.00] 直接跳至 [02:50.00]），**无需文字描述**。

        ### **负面示例（禁止出现）**
        - 错误：[01:30.00](钢琴间奏)
        - 错误：[02:00.00][副歌]
        - 错误：空行、换行符、注释
        """

        response = client.chat.completions.create(
            model="ep-20250304144033-nr9wl",
            messages=[
                {"role": "system", "content": "You are a professional musician who has been invited to make music-related comments."},
                {"role": "user", "content": llm_prompt.format(theme=theme, tags=tags_gen, language=language)},
            ],
            stream=False
        )
        
        info = response.choices[0].message.content

        return info

    except requests.exceptions.RequestException as e:
        print(f'请求出错: {e}')
        return {}



def R1_infer2(tags_lyrics, lyrics_input):
    client = OpenAI(api_key=os.getenv('HS_DP_API'), base_url = "https://ark.cn-beijing.volces.com/api/v3")

    llm_prompt = """
    {lyrics_input}这是一首歌的歌词,每一行是一句歌词,{tags_lyrics}是我希望这首歌的风格，我现在想要给这首歌的每一句歌词打时间戳得到LRC，我希望时间戳分配应根据歌曲的标签、歌词的情感、节奏来合理推测，而非机械地按照歌词长度分配。第一句歌词的时间戳应考虑前奏长度，避免歌词从 `[00:00.00]` 直接开始。严格按照 LRC 格式输出歌词，每行格式为 `[mm:ss.xx]歌词内容`。最后的结果只输出LRC,不需要其他的解释。
    """

    response = client.chat.completions.create(
        model="ep-20250304144033-nr9wl",
        messages=[
            {"role": "system", "content": "You are a professional musician who has been invited to make music-related comments."},
            {"role": "user", "content": llm_prompt.format(lyrics_input=lyrics_input, tags_lyrics=tags_lyrics)},
        ],
        stream=False
    )

    info = response.choices[0].message.content

    return info

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


with gr.Blocks(css=css) as demo:
    gr.HTML(f"""
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
        
        # page 1
        with gr.Tab("Music Generate", id=0):
            with gr.Row():
                with gr.Column():
                    lrc = gr.Textbox(
                        label="Lrc",
                        placeholder="Input the full lyrics",
                        lines=12,
                        max_lines=50,
                        elem_classes="lyrics-scroll-box",
                        value="""[00:10.00]Moonlight spills through broken blinds\n[00:13.20]Your shadow dances on the dashboard shrine\n[00:16.85]Neon ghosts in gasoline rain\n[00:20.40]I hear your laughter down the midnight train\n[00:24.15]Static whispers through frayed wires\n[00:27.65]Guitar strings hum our cathedral choirs\n[00:31.30]Flicker screens show reruns of June\n[00:34.90]I'm drowning in this mercury lagoon\n[00:38.55]Electric veins pulse through concrete skies\n[00:42.10]Your name echoes in the hollow where my heartbeat lies\n[00:45.75]We're satellites trapped in parallel light\n[00:49.25]Burning through the atmosphere of endless night\n[01:00.00]Dusty vinyl spins reverse\n[01:03.45]Our polaroid timeline bleeds through the verse\n[01:07.10]Telescope aimed at dead stars\n[01:10.65]Still tracing constellations through prison bars\n[01:14.30]Electric veins pulse through concrete skies\n[01:17.85]Your name echoes in the hollow where my heartbeat lies\n[01:21.50]We're satellites trapped in parallel light\n[01:25.05]Burning through the atmosphere of endless night\n[02:10.00]Clockwork gears grind moonbeams to rust\n[02:13.50]Our fingerprint smudged by interstellar dust\n[02:17.15]Velvet thunder rolls through my veins\n[02:20.70]Chasing phantom trains through solar plane\n[02:24.35]Electric veins pulse through concrete skies\n[02:27.90]Your name echoes in the hollow where my heartbeat lies"""    
                    )
                    audio_prompt = gr.Audio(label="Audio Prompt", type="filepath", value="./src/prompt/default.wav")
                    
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

                        2. **Generation Duration Limits**
                        - Current version supports maximum **95 seconds** of music generation
                        - Total timestamps should not exceed 01:35.00 (95 seconds)

                        3. **Audio Prompt Requirements**
                        - Reference audio should be ≥ 1 second, audio >10 seconds will be randomly clipped into 10 seconds
                        - For optimal results, the 10-second clips should be carefully selected
                        - Shorter clips may lead to incoherent generation
                        
                        4. **Supported Languages**
                        - Chinese and English
                        - More languages comming soon
                        """)
                    
                    lyrics_btn = gr.Button("Generate", variant="primary")
                    audio_output = gr.Audio(label="Audio Result", type="filepath", elem_id="audio_output")
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
                        file_type = gr.Dropdown(["wav", "mp3", "ogg"], label="Output Format", value="wav")


            gr.Examples(
                examples=[
                    ["./src/prompt/pop_cn.wav"], 
                    ["./src/prompt/pop_en.wav"], 
                    ["./src/prompt/rock_cn.wav"], 
                    ["./src/prompt/rock_en.wav"], 
                    ["./src/prompt/country_cn.wav"], 
                    ["./src/prompt/country_en.wav"],
                    ["./src/prompt/classic_cn.wav"],
                    ["./src/prompt/classic_en.wav"],
                    ["./src/prompt/jazz_cn.wav"],
                    ["./src/prompt/jazz_en.wav"],
                    ["./src/prompt/default.wav"]
                ],
                inputs=[audio_prompt],  
                label="Audio Examples",
                examples_per_page=11,
                elem_id="audio-examples-container" 
            )

            gr.Examples(
                examples=[
                    ["""[00:10.00]Moonlight spills through broken blinds\n[00:13.20]Your shadow dances on the dashboard shrine\n[00:16.85]Neon ghosts in gasoline rain\n[00:20.40]I hear your laughter down the midnight train\n[00:24.15]Static whispers through frayed wires\n[00:27.65]Guitar strings hum our cathedral choirs\n[00:31.30]Flicker screens show reruns of June\n[00:34.90]I'm drowning in this mercury lagoon\n[00:38.55]Electric veins pulse through concrete skies\n[00:42.10]Your name echoes in the hollow where my heartbeat lies\n[00:45.75]We're satellites trapped in parallel light\n[00:49.25]Burning through the atmosphere of endless night\n[01:00.00]Dusty vinyl spins reverse\n[01:03.45]Our polaroid timeline bleeds through the verse\n[01:07.10]Telescope aimed at dead stars\n[01:10.65]Still tracing constellations through prison bars\n[01:14.30]Electric veins pulse through concrete skies\n[01:17.85]Your name echoes in the hollow where my heartbeat lies\n[01:21.50]We're satellites trapped in parallel light\n[01:25.05]Burning through the atmosphere of endless night\n[02:10.00]Clockwork gears grind moonbeams to rust\n[02:13.50]Our fingerprint smudged by interstellar dust\n[02:17.15]Velvet thunder rolls through my veins\n[02:20.70]Chasing phantom trains through solar plane\n[02:24.35]Electric veins pulse through concrete skies\n[02:27.90]Your name echoes in the hollow where my heartbeat lies"""],
                    ["""[00:04.34]Tell me that I'm special\n[00:06.57]Tell me I look pretty\n[00:08.46]Tell me I'm a little angel\n[00:10.58]Sweetheart of your city\n[00:13.64]Say what I'm dying to hear\n[00:17.35]Cause I'm dying to hear you\n[00:20.86]Tell me I'm that new thing\n[00:22.93]Tell me that I'm relevant\n[00:24.96]Tell me that I got a big heart\n[00:27.04]Then back it up with evidence\n[00:29.94]I need it and I don't know why\n[00:34.28]This late at night\n[00:36.32]Isn't it lonely\n[00:39.24]I'd do anything to make you want me\n[00:43.40]I'd give it all up if you told me\n[00:47.42]That I'd be\n[00:49.43]The number one girl in your eyes\n[00:52.85]Your one and only\n[00:55.74]So what's it gon' take for you to want me\n[00:59.78]I'd give it all up if you told me\n[01:03.89]That I'd be\n[01:05.94]The number one girl in your eyes\n[01:11.34]Tell me I'm going real big places\n[01:14.32]Down to earth so friendly\n[01:16.30]And even through all the phases\n[01:18.46]Tell me you accept me\n[01:21.56]Well that's all I'm dying to hear\n[01:25.30]Yeah I'm dying to hear you\n[01:28.91]Tell me that you need me\n[01:30.85]Tell me that I'm loved\n[01:32.90]Tell me that I'm worth it"""]
                ],
                
                inputs=[lrc],
                label="Lrc Examples",
                examples_per_page=2,
                elem_id="lrc-examples-container",
            )

        # page 2
        with gr.Tab("LLM Generate LRC", id=1):
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Notice", open=False):
                        gr.Markdown("**Two Generation Modes:**\n1. Generate from theme & tags\n2. Add timestamps to existing lyrics")
                    
                    with gr.Group():
                        gr.Markdown("### Method 1: Generate from Theme")
                        theme = gr.Textbox(label="theme", placeholder="Enter song theme, e.g: Love and Heartbreak")
                        tags_gen = gr.Textbox(label="tags", placeholder="Enter song tags, e.g: pop confidence healing")
                        language = gr.Radio(["zh", "en"], label="Language", value="en")
                        gen_from_theme_btn = gr.Button("Generate LRC (From Theme)", variant="primary")
                        
                        gr.Examples(
                            examples=[
                                [
                                    "Love and Heartbreak", 
                                    "vocal emotional piano pop",
                                    "en"
                                ],
                                [
                                    "Heroic Epic", 
                                    "choir orchestral powerful",
                                    "zh"
                                ]
                            ],
                            inputs=[theme, tags_gen, language],
                            label="Examples: Generate from Theme"
                        )

                    with gr.Group(visible=True): 
                        gr.Markdown("### Method 2: Add Timestamps to Lyrics")
                        tags_lyrics = gr.Textbox(label="tags", placeholder="Enter song tags, e.g: ballad piano slow")
                        lyrics_input = gr.Textbox(
                            label="Raw Lyrics (without timestamps)",
                            placeholder="Enter plain lyrics (without timestamps), e.g:\nYesterday\nAll my troubles...",
                            lines=10,
                            max_lines=50,
                            elem_classes="lyrics-scroll-box"
                        )
                        
                        gen_from_lyrics_btn = gr.Button("Generate LRC (From Lyrics)", variant="primary")

                        gr.Examples(
                            examples=[
                                [
                                    "acoustic folk happy", 
                                    """I'm sitting here in the boring room\nIt's just another rainy Sunday afternoon"""
                                ],
                                [
                                    "electronic dance energetic",
                                    """We're living in a material world\nAnd I am a material girl"""
                                ]
                            ],
                            inputs=[tags_lyrics, lyrics_input],
                            label="Examples: Generate from Lyrics"
                        )


                with gr.Column():
                    lrc_output = gr.Textbox(
                        label="Generated LRC Lyrics",
                        placeholder="Timed lyrics will appear here",
                        lines=57,
                        elem_classes="lrc-output",
                        show_copy_button=True
                    )

            # Bind functions
            gen_from_theme_btn.click(
                fn=R1_infer1,
                inputs=[theme, tags_gen, language],
                outputs=lrc_output
            )
            
            gen_from_lyrics_btn.click(
                fn=R1_infer2,
                inputs=[tags_lyrics, lyrics_input],
                outputs=lrc_output
            )

    tabs.select(
    lambda s: None, 
    None, 
    None 
    )
    
    lyrics_btn.click(
        fn=infer_music,
        inputs=[lrc, audio_prompt, seed, randomize_seed, steps, file_type],
        outputs=audio_output
    )


demo.queue().launch(show_api=False, show_error=True)



if __name__ == "__main__":
    demo.launch()