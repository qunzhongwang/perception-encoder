from huggingface_hub.hf_api import HfFolder
# get your token here https://huggingface.co/settings/tokens
HfFolder.save_token('YOUR_HF_TOKEN') 

import os
import torch
from PIL import Image
import time
from base64 import b64encode
import textwrap
import requests
import urllib.request

from core.args import dataclass_from_dict
from core.transforms.image_transform import get_image_transform
from core.transforms.video_transform import get_video_transform
from apps.plm.generate import PackedCausalTransformerGeneratorArgs, PackedCausalTransformerGenerator, load_consolidated_model_and_tokenizer

ckpt = "facebook/Perception-LM-8B" 
model, tokenizer, config = load_consolidated_model_and_tokenizer(ckpt)

def generate(
    media_path="",
    question="Describe the image in details.",
    media_type="image",
    number_of_frames=4,
    number_of_tiles=1,
    temperature=0.0,
    top_p=None,
    top_k=None,
):
    prompts = []
    if media_type == "image":
        transform = get_image_transform(
            vision_input_type=(
                "vanilla" if number_of_tiles == 1 else config.data.vision_input_type
            ),
            image_res=model.vision_model.image_size,
            max_num_tiles=number_of_tiles,
        )
        image = Image.open(media_path).convert("RGB")
        image, _ = transform(image)
        prompts.append((question, image))
    elif media_type == "video":
        transform = get_video_transform(
            image_res=model.vision_model.image_size,
        )
        video_info = (media_path, number_of_frames, None, None, None)
        frames, _ = transform(video_info)
        prompts.append((question, frames))
    else:
        raise NotImplementedError(
            f"The provided generate function only supports image and video."
        )
    # Create generator
    gen_cfg = dataclass_from_dict(
        PackedCausalTransformerGeneratorArgs,
        {"temperature": temperature, "top_p": top_p, "top_k": top_k},
        strict=False,
    )
    generator = PackedCausalTransformerGenerator(gen_cfg, model, tokenizer)
    # Run generation
    start_time = time.time()
    generation, loglikelihood, greedy = generator.generate(prompts)
    end_time = time.time()
    for i, gen in enumerate(generation):
        # Calculate tokens per second
        total_tokens = sum(
            len(tokenizer.encode(gen, False, False)) for gen in generation
        )
        tokens_per_second = total_tokens / (end_time - start_time)
        print("=================================================")
        print(textwrap.fill(gen, width=75))
        print(f"Tokens per second: {tokens_per_second:.2f}")
        print("=================================================")

# imgURL = "http://images.cocodataset.org/val2017/000000281759.jpg"

# urllib.request.urlretrieve(imgURL, "000000281759.jpg")

# media_path = "000000281759.jpg"
# question = "Describe the image in details."

# print("Generating...")
# # with basic colab we can only run with with 1 to 4 tiles, instead of full 36 tiles.
# # generate(media_path=media_path, question=question, media_type="image")
# print("Generating with 4 tiles + 1 tumb...")
# generate(media_path=media_path, question=question, number_of_tiles=4, media_type="image")

media_path = "shard_0/97490574.mp4"
question = "What is happening in the video?"

mp4 = open(media_path,'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
# HTML("""
# <video width=400 controls>
#       <source src="%s" type="video/mp4">
# </video>
# """ % data_url)
print("Generating with 4 frames ...")
# with basic colab we can only run with 4 frames
generate(media_path=media_path, question=question, media_type="video")