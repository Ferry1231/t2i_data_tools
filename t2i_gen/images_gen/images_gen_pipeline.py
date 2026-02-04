import torch
import json
import os
import argparse
from typing import Literal
from torch.utils.data import Dataset, DataLoader
from diffusers import ZImagePipeline, DiffusionPipeline, FluxPipeline, Flux2Pipeline, StableDiffusion3Pipeline, KolorsPipeline, SanaPipeline

from tqdm import tqdm


class GenSettings:
    def __init__(
            self,
            model_name: str,
            dtype: torch.dtype = torch.bfloat16,
            height: int = 1024,
            width: int = 1024,
            kwargs: dict = None,
        ):
        self.model_name = model_name
        self.height = height
        self.width = width
        self.dtype = dtype
        self.kwargs = kwargs
        self.name_to_pipeline_map = self.name_to_pipeline()
        self.model_path = kwargs.get("model_path", None)

    def get_dict(self):
        return {
            "height": self.height,
            "width": self.width,
        } | {k: v for k, v in self.kwargs.items() if k != "model_path"}
    
    def get_pipeline(self):
        return self.name_to_pipeline_map[self.model_name]
    
    def name_to_pipeline(self):
        name_to_pipeline_map = {
            "Z-Image-Turbo": ZImagePipeline,
            "Z-Image": ZImagePipeline,
            "Qwen-Image-2512": DiffusionPipeline,
            "FLUX.1-dev": FluxPipeline,
            "FLUX.1-Krea-dev": FluxPipeline,
            "FLUX.1-schnell": FluxPipeline,
            "FLUX.2-dev": Flux2Pipeline,
            # "FLUX.2-klein-9B-base": Flux2KleinPipeline,
            # "FLUX.2-klein-9B": Flux2KleinPipeline,
            # "FLUX.2-klein-4B-base": Flux2KleinPipeline,
            # "FLUX.2-klein-4B": Flux2KleinPipeline,
            "SD3.5-large": StableDiffusion3Pipeline,
            "SD3.5-medium": StableDiffusion3Pipeline,
            "Kolors": KolorsPipeline,
            "SANA1.5_4.8B": SanaPipeline
        }
        return name_to_pipeline_map


class PromptDataset(Dataset):
    def __init__(
        self, 
        prompts_file, 
        prompt_choice: Literal['prompt', 'prompt_fault'],
        key_choice: str = 'key'
    ):
        self.data = prompts_file
        self.prompt_choice = prompt_choice
        self.key_choice = key_choice

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        key = item[self.key_choice]
        prompt = item[self.prompt_choice]
        
        return prompt, key


def create_dataloader(
    json_file: list,
    prompt_choice: Literal['prompt', 'prompt_fault'] = 'prompt',
    key_choice: str = 'key',
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 16,
):  
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    dataset = PromptDataset(
        prompts_file=json_data,
        prompt_choice=prompt_choice,
        key_choice=key_choice,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )

    return dataloader


def img_gen_pipeline(opt: GenSettings, device="cuda"):
    pipeline = opt.get_pipeline()
    prompt = [
        "Painting: A solitary woman stands on the shoreline, barefoot and still, as she gazes out at the vast open water of the lake. Tall mountains stand solemn and unwavering against the cerulean sky in the backdrop, their jagged edges caught in a soft golden light. Mist curling gently from the water's surface adds a hazy, tranquil atmosphere to the scene, while the gentle rustle of wind passing through surrounding trees enhances the peaceful ambience. The overall mood is one of calm reflection and quiet wonder, capturing an image that is both serene and evocative of the vastness of nature. ",
        # "a flower"
    ]
    pipe = pipeline.from_pretrained(
        opt.model_path,
        torch_dtype=opt.dtype
    ).to(device)
    image = pipe(
        prompt,
        **opt.get_dict()
    ).images[0]
    image.save(f"example_2_{opt.model_name}.png")


def generation_pipeline(
    opt: GenSettings, 
    dataloader: DataLoader,
    positive: Literal['positive', 'negative'],
    save_root: str,
    device="cuda"
):  
    model_name = opt.model_name
    save_path = os.path.join(save_root, model_name)
    os.makedirs(save_path, exist_ok=True)

    pipeline = opt.get_pipeline()
    pipe = pipeline.from_pretrained(opt.model_path, torch_dtype=opt.dtype).to(device)

    for batch in dataloader:
        prompts, keys = batch
        images = pipe(list(prompts), **opt.get_dict()).images
        for i, image in enumerate(images):
            image.save(os.path.join(save_path, f"{keys[i]}_{positive}.png"))




if __name__ ==  "__main__":
    parser = argparse.ArgumentParser(description="Run prompt dataloader and image generation pipeline.")
    parser.add_argument("--st_file", type=str, default="/root/fengyuan/tools/t2i_data_tools/t2i_gen/images_gen/gen_settings/settings.json")
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--prompt_choice", type=str, required=True, choices=["prompt", "prompt_fault"], help="Which prompt field to use: 'prompt' or 'prompt_fault'.")
    parser.add_argument("--postive", type=str, required=True, choices=["positive", "negative"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--save_root", type=str, default="/root/fengyuan/datasets/vision_auto_rubric/images")
    args = parser.parse_args()
    
    with open(args.st_file, "r") as f:
        gen_settings = json.load(f)

    opt = GenSettings(args.model_name, kwargs=gen_settings[args.model_name])

    dataloader = create_dataloader(
        json_file=args.data_file,
        prompt_choice=args.prompt_choice, 
        batch_size=args.batch_size
    )

    generation_pipeline(
        opt=opt,
        dataloader=dataloader,
        positive=args.postive,
        save_root=args.save_root,
    )



    # st_file = "/root/fengyuan/tools/t2i_data_tools/t2i_gen/images_gen/gen_settings/settings.json"
    # with open(st_file, "r") as f:
    #     gen_settings = json.load(f)
    # # print(gen_settings.keys())
    # model_name = "SANA1.5_4.8B"
    # # print(gen_settings[model_name])
    # opt = GenSettings(model_name, kwargs=gen_settings[model_name])
    # img_gen_pipeline(opt)

