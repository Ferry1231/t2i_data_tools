import torch
import json
from diffusers import ZImagePipeline, DiffusionPipeline, FluxPipeline, Flux2Pipeline, StableDiffusion3Pipeline, KolorsPipeline


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

    def get_dict(self):
        return {
            "height": self.height,
            "width": self.width,
        } | self.kwargs.pop("model_path", None)
    
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
            "FLUX.2-klein-9B-base": Flux2Pipeline,
            "FLUX.2-klein-9B": Flux2Pipeline,
            "FLUX.2-klein-4B-base": Flux2Pipeline,
            "FLUX.2-klein-4B": Flux2Pipeline,
            "SD3.5-large": StableDiffusion3Pipeline,
            "Kolors": KolorsPipeline
        }
        return name_to_pipeline_map



def img_gen_pipeline(opt: GenSettings, device="cuda"):
    pipeline = opt.get_pipeline()
    prompt = None
    pipe = pipeline.from_pretrained(
        opt.model_name,
        torch_dtype=opt.dtype
    ).to(device)
    image = pipe(
        prompt,
        **opt.get_dict()
    ).images[0]
    image.save("example.png")

if __name__ ==  "__main__":
    st_file = "/root/fengyuan/tools/t2i_data_tools/t2i_gen/images_gen/gen_settings/settings.json"
    with open(st_file, "r") as f:
        gen_settings = json.load(f)
    model_name = "Z-Image-Turbo"
    opt = GenSettings(model_name, gen_settings[model_name])
    img_gen_pipeline(opt)