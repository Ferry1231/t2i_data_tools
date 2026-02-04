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
            "Kolors": KolorsPipeline
        }
        return name_to_pipeline_map



def img_gen_pipeline(opt: GenSettings, device="cuda"):
    pipeline = opt.get_pipeline()
    prompt = "A 20-year-old East Asian girl with delicate, charming features and large, bright brown eyes—expressive and lively, with a cheerful or subtly smiling expression. Her naturally wavy long hair is either loose or tied in twin ponytails. She has fair skin and light makeup accentuating her youthful freshness. She wears a modern, cute dress or relaxed outfit in bright, soft colors—lightweight fabric, minimalist cut. She stands indoors at an anime convention, surrounded by banners, posters, or stalls. Lighting is typical indoor illumination—no staged lighting—and the image resembles a casual iPhone snapshot: unpretentious composition, yet brimming with vivid, fresh, youthful charm."
    pipe = pipeline.from_pretrained(
        opt.model_path,
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
    print(gen_settings.keys())
    model_name = "SD3.5-medium"
    print(gen_settings[model_name])
    opt = GenSettings(model_name, kwargs=gen_settings[model_name])
    img_gen_pipeline(opt)