import torch
from diffusers import DiffusionPipeline
from PIL import Image

# switch to "mps" for apple devices
pipe = DiffusionPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", dtype=torch.bfloat16, device_map="cuda").to("cuda")


prompt = "Change this cat into a sleeping black cat with his head laid"

input_image = Image.open("/workspace/workdir/source/pet_train_6.jpg").convert("RGB")

# 3. Generate with matched dtypes and guidance_scale=0.0
with torch.autocast("cuda", dtype=torch.bfloat16):
    image = pipe(
        prompt=prompt,
        image=input_image,
        guidance_scale=0.0,
        num_inference_steps=4
    ).images[0]

image.save("output_2.png")