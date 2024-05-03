import gradio as gr
import numpy as np
import torch, torchvision
from PIL import Image
from tqdm.auto import tqdm
from diffusers import DDPMPipeline
from diffusers import DDIMScheduler

device = 'cuda'
# Load the pretrained pipeline
pipeline_name = "pokemon_model"
image_pipe = DDPMPipeline.from_pretrained(pipeline_name).to(device)

# Sample some images with a DDIM Scheduler over 40 steps
scheduler = DDIMScheduler.from_pretrained(pipeline_name)
scheduler.set_timesteps(num_inference_steps=40)


# The function that does the hard work
def generate():
    x = torch.randn(1, 3, 128, 128).to(device)
    for i, t in tqdm(enumerate(scheduler.timesteps)):
        model_input = scheduler.scale_model_input(x, t)
        with torch.no_grad():
            noise_pred = image_pipe.unet(model_input, t)["sample"]
        x = scheduler.step(noise_pred, t, x).prev_sample
    grid = torchvision.utils.make_grid(x, nrow=4)
    im = grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
    im = Image.fromarray(np.array(im * 255).astype(np.uint8))
    im.save("test.jpeg")
    return im


outputs = gr.Image(label="result")

# And the interface
demo = gr.Interface(
    fn=generate,
    inputs=None,
    outputs=outputs
)
demo.launch(debug=True)
