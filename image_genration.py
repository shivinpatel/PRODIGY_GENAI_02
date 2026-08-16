import torch
from diffusers import StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"

# Automatically select GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)
print("Loading AI model...")

# Load pre-trained model
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

pipe = pipe.to(device)

# Text prompt
prompt = "A futuristic city with flying cars at sunset, cinematic digital art"

print("Generating image...")

# Generate image
image = pipe(
    prompt,
    num_inference_steps=20
).images[0]

image.save("generated_image.png")

print("Image generated successfully!")
print("Saved as: generated_image.png")
