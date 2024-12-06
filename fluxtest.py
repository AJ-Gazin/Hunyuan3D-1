import torch
from diffusers import FluxPipeline
from PIL import Image
import numpy as np
import os

def test_flux_image_generation(prompt="a lovely rabbit", steps=25, output_path="./test_output.png"):
    try:
        # Load the FLUX model
        print("Loading FLUX model...")
        model_path = "weights/flux1dev"  # Adjust if your model is stored elsewhere
        pipe = FluxPipeline.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16
        ).to("cuda")

        print(f"Generating image for prompt: '{prompt}'")
        
        # Generate the image
        result = pipe(
            prompt="A large orange bunny",
            num_inference_steps=steps,
            width=1024,
            height=1024,
            guidance_scale=3.5  # Adjust as needed
        )
        
        # Process the output
        image = result.images[0]
        if isinstance(image, Image.Image):
            print("Image generated successfully.")
        else:
            print("Error: The output is not a valid PIL Image.")
            return

        # Save the output
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        print(f"Image saved to {output_path}")

        # Validate the image
        image_array = np.array(image)
        if np.isnan(image_array).any():
            print("Warning: NaN values detected in the image array.")
        elif image_array.size == 0:
            print("Error: Generated image is empty.")
        else:
            print("Image validation passed.")

    except Exception as e:
        print(f"Error during FLUX image generation: {e}")

if __name__ == "__main__":
    test_flux_image_generation()
