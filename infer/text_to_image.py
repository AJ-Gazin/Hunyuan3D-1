import os
import sys
import numpy as np
from PIL import Image
import torch
from diffusers import DiffusionPipeline
import warnings
import tempfile
import requests
from huggingface_hub import hf_hub_download
from pathlib import Path

sys.path.insert(0, f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

from infer.utils import seed_everything, timing_decorator, auto_amp_inference
from infer.utils import get_parameter_number, set_parameter_grad_false

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Text2Image():
    def __init__(self, pretrain="weights/flux1dev", device="cuda", save_memory=None):
        '''
        Initialize the Text2Image model using FLUX.
        '''
        self.save_memory = save_memory
        self.device = device
        self.active_lora = None
        self.lora_scale = 1.0
        self.cache_dir = './lora_cache'
        os.makedirs(self.cache_dir, exist_ok=True)
        
        try:
            # Load FLUX model
            self.pipe = DiffusionPipeline.from_pretrained(
                pretrain,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            ).to(device)
            
            # Configure pipeline settings
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()
            
            print(f"FLUX model loaded from {pretrain} on device: {device}")
            
        except Exception as e:
            print(f"Error loading model from {pretrain}: {str(e)}")
            raise

    def validate_image(self, image, stage="unknown"):
        """
        Validate image and print diagnostic information
        """
        if image is None:
            raise ValueError(f"[{stage}] Image is None")
            
        if isinstance(image, Image.Image):
            if image.size == (0, 0):
                raise ValueError(f"[{stage}] Image has zero dimensions")
            if image.mode not in ['RGB', 'RGBA']:
                print(f"[{stage}] Warning: Unexpected image mode: {image.mode}")
            print(f"[{stage}] Image info - Size: {image.size}, Mode: {image.mode}")
            return True
            
        if isinstance(image, np.ndarray):
            if image.size == 0:
                raise ValueError(f"[{stage}] Numpy array is empty")
            if len(image.shape) < 2:
                raise ValueError(f"[{stage}] Invalid image shape: {image.shape}")
            print(f"[{stage}] Array info - Shape: {image.shape}, Type: {image.dtype}")
            print(f"[{stage}] Value range - Min: {image.min()}, Max: {image.max()}")
            return True
            
        raise ValueError(f"[{stage}] Invalid image type: {type(image)}")

    def process_image(self, image):
        """
        Process and validate the generated image.
        """
        try:
            self.validate_image(image, "pre-process")
            
            # Convert to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Handle NaN values
            if np.isnan(image).any():
                print("Warning: NaN values found in image, replacing with zeros")
                image = np.nan_to_num(image, nan=0.0)
            
            # Normalize value range
            if image.max() <= 1.0:
                image = image * 255
            
            # Ensure proper range
            image = np.clip(image, 0, 255)
            
            # Convert to uint8
            image = image.astype(np.uint8)
            
            # If image is all zeros or all ones, raise error
            if np.all(image == 0) or np.all(image == 255):
                raise ValueError("Image appears to be all black or all white")
            
            # Convert back to PIL Image
            pil_image = Image.fromarray(image)
            self.validate_image(pil_image, "post-process")
            
            return pil_image
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            raise

    def download_lora(self, lora_url):
        """
        Download a LoRA model from a URL or HuggingFace repo.
        Returns the local path to the downloaded file.
        """
        try:
            if lora_url.startswith(('http://', 'https://')):
                if 'huggingface.co' in lora_url:
                    # Extract repo_id and filename from HF URL
                    parts = lora_url.split('huggingface.co/')[-1].split('/')
                    repo_id = '/'.join(parts[:2])
                    filename = f"{parts[-1]}.safetensors"
                    
                    # Download from HF
                    local_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=self.cache_dir
                    )
                else:
                    # Direct download for other URLs
                    response = requests.get(lora_url, stream=True)
                    response.raise_for_status()
                    
                    # Create temp file with unique name based on URL
                    url_hash = hash(lora_url)
                    temp_path = os.path.join(self.cache_dir, f'lora_{url_hash}.safetensors')
                    
                    with open(temp_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    local_path = temp_path
                
                print(f"LoRA downloaded successfully to {local_path}")
                return local_path
            else:
                raise ValueError("URL must start with http:// or https://")
                
        except Exception as e:
            print(f"Error downloading LoRA: {str(e)}")
            raise

    def cleanup_lora_cache(self, max_files=10):
        """
        Clean up old LoRA files from cache directory
        """
        try:
            files = list(Path(self.cache_dir).glob('*.safetensors'))
            if len(files) > max_files:
                # Sort by modification time, oldest first
                files.sort(key=lambda x: x.stat().st_mtime)
                # Remove oldest files
                for file in files[:-max_files]:
                    file.unlink()
                print(f"Cleaned up {len(files) - max_files} old LoRA files from cache")
        except Exception as e:
            print(f"Error cleaning up LoRA cache: {str(e)}")

    def load_lora(self, lora_path, scale=1.0):
        """
        Load and apply a LoRA to the pipeline.
        """
        try:
            if not os.path.exists(lora_path):
                raise FileNotFoundError(f"LoRA file not found at {lora_path}")
                
            print(f"Loading LoRA from {lora_path} with scale {scale}")
            self.pipe.load_lora_weights(lora_path)
            self.active_lora = lora_path
            self.lora_scale = scale
            print("LoRA loaded successfully")
            
        except Exception as e:
            print(f"Error loading LoRA: {str(e)}")
            raise

    def unload_lora(self):
        """
        Remove the currently active LoRA.
        """
        if self.active_lora:
            try:
                self.pipe.unload_lora_weights()
                self.active_lora = None
                self.lora_scale = 1.0
                print("LoRA unloaded successfully")
            except Exception as e:
                print(f"Error unloading LoRA: {str(e)}")
                raise

    @torch.no_grad()
    @timing_decorator('text to image')
    @auto_amp_inference
    def __call__(self, *args, **kwargs):
        if self.save_memory:
            self.pipe = self.pipe.to(self.device)
            torch.cuda.empty_cache()
            res = self.call(*args, **kwargs)
            self.pipe = self.pipe.to("cpu")
            if self.active_lora:
                self.unload_lora()
        else:
            res = self.call(*args, **kwargs)
        torch.cuda.empty_cache()
        return res

    def call(self, prompt, seed=0, steps=25, lora_url=None, lora_scale=1.0):
        try:
            print("Generating image for prompt:", prompt)
    
            # Handle LoRA if provided
            if lora_url:
                if self.active_lora:
                    self.unload_lora()
                lora_path = self.download_lora(lora_url)
                self.load_lora(lora_path, lora_scale)
                self.cleanup_lora_cache()
    
            # Set random seed
            seed_everything(seed)
            generator = torch.Generator(device=self.device).manual_seed(seed)
    
            # Prepare kwargs based on support
            kwargs = {
                "prompt": prompt,
                "num_inference_steps": steps,
                "width": 1024,
                "height": 1024,
                "guidance_scale": 7.5,
                "generator": generator,
            }
            if self.active_lora:
                # Add LoRA scale only if the pipeline supports cross_attention_kwargs
                if hasattr(self.pipe, "__call__") and "cross_attention_kwargs" in self.pipe.__call__.__code__.co_varnames:
                    kwargs["cross_attention_kwargs"] = {"scale": self.lora_scale}
    
            # Generate image
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = self.pipe(**kwargs)
    
            # Get image
            if hasattr(output, 'images'):
                image = output.images[0]
            else:
                image = output[0][0]
    
            self.validate_image(image, "pipeline-output")
            processed_image = self.process_image(image)
            print("Image generated successfully.")
            return processed_image
    
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            raise


if __name__ == "__main__":
    import argparse
    
    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--text_prompt", default="", type=str)
        parser.add_argument("--output_img_path", default="./outputs/test/img.jpg", type=str)
        parser.add_argument("--device", default="cuda", type=str)
        parser.add_argument("--seed", default=0, type=int)
        parser.add_argument("--steps", default=25, type=int)
        parser.add_argument("--lora_url", default="", type=str,
                          help="URL to LoRA model (e.g., https://huggingface.co/path/to/lora)")
        parser.add_argument("--lora_scale", default=1.0, type=float,
                          help="Scale factor for LoRA influence (default: 1.0)")
        return parser.parse_args()
    
    try:
        args = get_args()
        os.makedirs(os.path.dirname(args.output_img_path), exist_ok=True)
        
        model = Text2Image(device=args.device)
        
        image = model(
            args.text_prompt, 
            seed=args.seed, 
            steps=args.steps,
            lora_url=args.lora_url if args.lora_url else None,
            lora_scale=args.lora_scale
        )
        
        # Validate before saving
        model.validate_image(image, "final")
        
        # Save with additional checks
        if image.size == (0, 0):
            raise ValueError("Cannot save image with zero dimensions")
            
        image.save(args.output_img_path)
        print(f"Image saved to {args.output_img_path}")
        
        # Verify saved file
        if not os.path.exists(args.output_img_path):
            raise ValueError("Failed to save image")
        if os.path.getsize(args.output_img_path) < 1000:
            raise ValueError(f"Saved image is suspiciously small: {os.path.getsize(args.output_img_path)} bytes")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        sys.exit(1)