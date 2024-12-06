# Open Source Model Licensed under the Apache License Version 2.0 
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited 
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved. 
# The below software and/or models in this distribution may have been 
# modified by THL A29 Limited ("Tencent Modifications"). 
# All Tencent Modifications are Copyright (C) THL A29 Limited.

import os
import warnings
import argparse
import time
from PIL import Image
import torch
from glob import glob

warnings.simplefilter('ignore', category=UserWarning)
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)

from infer import Text2Image, Removebg, Image2Views, Views2Mesh, GifRenderer
from third_party.mesh_baker import MeshBaker
from third_party.check import check_bake_available

try:
    from third_party.mesh_baker import MeshBaker
    assert check_bake_available()
    BAKE_AVAILABLE = True
except Exception as err:
    print(err)
    print("import baking related fail, run without baking")
    BAKE_AVAILABLE = False

def get_args():
    parser = argparse.ArgumentParser(description='3D Generation Pipeline with FLUX')
    parser.add_argument(
        "--use_lite", 
        default=False, 
        action="store_true",
        help="Use lite version of models"
    )
    parser.add_argument(
        "--mv23d_cfg_path", 
        default="./svrm/configs/svrm.yaml", 
        type=str,
        help="Path to mv23d config file"
    )
    parser.add_argument(
        "--mv23d_ckt_path", 
        default="weights/svrm/svrm.safetensors", 
        type=str,
        help="Path to mv23d checkpoint"
    )
    parser.add_argument(
        "--text2image_path", 
        default="weights/flux1dev", 
        type=str,
        help="Path to FLUX model"
    )
    parser.add_argument(
        "--save_folder", 
        default="./outputs/test/", 
        type=str,
        help="Output directory for final results"
    )
    parser.add_argument(
        "--text_prompt", 
        default="", 
        type=str,
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "--image_prompt", 
        default="", 
        type=str,
        help="Path to input image (alternative to text prompt)"
    )
    parser.add_argument(
        "--device", 
        default="cuda:0", 
        type=str,
        help="Device to run models on"
    )
    parser.add_argument(
        "--t2i_seed", 
        default=0, 
        type=int,
        help="Seed for text-to-image generation"
    )
    parser.add_argument(
        "--t2i_steps", 
        default=25, 
        type=int,
        help="Number of steps for text-to-image generation"
    )
    parser.add_argument(
        "--gen_seed", 
        default=0, 
        type=int,
        help="Seed for other generation steps"
    )
    parser.add_argument(
        "--gen_steps", 
        default=50, 
        type=int,
        help="Number of steps for other generation steps"
    )
    parser.add_argument(
        "--max_faces_num", 
        default=120000, 
        type=int, 
        help="Max number of faces (suggest 120000 for vertex color, 10000 for texture/baking color)"
    )
    parser.add_argument(
        "--save_memory", 
        default=False, 
        action="store_true",
        help="Offload models to CPU when not in use"
    )
    parser.add_argument(
        "--do_texture_mapping", 
        default=False, 
        action="store_true",
        help="Perform texture mapping"
    )
    parser.add_argument(
        "--do_render", 
        default=False, 
        action="store_true",
        help="Render final output"
    )
    parser.add_argument(
        "--do_bake", 
        default=False, 
        action="store_true",
        help="Perform baking"
    )
    parser.add_argument(
        "--bake_align_times", 
        default=3, 
        type=int,
        help="Number of alignment iterations for baking (suggest 1-6)"
    )
    # Add FLUX-specific parameters
    parser.add_argument(
        "--image_height",
        default=1024,
        type=int,
        help="Height of generated image (should be multiple of 32)"
    )
    parser.add_argument(
        "--image_width",
        default=1024,
        type=int,
        help="Width of generated image (should be multiple of 32)"
    )
    parser.add_argument(
        "--guidance_scale",
        default=3.5,
        type=float,
        help="Guidance scale for FLUX generation"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    
    # Validate input arguments
    assert not (args.text_prompt and args.image_prompt), "Only one of text_prompt or image_prompt should be provided"
    assert args.text_prompt or args.image_prompt, "Either text_prompt or image_prompt must be provided"

    # Validate FLUX-specific parameters
    assert args.image_height % 32 == 0, f"Image height must be multiple of 32, got {args.image_height}"
    assert args.image_width % 32 == 0, f"Image width must be multiple of 32, got {args.image_width}"
    assert args.guidance_scale >= 0, f"Guidance scale must be non-negative, got {args.guidance_scale}"

    # Create output directories
    os.makedirs(args.save_folder, exist_ok=True)

    # Initialize models
    st = time.time()
    
    print("Initializing models...")
    rembg_model = Removebg()
    image_to_views_model = Image2Views(
        device=args.device, 
        use_lite=args.use_lite,
        save_memory=args.save_memory
    )
    
    views_to_mesh_model = Views2Mesh(
        args.mv23d_cfg_path, 
        args.mv23d_ckt_path, 
        args.device, 
        use_lite=args.use_lite,
        save_memory=args.save_memory
    )
    
    if args.text_prompt:
        text_to_image_model = Text2Image(
            pretrain=args.text2image_path,
            device=args.device, 
            save_memory=args.save_memory
        )
        
    if args.do_bake and BAKE_AVAILABLE:
        mesh_baker = MeshBaker(
            device=args.device,
            align_times=args.bake_align_times
        )
            
    if check_bake_available():
        gif_renderer = GifRenderer(device=args.device)
        
    print(f"Model initialization completed in {time.time()-st:.2f}s")
    
    try:
        # Stage 1: Text to Image or Load Image
        if args.text_prompt:
            print(f"\nStage 1: Generating image from text: '{args.text_prompt}'")
            generator = torch.Generator(device=args.device).manual_seed(args.t2i_seed) if args.t2i_seed is not None else None
            res_rgb_pil = text_to_image_model(
                args.text_prompt,
                seed=args.t2i_seed,
                steps=args.t2i_steps
            )
            
            # Save to final output directory
            res_rgb_pil.save(os.path.join(args.save_folder, "img.jpg"))
        else:
            print(f"\nStage 1: Loading image from: {args.image_prompt}")
            res_rgb_pil = Image.open(args.image_prompt)

        # Stage 2: Remove Background
        print("\nStage 2: Removing background")
        res_rgba_pil = rembg_model(res_rgb_pil)
        res_rgba_pil.save(os.path.join(args.save_folder, "img_nobg.png"))

        # Stage 3: Generate Views
        print("\nStage 3: Generating multiple views")
        (views_grid_pil, cond_img), view_pil_list = image_to_views_model(
            res_rgba_pil,
            seed=args.gen_seed,
            steps=args.gen_steps
        )
        views_grid_pil.save(os.path.join(args.save_folder, "views.jpg"))

        # Stage 4: Generate Mesh
        print("\nStage 4: Generating 3D mesh")
        views_to_mesh_model(
            views_grid_pil, 
            cond_img, 
            seed=args.gen_seed,
            target_face_count=args.max_faces_num,
            save_folder=args.save_folder,
            do_texture_mapping=args.do_texture_mapping
        )
        
        # Stage 5: Baking (Optional)
        mesh_file_for_render = None
        if args.do_bake and BAKE_AVAILABLE:
            print("\nStage 5: Performing baking")
            mesh_file_for_render = mesh_baker(args.save_folder)
            
        # Stage 6: Render GIF (Optional)
        if args.do_render:
            print("\nStage 6: Rendering output GIF")
            if mesh_file_for_render and os.path.exists(mesh_file_for_render):
                mesh_file_for_render = mesh_file_for_render
            else:
                baked_fld_list = sorted(glob(args.save_folder + '/view_*/bake/mesh.obj'))
                mesh_file_for_render = baked_fld_list[-1] if len(baked_fld_list)>=1 else args.save_folder+'/mesh.obj'
                assert os.path.exists(mesh_file_for_render), f"Mesh file not found: {mesh_file_for_render}"
                
            print(f"Rendering 3D file: {mesh_file_for_render}")
            
            gif_renderer(
                mesh_file_for_render,
                gif_dst_path=os.path.join(args.save_folder, 'output.gif'),
            )
            
        print("\nPipeline completed successfully!")
        
    except Exception as e:
        print(f"\nError in pipeline execution: {str(e)}")
        raise
