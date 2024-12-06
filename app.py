# Open Source Model Licensed under the Apache License Version 2.0 
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited 
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

import os
import warnings
import argparse
import gradio as gr
from glob import glob
import shutil
import torch
import numpy as np
from PIL import Image
from einops import rearrange
import pandas as pd
from urllib.parse import urlparse
import re

# import spaces
from infer import seed_everything, save_gif
from infer import Text2Image, Removebg, Image2Views, Views2Mesh, GifRenderer
from third_party.check import check_bake_available

try:
    from third_party.mesh_baker import MeshBaker
    assert check_bake_available()
    BAKE_AVAILEBLE = True
except Exception as err:
    print(err)
    print("import baking related fail, run without baking")
    BAKE_AVAILEBLE = False

warnings.simplefilter('ignore', category=UserWarning)
warnings.simplefilter('ignore', category=FutureWarning)
warnings.simplefilter('ignore', category=DeprecationWarning)

parser = argparse.ArgumentParser()
parser.add_argument("--use_lite", default=False, action="store_true")
parser.add_argument("--mv23d_cfg_path", default="./svrm/configs/svrm.yaml", type=str)
parser.add_argument("--mv23d_ckt_path", default="weights/svrm/svrm.safetensors", type=str)
parser.add_argument("--text2image_path", default="weights/flux1dev", type=str)
parser.add_argument("--save_memory", default=False)
parser.add_argument("--device", default="cuda:0", type=str)
args = parser.parse_args()

################################################################
# initial setting
################################################################

CONST_NOTE = '''
❗️❗️❗️Usage❗️❗️❗️<br>

Limited by format, the model can only export *.obj mesh with vertex colors. The "face" mod can only work on *.glb.<br>
Please click "Do Rendering" to export a GIF.<br>
You can click "Do Baking" to bake multi-view images onto the shape.<br>
For LoRA support, paste a valid HuggingFace LoRA URL and adjust the scale as needed.<br>

If the results aren't satisfactory, please try a different random seed (default is 0).
'''

################################################################
# prepare text examples and image examples
################################################################

def get_example_img_list():
    print('Loading example img list ...')
    return sorted(glob('./demos/example_*.png'))

def get_example_txt_list():
    print('Loading example txt list ...')
    txt_list  = list()
    for line in open('./demos/example_list.txt'):
        txt_list.append(line.strip())
    return txt_list

example_is = get_example_img_list()
example_ts = get_example_txt_list()

################################################################
# initial models
################################################################

worker_xbg = Removebg()
print(f"loading {args.text2image_path}")
worker_t2i = Text2Image(
    pretrain=args.text2image_path, 
    device=args.device, 
    save_memory=args.save_memory
)
worker_i2v = Image2Views(
    use_lite=args.use_lite, 
    device=args.device,
    save_memory=args.save_memory
)
worker_v23 = Views2Mesh(
    args.mv23d_cfg_path, 
    args.mv23d_ckt_path, 
    use_lite=args.use_lite, 
    device=args.device,
    save_memory=args.save_memory
)
worker_gif = GifRenderer(args.device)

if BAKE_AVAILEBLE:
    worker_baker = MeshBaker()

### functional modules    

def gen_save_folder(max_size=30):
    os.makedirs('./outputs/app_output', exist_ok=True)
    exists = set(int(_) for _ in os.listdir('./outputs/app_output') if not _.startswith("."))
    if len(exists) == max_size: 
        shutil.rmtree(f"./outputs/app_output/0")
        cur_id = 0
    else:
        cur_id = min(set(range(max_size)) - exists)
    if os.path.exists(f"./outputs/app_output/{(cur_id + 1) % max_size}"):
        shutil.rmtree(f"./outputs/app_output/{(cur_id + 1) % max_size}")
    save_folder = f'./outputs/app_output/{cur_id}'
    os.makedirs(save_folder, exist_ok=True)
    print(f"mkdir {save_folder} success !!!")
    return save_folder

def validate_lora_url(url):
    if not url:
        return True
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ('http', 'https'):
            return "URL must start with http:// or https://"
        if 'huggingface.co' in parsed.netloc:
            path_parts = parsed.path.strip('/').split('/')
            if len(path_parts) < 2:
                return "Invalid HuggingFace URL format"
        return True
    except:
        return "Invalid URL format"

def stage_0_t2i(text, seed, step, save_folder, lora_url="", lora_scale=1.0):
    dst = save_folder + '/img.png'
    image = worker_t2i(text, seed, step, lora_url=lora_url if lora_url else None, lora_scale=lora_scale)
    image.save(dst)
    img_nobg = worker_xbg(image, force=True)
    dst = save_folder + '/img_nobg.png'
    img_nobg.save(dst)
    return dst

def stage_1_xbg(image, save_folder, force_remove): 
    if isinstance(image, str):
        image = Image.open(image)
    dst = save_folder + '/img_nobg.png'
    rgba = worker_xbg(image, force=force_remove)
    rgba.save(dst)
    return dst

def stage_2_i2v(image, seed, step, save_folder):
    if isinstance(image, str):
        image = Image.open(image)
    gif_dst = save_folder + '/views.gif'
    res_img, pils = worker_i2v(image, seed, step)
    save_gif(pils, gif_dst)
    views_img, cond_img = res_img[0], res_img[1]
    img_array = np.asarray(views_img, dtype=np.uint8)
    show_img = rearrange(img_array, '(n h) (m w) c -> (n m) h w c', n=3, m=2)
    show_img = show_img[worker_i2v.order, ...]
    show_img = rearrange(show_img, '(n m) h w c -> (n h) (m w) c', n=2, m=3)
    show_img = Image.fromarray(show_img) 
    return views_img, cond_img, show_img

def stage_3_v23(
    views_pil, 
    cond_pil, 
    seed, 
    save_folder,
    target_face_count=30000,
    texture_color='face'
): 
    do_texture_mapping = texture_color == 'face'
    worker_v23(
        views_pil, 
        cond_pil, 
        seed=seed, 
        save_folder=save_folder,
        target_face_count=target_face_count,
        do_texture_mapping=do_texture_mapping
    )
    glb_dst = save_folder + '/mesh.glb' if do_texture_mapping else None
    obj_dst = save_folder + '/mesh_vertex_colors.obj'
    return obj_dst, glb_dst

def stage_3p_baking(save_folder, color, bake, force, front, others, align_times):
    if color == "face" and bake:
        obj_dst = worker_baker(save_folder, force, front, others, align_times)
        glb_dst = obj_dst.replace(".obj", ".glb")
        return glb_dst
    else:
        return None

def stage_4_gif(save_folder, color, bake, render):
    if not render: return None
    baked_fld_list = sorted(glob(save_folder + '/view_*/bake/mesh.obj'))
    obj_dst = baked_fld_list[-1] if len(baked_fld_list)>=1 else save_folder+'/mesh.obj'
    assert os.path.exists(obj_dst), f"{obj_dst} file not found"
    gif_dst = obj_dst.replace(".obj", ".gif")
    worker_gif(obj_dst, gif_dst_path=gif_dst)
    return gif_dst

def check_image_available(image):
    if image is None:
        return "Please upload image", gr.update()
    elif not hasattr(image, 'mode'):
        return "Not support, please upload other image", gr.update()
    elif image.mode == "RGBA":
        data = np.array(image)
        alpha_channel = data[:, :, 3]
        unique_alpha_values = np.unique(alpha_channel)
        if len(unique_alpha_values) == 1:
            msg = "The alpha channel is missing or invalid. The background removal option is selected for you."
            return msg, gr.update(value=True, interactive=False)
        else:
            msg = "The image has four channels, and you can choose to remove the background or not."
            return msg, gr.update(value=False, interactive=True)
    elif image.mode == "RGB":
        msg = "The alpha channel is missing or invalid. The background removal option is selected for you."
        return msg, gr.update(value=True, interactive=False)
    else:
        raise Exception("Image Error")

def update_mode(mode):
    color_change = {
        'Vertex color': gr.update(value='vertex'),
        'Face color': gr.update(value='face'),
        'Baking': gr.update(value='face')
    }[mode]
    bake_change = {
        'Vertex color': gr.update(value=False, interactive=False, visible=False),
        'Face color': gr.update(value=False),
        'Baking': gr.update(value=BAKE_AVAILEBLE)
    }[mode]
    face_change = {
        'Vertex color': gr.update(value=120000, maximum=300000),
        'Face color': gr.update(value=60000, maximum=300000),
        'Baking': gr.update(value=10000, maximum=60000)
    }[mode]
    render_change = {
        'Vertex color': gr.update(value=False, interactive=False, visible=False),
        'Face color': gr.update(value=True),
        'Baking': gr.update(value=True)
    }[mode]
    return color_change, bake_change, face_change, render_change

# ===============================================================
# gradio display
# ===============================================================

with gr.Blocks() as demo:
    with gr.Row(variant="panel"):
        
        ###### Input region
        
        with gr.Column(scale=2):
            
            ### Text input region
            
            with gr.Tab("Text to 3D"):
                with gr.Column():
                    text = gr.TextArea("A blocky sign spelling out \"NeuronsLab\"", 
                                     lines=3, max_lines=20, label='Input text (within 70 words)')

                    # Add LoRA controls
                    with gr.Row():
                        lora_url = gr.Textbox(
                            label="LoRA URL (optional)", 
                            placeholder="https://huggingface.co/path/to/lora",
                            value=""
                        )
                        lora_scale = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.05,
                            label="LoRA Scale"
                        )

                    textgen_mode = gr.Radio(
                        choices=['Vertex color', 'Face color', 'Baking'], 
                        label="Texture mode",
                        value='Baking',
                        interactive=True
                    )
                    
                    with gr.Accordion("Custom settings", open=False):
                        textgen_color = gr.Radio(choices=["vertex", "face"], label="Color", value="face")
                        
                        with gr.Row():
                            textgen_render = gr.Checkbox(
                                label="Do Rendering", 
                                value=True, 
                                interactive=True
                            )
                            textgen_bake = gr.Checkbox(
                                label="Do Baking", 
                                value=True if BAKE_AVAILEBLE else False, 
                                interactive=True if BAKE_AVAILEBLE else False
                            )
                            
                        with gr.Row():
                            textgen_seed = gr.Number(value=0, label="T2I seed", precision=0, interactive=True)
                            textgen_SEED = gr.Number(value=0, label="Gen seed", precision=0, interactive=True)

                        textgen_step = gr.Slider(
                            value=25,
                            minimum=15,
                            maximum=50,
                            step=1,
                            label="T2I steps",
                            interactive=True
                        )
                        textgen_STEP = gr.Slider(
                            value=50,
                            minimum=20,
                            maximum=80,
                            step=1,
                            label="Gen steps",
                            interactive=True
                        )
                        textgen_max_faces = gr.Slider(
                            value=10000,
                            minimum=2000,
                            maximum=60000,
                            step=1000,
                            label="Face number limit",
                            interactive=True
                        )

                        with gr.Accordion("Baking Options", open=False):
                            textgen_force_bake = gr.Checkbox(
                                label="Force (Ignore the degree of matching)", 
                                value=False, 
                                interactive=True
                            )
                            textgen_front_baking = gr.Radio(
                                choices=['input image', 'multi-view front view', 'auto'], 
                                label="Front view baking",
                                value='auto',
                                interactive=True,
                                visible=True
                            )
                            textgen_other_views = gr.CheckboxGroup(
                                choices=['60°', '120°', '180°', '240°', '300°'], 
                                label="Other views baking",
                                value=['180°'],
                                interactive=True,
                                visible=True
                            )
                            textgen_align_times = gr.Slider(
                                value=3,
                                minimum=1,
                                maximum=5,
                                step=1,
                                label="Number of alignment attempts per view",
                                interactive=True
                            )
                        
                    with gr.Row():
                        textgen_submit = gr.Button("Generate", variant="primary")

                    with gr.Row():
                        gr.Examples(examples=example_ts, inputs=[text], label="Text examples", examples_per_page=10)
            
            ### Image input region
            
            with gr.Tab("Image to 3D"):
                with gr.Row():
                    input_image = gr.Image(label="Input image", width=256, height=256, type="pil",
                                           image_mode="RGBA", sources="upload", interactive=True)
                with gr.Row():
                    alert_message = gr.Markdown("")  # for warning 
                    
                imggen_mode = gr.Radio(
                    choices=['Vertex color', 'Face color', 'Baking'], 
                    label="Texture mode",
                    value='Baking',
                    interactive=True
                )
                
                with gr.Accordion("Custom settings", open=False):
                    imggen_color = gr.Radio(choices=["vertex", "face"], label="Color", value="face")

                    with gr.Row():
                        imggen_removebg = gr.Checkbox(
                            label="Remove Background", 
                            value=True, 
                            interactive=True
                        )
                        imggen_render = gr.Checkbox(
                            label="Do Rendering", 
                            value=True, 
                            interactive=True
                        )
                        imggen_bake = gr.Checkbox(
                            label="Do Baking", 
                            value=True if BAKE_AVAILEBLE else False, 
                            interactive=True if BAKE_AVAILEBLE else False
                        )
                    imggen_SEED = gr.Number(value=0, label="Gen seed", precision=0, interactive=True)

                    imggen_STEP = gr.Slider(
                        value=50,
                        minimum=20,
                        maximum=80,
                        step=1,
                        label="Gen steps",
                        interactive=True
                    )
                    imggen_max_faces = gr.Slider(
                        value=10000,
                        minimum=2000,
                        maximum=60000,
                        step=1000,
                        label="Face number limit",
                        interactive=True
                    )

                    with gr.Accordion("Baking Options", open=False):
                        imggen_force_bake = gr.Checkbox(
                                label="Force (Ignore the degree of matching)", 
                                value=False, 
                                interactive=True
                            )
                        imggen_front_baking = gr.Radio(
                            choices=['input image', 'multi-view front view', 'auto'], 
                            label="Front view baking",
                            value='auto',
                            interactive=True,
                            visible=True
                        )
                        imggen_other_views = gr.CheckboxGroup(
                            choices=['60°', '120°', '180°', '240°', '300°'], 
                            label="Other views baking",
                            value=['180°'],
                            interactive=True,
                            visible=True
                        )
                        imggen_align_times = gr.Slider(
                            value=3,
                            minimum=1,
                            maximum=5,
                            step=1,
                            label="Number of alignment attempts per view",
                            interactive=True
                        )

                with gr.Row():
                    imggen_submit = gr.Button("Generate", variant="primary")      

                with gr.Row():
                    gr.Examples(examples=example_is, inputs=[input_image], 
                              label="Img examples", examples_per_page=10)
                    
            gr.Markdown(CONST_NOTE)
                    
        ###### Output region

        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column(scale=2):
                    rem_bg_image = gr.Image(
                        label="Image without background", 
                        type="pil",
                        image_mode="RGBA", 
                        interactive=False
                    )
                with gr.Column(scale=3):
                    result_image = gr.Image(
                        label="Multi-view images", 
                        type="pil", 
                        interactive=False
                    )
            
            result_3dobj = gr.Model3D(
                clear_color=[0.0, 0.0, 0.0, 0.0],
                label="OBJ vertex color",
                show_label=True,
                visible=True,
                camera_position=[90, 90, None],
                interactive=False
            )
                
            result_3dglb_texture = gr.Model3D(
                clear_color=[0.0, 0.0, 0.0, 0.0],
                label="GLB face color",
                show_label=True,
                visible=True,
                camera_position=[90, 90, None],
                interactive=False)

            result_3dglb_baked = gr.Model3D(
                clear_color=[0.0, 0.0, 0.0, 0.0],
                label="GLB baking",
                show_label=True,
                visible=True,
                camera_position=[90, 90, None],
                interactive=False)
            
            result_gif = gr.Image(label="GIF", interactive=False)
                
            with gr.Row():
                gr.Markdown(
                    "Due to Gradio limitations, OBJ files are displayed with vertex shading only, "
                    "while GLB files can be viewed with face color. <br>For the best experience, "
                    "we recommend downloading the GLB files and opening them with 3D software "
                    "like Blender or MeshLab."
                )

    #===============================================================
    # UI Event Handlers
    #===============================================================
    
    # Validate LoRA URL
    lora_url.change(
        fn=validate_lora_url,
        inputs=[lora_url],
        outputs=[gr.Textbox(visible=False)]
    )
    
    # Mode change handlers
    textgen_mode.change(
        fn=update_mode,
        inputs=textgen_mode, 
        outputs=[textgen_color, textgen_bake, textgen_max_faces, textgen_render]
    )
    
    imggen_mode.change(
        fn=update_mode,
        inputs=imggen_mode, 
        outputs=[imggen_color, imggen_bake, imggen_max_faces, imggen_render]
    )

    # Image upload handler
    input_image.change(
        fn=check_image_available, 
        inputs=input_image, 
        outputs=[alert_message, imggen_removebg]
    )
    
    save_folder = gr.State()
    cond_image = gr.State()
    views_image = gr.State()
    
    def handle_click(save_folder):
        if save_folder is None:
            save_folder = gen_save_folder()
        return save_folder

    # Text generation pipeline
    textgen_submit.click(
        fn=handle_click,
        inputs=[save_folder],
        outputs=[save_folder]
    ).success(
        fn=stage_0_t2i, 
        inputs=[text, textgen_seed, textgen_step, save_folder, lora_url, lora_scale], 
        outputs=[rem_bg_image],
    ).success(
        fn=stage_2_i2v, 
        inputs=[rem_bg_image, textgen_SEED, textgen_STEP, save_folder], 
        outputs=[views_image, cond_image, result_image],
    ).success(
        fn=stage_3_v23, 
        inputs=[views_image, cond_image, textgen_SEED, save_folder, textgen_max_faces, textgen_color], 
        outputs=[result_3dobj, result_3dglb_texture],
    ).success(
        fn=stage_3p_baking, 
        inputs=[save_folder, textgen_color, textgen_bake,
               textgen_force_bake, textgen_front_baking, textgen_other_views, textgen_align_times], 
        outputs=[result_3dglb_baked],
    ).success(
        fn=stage_4_gif, 
        inputs=[save_folder, textgen_color, textgen_bake, textgen_render], 
        outputs=[result_gif],
    ).success(lambda: print('Text_to_3D Done ...\n'))

    # Image generation pipeline
    imggen_submit.click(
        fn=handle_click,
        inputs=[save_folder],
        outputs=[save_folder]
    ).success(
        fn=stage_1_xbg, 
        inputs=[input_image, save_folder, imggen_removebg], 
        outputs=[rem_bg_image],
    ).success(
        fn=stage_2_i2v, 
        inputs=[rem_bg_image, imggen_SEED, imggen_STEP, save_folder], 
        outputs=[views_image, cond_image, result_image],
    ).success(
        fn=stage_3_v23, 
        inputs=[views_image, cond_image, imggen_SEED, save_folder, imggen_max_faces, imggen_color],
        outputs=[result_3dobj, result_3dglb_texture],
    ).success(
        fn=stage_3p_baking, 
        inputs=[save_folder, imggen_color, imggen_bake, 
                imggen_force_bake, imggen_front_baking, imggen_other_views, imggen_align_times], 
        outputs=[result_3dglb_baked],
    ).success(
        fn=stage_4_gif, 
        inputs=[save_folder, imggen_color, imggen_bake, imggen_render], 
        outputs=[result_gif],
    ).success(lambda: print('Image_to_3D Done ...\n'))
    
    #===============================================================
    # Launch server
    #===============================================================
    CONST_PORT = 7860
    CONST_MAX_QUEUE = 1
    CONST_SERVER = '0.0.0.0'

    demo.queue(max_size=CONST_MAX_QUEUE)
    demo.launch(server_name=CONST_SERVER, server_port=CONST_PORT)