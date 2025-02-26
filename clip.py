import torch
import open_clip
from optim_utils import *
from PIL import Image
from tqdm import tqdm
import argparse

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-g-14', pretrained='laion2b_s12b_b42k', device=device
    )
    ref_tokenizer = open_clip.get_tokenizer('ViT-g-14')

    
    
    dataset, prompt_key = get_dataset(args)
    sum_sims=[0,0,0,0,0,0,0]
    sum_sims = torch.tensor(sum_sims, dtype=torch.float32, device=device)
    for i in tqdm(range(0,99)):   
        current_prompt = dataset[i][prompt_key]
        img_paths = [
        f"generated_images/no_watermark/image_{i}.png",
        f"generated_images/srand/image_{i}.png",
        f"generated_images/sring/image_{i}.png",
        f"generated_images/szero/image_{i}.png",
        f"generated_images/ring/image_{i}.png",
        f"generated_images/zero/image_{i}.png",
        f"generated_images/rand/image_{i}.png"
        ]
        images = [Image.open(img).convert("RGB") for img in img_paths]
        sims = measure_similarity(
            images,  # Ensure `measure_similarity` can handle tensors
            current_prompt, 
            ref_model, ref_clip_preprocess, ref_tokenizer, device
        )
        sum_sims += sims
        print(sum_sims) 
    print(sum_sims/100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)

    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='seed_zero')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    
    # for image distortion
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)

    args = parser.parse_args()

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps
    
    main(args)

tensor([0.3607, 0.3612, 0.3586, 0.3607, 0.3617, 0.3576, 0.3550],