import argparse
import wandb   
import copy    
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics

import os
import torch

from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from optim_utils import *
from io_utils import *
from pytorch_fid.fid_score import *


def main(args):
    table = None
    if args.with_tracking:
        wandb.init(project='diffusion_watermark', name=args.run_name, tags=['tree_ring_watermark'])
        wandb.config.update(args)
        table = wandb.Table(columns=['gen_no_w', 'gen_w','gen_z','gen_ra','gen_c','gen_sr','gen_sra','gen_sz','prompt'])
    
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
        )
    pipe = pipe.to(device)

    # reference model
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model, pretrained=args.reference_model_pretrain, device=device)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    '''
    dataset, prompt_key = get_dataset(args)
    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)
    '''

    with open(args.dataset) as f:
        dataset = json.load(f)
        image_files = dataset['images']
        dataset = dataset['annotations']
        prompt_key = 'caption'
    

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed
        current_prompt = dataset[i][prompt_key]
        
        ### generation
        # generation without watermarking
        set_random_seed(seed)
        init_latents_no_w = pipe.get_random_latents()
        outputs_no_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_no_w,
            )         
        orig_image_no_w = outputs_no_w.images[0]
                      
        # generation with watermarking
        if init_latents_no_w is None:
            set_random_seed(seed)
            init_latents_w = pipe.get_random_latents()
        else:
            init_latents_w = copy.deepcopy(init_latents_no_w)
        
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)
        gt_patch = get_watermarking_pattern(pipe, args, device, "ring")


        # inject watermark
        init_latents_w_r = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args,"complex")
        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w_r,
            )
        orig_image_w_ring = outputs_w.images[0]
        
                
        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)
        gt_patch = get_watermarking_pattern(pipe, args, device,"rand")
        # inject watermark
        init_latents_w_ra = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args,"complex")
        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w_ra,
            )
        orig_image_w_rand = outputs_w.images[0]

                
        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)
        gt_patch = get_watermarking_pattern(pipe, args, device,"zeros")
        # inject watermark
        init_latents_w_z = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args,"complex")
        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w_z,
            )
        orig_image_w_zero = outputs_w.images[0]

        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)
        gt_patch = get_watermarking_pattern(pipe, args, device,"seed_ring")
        # inject watermark
        init_latents_w_sr = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args,"seed")
        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w_sr,
            )
        orig_image_w_sring = outputs_w.images[0]

        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)
        gt_patch = get_watermarking_pattern(pipe, args, device,"seed_zeros")
        # inject watermark
        init_latents_w_sz = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args,"seed")
        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w_sz,
            )
        orig_image_w_szero = outputs_w.images[0]
        
        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)
        gt_patch = get_watermarking_pattern(pipe, args, device,"seed_rand")
        # inject watermark
        init_latents_w_sra = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args,"seed")
        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w_sra,
            )
        orig_image_w_srand = outputs_w.images[0]

        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)
        gt_patch = get_watermarking_pattern(pipe, args, device,"const")
        # inject watermark
        init_latents_w = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args,"complex")
        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
            )
        orig_image_w_const = outputs_w.images[0]
       
        output_dir = f"fid_outputs/coco/{args.run_name}"
        os.makedirs(output_dir, exist_ok=True)
        no_w_dir = os.path.join(output_dir, "no_watermark")
        ring = os.path.join(output_dir, "ring")
        zero = os.path.join(output_dir, "zero")
        rand = os.path.join(output_dir, "rand")
        szero = os.path.join(output_dir, "szero")
        sring = os.path.join(output_dir, "sring")
        srand = os.path.join(output_dir, "srand")
        const = os.path.join(output_dir, "const")

        os.makedirs(no_w_dir, exist_ok=True)
        os.makedirs(ring, exist_ok=True)
        os.makedirs(zero, exist_ok=True)
        os.makedirs(rand, exist_ok=True)
        os.makedirs(const, exist_ok=True)
        os.makedirs(szero, exist_ok=True)
        os.makedirs(sring, exist_ok=True)
        os.makedirs(srand, exist_ok=True)

        
        orig_image_no_w.save(os.path.join(no_w_dir, f"image_{i}.png"))
        orig_image_w_ring.save(os.path.join(ring, f"image_{i}.png"))
        orig_image_w_zero.save(os.path.join(zero, f"image_{i}.png"))
        orig_image_w_rand.save(os.path.join(rand, f"image_{i}.png"))
        orig_image_w_szero.save(os.path.join(szero, f"image_{i}.png"))
        orig_image_w_sring.save(os.path.join(sring, f"image_{i}.png"))
        orig_image_w_srand.save(os.path.join(srand, f"image_{i}.png"))
        orig_image_w_const.save(os.path.join(const, f"image_{i}.png"))

        table.add_data(
        wandb.Image(orig_image_no_w), 
        wandb.Image(orig_image_w_ring), 
        wandb.Image(orig_image_w_rand), 
        wandb.Image(orig_image_w_zero), 
        wandb.Image(orig_image_w_sring), 
        wandb.Image(orig_image_w_szero), 
        wandb.Image(orig_image_w_srand), 
        wandb.Image(orig_image_w_const), 
        current_prompt
        )

    try:
        num_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        num_cpus = os.cpu_count()

    num_workers = min(num_cpus, 8) if num_cpus is not None else 0
    
    fid_value = calculate_fid_given_paths(['fid_outputs/coco/ground_truth', no_w_dir],50,device,2048,num_workers)
    fid_value_r = calculate_fid_given_paths([no_w_dir, ring],50,device,2048,num_workers)
    fid_value_z = calculate_fid_given_paths([no_w_dir, zero],50,device,2048,num_workers)
    fid_value_rd = calculate_fid_given_paths([no_w_dir, rand],50,device,2048,num_workers)
    

    fid_value_sz = calculate_fid_given_paths(['fid_outputs/coco/ground_truth', szero],50,device,2048,num_workers)
    fid_value_sr = calculate_fid_given_paths(['fid_outputs/coco/ground_truth', sring],50,device,2048,num_workers)
    fid_value_srd = calculate_fid_given_paths(['fid_outputs/coco/ground_truth', srand],50,device,2048,num_workers)
    fid_value_c = calculate_fid_given_paths(['fid_outputs/coco/ground_truth', const],50,device,2048,num_workers)


    if args.with_tracking:
        wandb.log({
            'Table': table,
            'fid_no_w': fid_value,
            'fid_ring': fid_value_r,
            'fid_zero': fid_value_z,
            'fid_rand': fid_value_rd,
            'fid_szero': fid_value_sz,
            'fid_sring': fid_value_sr,
            'fid_srand': fid_value_srd,
            'fid_const': fid_value_c,
        })

        print(f'fid_no_w: {fid_value}, fid_ring: {fid_value_r}, fid_zero: {fid_value_z}, fid_rand: {fid_value_rd}, '
            f'fid_szero: {fid_value_sz}, fid_sring: {fid_value_sr}, fid_srand: {fid_value_srd}, fid_const: {fid_value_c}')

import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default=f"fid_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument('--dataset', default='fid_outputs/coco/meta_data.json')
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