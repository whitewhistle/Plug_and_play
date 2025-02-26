from inverse_stable_diffusion import InversableStableDiffusionPipeline
import torch
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
from optim_utils import *
import argparse
from io_utils import *
from scipy.special import kl_div

def jsd(P, Q):
    # Normalize P and Q to ensure they are probability distributions
    P = np.array(P) / np.sum(P)
    Q = np.array(Q) / np.sum(Q)

    # Compute the midpoint distribution M
    M = 0.5 * (P + Q)

    # Compute the KL divergence between P and M, and Q and M
    kl_pm = kl_div(P, M).sum()  # KL(P || M)
    kl_qm = kl_div(Q, M).sum()  # KL(Q || M)

    # The JSD is the average of the two KL divergences
    return 0.5 * (kl_pm + kl_qm)

def main(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DPMSolverMultistepScheduler.from_pretrained('stabilityai/stable-diffusion-2-1-base', subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        'stabilityai/stable-diffusion-2-1-base',
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
        )
    pipe = pipe.to(device)
    
    all_pixel_values=[]
    all_pixel_values_r=[]
    all_pixel_values_ra=[]
    all_pixel_values_z=[]
    all_pixel_values_sz=[]
    all_pixel_values_sr=[]
    all_pixel_values_sra=[]


    for i in tqdm(range(args.start,args.end)):
        init_latents= pipe.get_random_latents()
        pixel_values = init_latents[0][3].cpu().numpy().flatten()

        watermarking_mask = get_watermarking_mask(init_latents, args, device)

        gt_patch = get_watermarking_pattern(pipe, args, device, 'ring')
        init_latents_r = inject_watermark(init_latents, watermarking_mask, gt_patch, args,"complex")
        pixel_values_r = init_latents_r[0][3].cpu().numpy().flatten()

        gt_patch = get_watermarking_pattern(pipe, args, device, "zeros")
        init_latents_z = inject_watermark(init_latents, watermarking_mask, gt_patch, args,"complex")
        pixel_values_z = init_latents_z[0][3].cpu().numpy().flatten()

        gt_patch = get_watermarking_pattern(pipe, args, device, "rand")
        init_latents_ra = inject_watermark(init_latents, watermarking_mask, gt_patch, args,"complex")
        pixel_values_ra = init_latents_ra[0][3].cpu().numpy().flatten()

        gt_patch = get_watermarking_pattern(pipe, args, device, "seed_ring")
        init_latents_sr = inject_watermark(init_latents, watermarking_mask, gt_patch, args,"seed")
        pixel_values_sr = init_latents_sr[0][3].cpu().numpy().flatten()                        

        gt_patch = get_watermarking_pattern(pipe, args, device, "seed_rand")
        init_latents_sra = inject_watermark(init_latents, watermarking_mask, gt_patch, args,"seed")
        pixel_values_sra = init_latents_sra[0][3].cpu().numpy().flatten()

        gt_patch = get_watermarking_pattern(pipe, args, device, "seed_zeros")
        init_latents_sz = inject_watermark(init_latents, watermarking_mask, gt_patch, args,"seed")
        pixel_values_sz = init_latents_sz[0][3].cpu().numpy().flatten()


        all_pixel_values.extend(pixel_values)
        all_pixel_values_r.extend(pixel_values_r)
        all_pixel_values_z.extend(pixel_values_z)
        all_pixel_values_ra.extend(pixel_values_ra)
        all_pixel_values_sr.extend(pixel_values_sr)
        all_pixel_values_sra.extend(pixel_values_sra)
        all_pixel_values_sz.extend(pixel_values_sz)

    bins = np.linspace(min(min(all_pixel_values), min(all_pixel_values_r), min(all_pixel_values_z), min(all_pixel_values_ra), min(all_pixel_values_sr), min(all_pixel_values_sra), min(all_pixel_values_sz)), 
                    max(max(all_pixel_values), max(all_pixel_values_r), max(all_pixel_values_z), max(all_pixel_values_ra), max(all_pixel_values_sr), max(all_pixel_values_sra), max(all_pixel_values_sz)), 
                    100) 

    # Compute histograms for each watermarking pattern
    hist_nw, bin_edges_nw = np.histogram(all_pixel_values, bins=bins, density=True)
    hist_r, bin_edges_r = np.histogram(all_pixel_values_r, bins=bins, density=True)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Original Distribution
    axes[0].bar(bin_edges_nw[:-1], hist_nw, width=np.diff(bin_edges_nw), alpha=0.7, color='b', label="Original")
    axes[0].set_xlabel("Pixel Intensity")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Original Histogram")
    axes[0].legend()
    axes[0].grid(True)

    # Plot Transformed Distribution
    axes[1].bar(bin_edges_r[:-1], hist_r, width=np.diff(bin_edges_r), alpha=0.7, color='r', label="Transformed")
    axes[1].set_xlabel("Pixel Intensity")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Transformed Histogram")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    hist_z, _ = np.histogram(all_pixel_values_z, bins=bins, density=True)
    hist_ra, _ = np.histogram(all_pixel_values_ra, bins=bins, density=True)
    hist_sr, _ = np.histogram(all_pixel_values_sr, bins=bins, density=True)
    hist_sra, _ = np.histogram(all_pixel_values_sra, bins=bins, density=True)
    hist_sz, _ = np.histogram(all_pixel_values_sz, bins=bins, density=True)

    # Avoid zero probabilities

    hist_nw += 1e-10
    hist_r += 1e-10
    hist_z += 1e-10
    hist_ra += 1e-10
    hist_sr += 1e-10
    hist_sra += 1e-10
    hist_sz += 1e-10

    # Calculate JSD between histograms
    jsd_r = jsd(hist_nw, hist_r)
    jsd_ra = jsd(hist_nw, hist_ra)
    jsd_sr = jsd(hist_nw, hist_sr)
    jsd_sra = jsd(hist_nw, hist_sra)
    jsd_sz = jsd(hist_nw, hist_sz)
    jsd_z = jsd(hist_nw, hist_z)

    print(f'Jensen-Shannon Divergence (hist_nw vs hist_r): {jsd_r}')
    print(f'Jensen-Shannon Divergence (hist_nw vs hist_ra): {jsd_ra}')
    print(f'Jensen-Shannon Divergence (hist_nw vs hist_sr): {jsd_sr}')
    print(f'Jensen-Shannon Divergence (hist_nw vs hist_sra): {jsd_sra}')
    print(f'Jensen-Shannon Divergence (hist_nw vs hist_sz): {jsd_sz}')
    print(f'Jensen-Shannon Divergence (hist_nw vs hist_z): {jsd_z}')

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

'''
KL Divergence ring: 0.054888846567015176
KL Divergence rand: 0.051002081563955
KL Divergence seed ring: 0.003338158338916128
KL Divergence seed rand: 0.0005022497182810166
KL Divergence seed zero: 0.00659376310973513
KL Divergence zero: 0.03667846235228483      

KL Divergence ring: 0.032169923223148614
KL Divergence rand: 0.0405147988712209
KL Divergence seed ring: 0.0034120430999252426
KL Divergence seed rand: 0.0004569024863606844
KL Divergence seed zero: 0.006020207598936763
KL Divergence zero: 0.04028287326950613

KL Divergence ring: 0.028382293704831713
KL Divergence rand: 0.03057848912171447
KL Divergence seed ring: 0.0028814146837220908
KL Divergence seed rand: 0.0004071126188149772
KL Divergence seed zero: 0.00588959602812501
KL Divergence zero: 0.027085116895759397

Ring: 0.0385
Rand: 0.0407
Seed Ring: 0.0032
Seed Rand: 0.0005
Seed Zero: 0.0062
Zero: 0.0347
'''