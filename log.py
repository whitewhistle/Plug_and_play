import wandb
import os
from glob import glob
from dreamsim import dreamsim
from PIL import Image

# Initialize WandB
wandb.init(project='diffusion_watermark', name="100images", tags=['tree_ring_watermark'])

# Define table columns, including similarity scores and running average
table = wandb.Table(columns=['gen_no_w', 'gen_r', 'gen_z', 'gen_ra', 'gen_sr', 'gen_sra', 'gen_sz', 'prompt'])

output_dir = "generated_images"
subdirs = ["no_watermark", "ring", "zero", "rand", "szero", "sring", "srand"]

# Create full paths for each directory
image_dirs = {name: os.path.join(output_dir, name) for name in subdirs}

# Ensure directories exist
for dir_path in image_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Get list of image files (assuming they are sorted)
image_lists = {name: sorted(glob(os.path.join(path, "*"))) for name, path in image_dirs.items()}

# Ensure all directories have the same number of images
num_images = len(image_lists["no_watermark"])
assert all(len(images) == num_images for images in image_lists.values()), "Mismatch in number of images across directories."

# Load DreamSim model
device = "cuda"
model, preprocess = dreamsim(pretrained=True, device=device)

# Running sum of similarity scores
cumulative_sim_r = 0
cumulative_sim_z = 0
cumulative_sim_ra = 0
cumulative_sim_sr = 0
cumulative_sim_sra = 0
cumulative_sim_sz = 0

# Log images and similarity scores
for i in range(num_images):
    # Load and preprocess images
    img1 = preprocess(Image.open(image_lists["no_watermark"][i])).to(device)
    img_r = preprocess(Image.open(image_lists["ring"][i])).to(device)
    img_z = preprocess(Image.open(image_lists["zero"][i])).to(device)
    img_ra = preprocess(Image.open(image_lists["rand"][i])).to(device)
    img_sr = preprocess(Image.open(image_lists["sring"][i])).to(device)
    img_sra = preprocess(Image.open(image_lists["srand"][i])).to(device)
    img_sz = preprocess(Image.open(image_lists["szero"][i])).to(device)

    # Compute DreamSim similarity scores
    sim_r = model(img1, img_r).item()
    sim_z = model(img1, img_z).item()
    sim_ra = model(img1, img_ra).item()
    sim_sr = model(img1, img_sr).item()
    sim_sra = model(img1, img_sra).item()
    sim_sz = model(img1, img_sz).item()

    # Compute the running average of all similarity scores
    cumulative_sim_r += sim_r
    cumulative_sim_z += sim_z
    cumulative_sim_ra += sim_ra
    cumulative_sim_sr += sim_sr
    cumulative_sim_sra += sim_sra
    cumulative_sim_sz += sim_sz

    # Compute running averages
    drm_sim_r = cumulative_sim_r / (i + 1)
    drm_sim_z = cumulative_sim_z / (i + 1)
    drm_sim_ra = cumulative_sim_ra / (i + 1)
    drm_sim_sr = cumulative_sim_sr / (i + 1)
    drm_sim_sra = cumulative_sim_sra / (i + 1)
    drm_sim_sz = cumulative_sim_sz / (i + 1)

    
    # Log images and similarity scores into the table
    table.add_data(
        wandb.Image(image_lists["no_watermark"][i]),
        wandb.Image(image_lists["ring"][i]),
        wandb.Image(image_lists["zero"][i]),
        wandb.Image(image_lists["rand"][i]),
        wandb.Image(image_lists["sring"][i]),
        wandb.Image(image_lists["srand"][i]),
        wandb.Image(image_lists["szero"][i]),
        f"Prompt {i}"  # Replace with actual prompt if available
    )

    # Log the running average separately for visualization
    wandb.log({
        "Running Average Similarity (R)": drm_sim_r,
        "Running Average Similarity (Z)": drm_sim_z,
        "Running Average Similarity (RA)": drm_sim_ra,
        "Running Average Similarity (SR)": drm_sim_sr,
        "Running Average Similarity (SRA)": drm_sim_sra,
        "Running Average Similarity (SZ)": drm_sim_sz
    })

# Log table to Weights & Biases
wandb.log({'Table': table})
