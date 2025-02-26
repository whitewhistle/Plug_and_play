import torch
import transformers
import diffusers
import sys
import huggingface_hub
import datasets
import accelerate
import ftfy


print("Hugging Face Hub version:", huggingface_hub.__version__)
print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("Diffusers version:", diffusers.__version__)
print("datasets:", datasets.__version__)
print("accelerate:", accelerate.__version__)
print("ftfy:", ftfy.__version__)

'''
pip install karna

Hugging Face Hub version: 0.10.1
Python version: 3.11.5 | packaged by Anaconda, Inc. | (main, Sep 11 2023, 13:26:23) [MSC v.1916 64 bit (AMD64)]
PyTorch version: 2.1.2+cu118
Transformers version: 4.23.1
Diffusers version: 0.11.1
datasets: 2.12.0
accelerate: 0.15.0
ftfy: 6.3.1
'''