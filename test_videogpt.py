import os
import torch
from torchvision.io import write_video
import transformers
from transformers import GPT2Tokenizer # , GPT2LMHeadModel,
#import matplotlib.pyplot as plt
from VideoGPT import VideoGPT
from train_videogpt import get_config
use_mingpt = True
model_type = "gpt2"

config = get_config()
tokenizer = GPT2Tokenizer.from_pretrained(config.model.model_type)
# construct the model
config.model.vocab_size = tokenizer.vocab_size
config.model.block_size = tokenizer.model_max_length
config.model.classes = 54#len(name2label)
model = VideoGPT(config.model)
ckpt_path = os.path.join(config.system.work_dir, "model.pt")

model.load_state_dict(torch.load(ckpt_path))
device = torch.device("cuda")
model.to(device)
model.eval()

prompt = "a girl with white clothes is doing floor exercise from right to left in gymnastics"# "a women is practicing floor gymnastics"
encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
x = encoded_input['input_ids']
# we'll process all desired num_samples in a batch, so expand out the batch dim
# x = x.expand(num_samples, -1)
# forward the model `steps` times to get samples, in a batch
y = model.generate(x, max_new_tokens=200)
write_video("results3.avi", ((y[0,:,:,:,:].cpu()+1)/2.0*255), fps=24)




