import os
import torch
from torchvision.io import write_video
import transformers
from transformers import GPT2Tokenizer # , GPT2LMHeadModel,
#import matplotlib.pyplot as plt
from VideoUnetGPT import VideoUnetGPT as VideoGPT
from Unet import Unet
from train_text2video import get_config
use_mingpt = True
model_type = "gpt2"

config = get_config()
tokenizer = GPT2Tokenizer.from_pretrained(config.model.model_type)
# construct the model
config.model.vocab_size = tokenizer.vocab_size
config.model.block_size = tokenizer.model_max_length
config.model.classes = 54#len(name2label)

unet = Unet(
        dim = 16,
        cond_dim = 512,
        dim_mults = (1, 2,4,8),
        num_resnet_blocks = 2,
        layer_attns = (False, False,False,False),
        layer_cross_attns = (False,False,False, False)
    )
model = VideoGPT(config.model, unet)
ckpt_path = os.path.join(config.system.work_dir, "model.pt")

model.load_state_dict(torch.load(ckpt_path))
device = torch.device("cuda")
model.to(device)
model.eval()

prompt = "a girl with white clothes"# is doing floor gymnastics exercise from right to left"# "a women is practicing floor gymnastics"
#prompt = "two men wearing fencing suit practicing with sword against each other"
# prompt = "Stir the flour and water combination into source on the pan"
#prompt= "one guy with white uniform is throwing a frisbee"
encoded_input = tokenizer(prompt, return_tensors='pt').to(device)
x = encoded_input['input_ids']
# we'll process all desired num_samples in a batch, so expand out the batch dim
# x = x.expand(num_samples, -1)
# forward the model `steps` times to get samples, in a batch
y, idx = model.generate(x,coldstart=10, max_new_tokens=600)
generated_text = tokenizer.decode(idx.cpu().squeeze())
print(generated_text)
write_video("results3.avi", ((y[0,:,:,:,:].cpu()+1)/2.0*255), fps=24)




