import os, sys
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

import transformers
from transformers import GPT2Tokenizer # , GPT2LMHeadModel,

from VideoGPT import VideoGPT
from VideoData import loadData, getlabels
from TextVideoDataset import TextVideoDataset
from mingpt.utils import setup_logging,set_seed, CfgNode as CN


def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/videogpt'
    C.restore = True
    # data
    # C.data = TextVideoDataset.get_default_config()
    C.batch_size = 1
    C.num_workers  = 1
    C.max_iters = 30000
    # model
    C.model = VideoGPT.get_default_config()
    C.model.model_type = 'gpt2'
    C.model.weight_decay = 0.1
    # trainer
    C.model.learning_rate = 2e-4 # the model we're using is so small that we can go a bit faster
    C.model.betas = (0.9, 0.95)
    C.model.noise_schedule = list(np.linspace(0, 1, C.max_iters))
    C.grad_norm_clip = 1

    return C

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    #
    tokenizer = GPT2Tokenizer.from_pretrained(config.model.model_type)
    # construct the training dataset
    datapath = "/home/gangchen/Downloads/project/datasets/UCF-101"
    video = loadData(datapath, width=32, height=32) # don't worry we won't run out of file handles
    name2label = getlabels(video)
    train_dataset = TextVideoDataset(tokenizer, video, name2label, transform=None)

    # construct the model
    config.model.vocab_size = tokenizer.vocab_size
    config.model.block_size = 1024
    config.model.classes = len(name2label)
    model = VideoGPT(config.model)
    
    # checkpoint here
    ckpt_path = os.path.join(config.system.work_dir, "model.pt")
    config.restore = False
    if config.restore and os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path))

    optimizer = model.configure_optimizers(config.model)
    # setup the dataloader
    train_loader = DataLoader(
        train_dataset,
        sampler=torch.utils.data.RandomSampler(train_dataset, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    device = torch.device("cuda")
    model.to(device)
    model.train()
    iter_num = 0
    iter_time = time.time()
    # data_iter = iter(train_loader)
    num_epochs = 1000
    beta = 0
    for epoch in range(num_epochs):
        for batch in train_loader:

            for key, value in batch.items():
                if key=='data': continue
                batch[key] = batch[key].to(device)
        
            x, y, label = batch['query'], batch['answer'], batch['label']
            y = (y - 127.5)/127.5 #normalize it # y/255.0 #
            # forward the model
            warmup = (epoch<50) and ( not config.restore)
            
            if iter_num >=config.max_iters-1:
                beta = config.model.noise_schedule[config.max_iters-1]
            else:
                beta = config.model.noise_schedule[iter_num]
            loss, logits, rec_imgs = model(x, y, label, beta, warmup)

            # backprop and update the parameters
            model.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            optimizer.step()


            iter_num += 1
            tnow = time.time()
            iter_dt = tnow - iter_time
            iter_time = tnow

            # save model
            if iter_num%500==0:
                ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                torch.save(model.state_dict(), ckpt_path)
                print(loss)
        # termination conditions
        if config.max_iters is not None and iter_num >= config.max_iters:
            break

        