
# text2video GPT

![text2video](text2video.pdf)

A PyTorch implementation of text 2 video based on transformers. In order to generate videos, we need encoder and decoder to handle frames in the same hidden space as text. In addition, the transformer will handle the temporal & sequential relationship between frames.


### Usage

Here's how you'd instantiate a GPT-2 (124M param version):

```python
from VideoVAEGPT import VideoVAEGPT as VideoGPT
from VideoData import loadData, getlabels
from TextVideoDataset import TextVideoDataset
config.model.vocab_size = tokenizer.vocab_size
config.model.block_size = 1024
config.model.classes = len(name2label)
model = VideoGPT(config.model)
```

And here's how you'd train it:

```python
python train_text2video.py
```

A simple version without encoder/decoder
```python
python train_videogpt.py
```

And the dataset is the (text, video) pair, which are from UCF101 dataset

### Library Dependences
pytorch
```
pip install pytorch
```
minGPT

If you want to `import mingpt` into your project:

```
git clone https://github.com/karpathy/minGPT.git
cd minGPT
pip install -e .
```


### References

Code:

- [minGPT](https://github.com/karpathy/minGPT.git)
- [openai/image-gpt](https://github.com/openai/image-gpt) classification part
- [huggingface/transformers](https://github.com/huggingface/transformers)

Papers + some implementation notes:

### License

MIT
