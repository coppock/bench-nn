from diffusers import DiffusionPipeline
import torch

from common import *

parser = ArgumentParser()
parser.add_argument('model')
args = parser.parse_args()

iter = Iterator(args.iteration_count)

pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16,
					 use_safetensors=True, variant='fp16')
pipe.to('cuda')
for _ in iter:
    image = pipe(args.batch_size*['astronaut riding a green horse']).images[0]
