from common import *

parser = ArgumentParser()
parser.add_argument('tokenizer')
parser.add_argument('engine')
parser.add_argument('-l', '--input-length', default=1, type=int)
parser.add_argument('-t', '--maximum-tokens-in-paged-kv-cache', type=int)
args = parser.parse_args()

# This should happen before importation of tensorrt_llm in order to handle
# stdout abuse.
iter = Iterator(args.iteration_count)

import torch
from tensorrt_llm.runtime import ModelRunnerCpp

from utils import read_model_name, load_tokenizer

runner = ModelRunnerCpp.from_dir(
    args.engine,
    max_tokens_in_paged_kv_cache=args.maximum_tokens_in_paged_kv_cache,
)
model_name, model_version = read_model_name(args.engine)
_, pad_id, end_id = load_tokenizer(args.tokenizer, model_name=model_name,
                                   model_version=model_version)

for _ in iter:
    batch_input_ids = torch.empty((args.batch_size, args.input_length),
                                  dtype=torch.int)
    with torch.no_grad():
        outputs = runner.generate(batch_input_ids, pad_id=pad_id,
                                  end_id=end_id)
        torch.cuda.synchronize()
