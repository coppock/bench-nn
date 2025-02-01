import random

from common import *

parser = ArgumentParser()
parser.add_argument('engine')
parser.add_argument('tokenizer')
parser.add_argument('-l', '--input-length', default=64, type=int)
parser.add_argument('-t', '--max-tokens', type=int)
args = parser.parse_args()

# This should happen before importation of tensorrt_llm in order to handle
# stdout abuse.
iter = Iterator(args.iteration_count)

from tensorrt_llm.llmapi import LLM, KvCacheConfig, SamplingParams

kv_cache_config = KvCacheConfig(max_tokens=args.max_tokens)
llm = LLM(args.engine, args.tokenizer, kv_cache_config=kv_cache_config)
sampling_params = SamplingParams(max_tokens=1)

for _ in iter:
    prompt = [random.randint(0, 2**31 - 1) for _ in range(args.input_length)]
    llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
