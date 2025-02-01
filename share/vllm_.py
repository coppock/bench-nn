import random

from common import *

parser = ArgumentParser()
parser.add_argument('model')
# parser.add_argument('tokenizer')
parser.add_argument('-l', '--input-length', default=64, type=int)
args = parser.parse_args()

# This should happen before importation of tensorrt_llm in order to handle
# stdout abuse.
iter = Iterator(args.iteration_count)

from vllm import LLM, SamplingParams, TokensPrompt

llm = LLM(model=args.model, enforce_eager=True)
sampling_params = SamplingParams(max_tokens=1)

for _ in iter:
    prompt_token_ids = [random.randint(0, llm.get_tokenizer().vocab_size - 1)
                        for _ in range(args.input_length)]
    prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)
    llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
