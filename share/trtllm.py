import itertools
import os
import signal
import sys
import time
from argparse import ArgumentParser

stdout = os.fdopen(os.dup(sys.stdout.fileno()), 'w')
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
# For some reason, this import must be outside the main function; otherwise,
# the program will hang after completion.
from tensorrt_llm.llmapi import LLM, SamplingParams, KvCacheConfig


# For some reason, we can't just put the main body directly in the if block;
# otherwise, the program will hang after completion.
def main():
    done = False

    def handler(self, *_):
        done = True
    for signum in signal.SIGTERM, signal.SIGINT:
        signal.signal(signum, handler)

    parser = ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('-c', '--max-tokens', default=8192, type=int)
    parser.add_argument('-l', '--input-length', default=64, type=int)
    parser.add_argument('-m', '--iteration-count', type=int)
    parser.add_argument('-n', '--batch-size', default=1, type=int)
    parser.add_argument('-t', '--tokenizer')
    parser.add_argument('-v', '--use-vllm', action='store_true')
    args = parser.parse_args()

    llm = LLM(model=args.model, tokenizer=args.tokenizer,
              kv_cache_config=KvCacheConfig(max_tokens=args.max_tokens))
    prompt = args.batch_size * [[0 for _ in range(args.input_length)]]
    sampling_params = SamplingParams(max_tokens=1)
    for _ in (range(args.iteration_count) if args.iteration_count
              else itertools.count()):
        if done:
            break
        t_i = time.time()
        llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
        t_f = time.time()
        print(t_f, t_f - t_i, file=stdout, flush=True)


if __name__ == '__main__':
    main()
