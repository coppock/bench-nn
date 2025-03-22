from argparse import ArgumentParser
import itertools
import os
import signal
import sys
import time

stdout = os.fdopen(os.dup(sys.stdout.fileno()), 'w')
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

import torch
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import ModelRunnerCpp

from run import parse_input
from utils import read_model_name, load_tokenizer

DONE = False
PROMPTS = [
    'hello',
]


def handler(*_):
    global DONE
    DONE = True


def main():
    """
    For some reason, we can't just put the main body directly in the if block;
    otherwise, the program will hang after completion.
    """
    for signum in signal.SIGTERM, signal.SIGINT:
        signal.signal(signum, handler)

    parser = ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('-c', '--max-tokens', default=8192, type=int)
    parser.add_argument('-f', '--file')
    parser.add_argument('-l', '--input-length', default=64, type=int)
    parser.add_argument('-o', '--output-length', default=1, type=int)
    parser.add_argument('-m', '--iteration-count', type=int)
    parser.add_argument('-n', '--batch-size', default=1, type=int)
    parser.add_argument('-t', '--tokenizer')
    parser.add_argument('-v', '--use-vllm', action='store_true')
    args = parser.parse_args()

    runner = ModelRunnerCpp.from_dir(
        args.model,
        max_output_len=args.output_length,
        max_tokens_in_paged_kv_cache=args.max_tokens,
    )
    model_name, model_version = read_model_name(args.model)
    logger.debug(f'Model name {model_name}, model version {model_version}')
    tokenizer, pad_id, end_id = load_tokenizer(args.tokenizer,
                                               model_name=model_name,
                                               model_version=model_version)
    file = open(args.file, 'w') if args.file else stdout
    for _ in (range(args.iteration_count) if args.iteration_count
              else itertools.count()):
        if DONE:
            break
        t_i = time.time()

        batch_input_ids = parse_input(
            tokenizer,
            itertools.islice(itertools.cycle([' '.join([p]*args.input_length) for p in PROMPTS]), args.batch_size),
            pad_id=pad_id,
            model_name=model_name,
            model_version=model_version,
        )
        logger.debug(repr(batch_input_ids))
        with torch.no_grad():
            runner.generate(
                batch_input_ids,
                max_new_tokens=args.output_length,
                end_id=end_id,
                pad_id=pad_id,
                output_sequence_lengths=True,
                return_dict=True,
            )
            torch.cuda.synchronize()

        t_f = time.time()
        print(t_f, t_f - t_i, file=file, flush=True)


if __name__ == '__main__':
    main()
