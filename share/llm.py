from common import *

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('-l', '--input-length', default=64, type=int)
    parser.add_argument('-t', '--tokenizer')
    parser.add_argument('-v', '--use-vllm', action='store_true')
    args = parser.parse_args()

    # This should happen before importation of tensorrt_llm in order to handle
    # stdout abuse.
    iter = Iterator(args.iteration_count)

    _temp = __import__(
        'vllm' if args.use_vllm else 'tensorrt_llm.llmapi',
        globals(),
        locals(),
        ['LLM', 'SamplingParams'],
        0,
    )
    LLM, SamplingParams = _temp.LLM, _temp.SamplingParams
    if args.use_vllm: from vllm import TokensPrompt

    llm = LLM(model=args.model, tokenizer=args.tokenizer,
              **({'enforce_eager': True} if args.use_vllm else {}))
    prompt = [0 for _ in range(args.input_length)]
    if args.use_vllm: prompt = TokensPrompt(prompt_token_ids=prompt)
    sampling_params = SamplingParams(max_tokens=1)
    for _ in iter:
        llm.generate(prompt, sampling_params=sampling_params, use_tqdm=False)
