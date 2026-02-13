from argparse import ArgumentParser
import torch
from openai import OpenAI
from guardbench import benchmark
from transformers import AutoProcessor

MAX_GEN_TOKENS = 5

def moderate(
    conversations: list[list[dict[str, str]]],
    processor: AutoProcessor,
    client: OpenAI,
    model: str,
    safe_keyword: str,
    unsafe_keyword: str,
    top_logprobs: int | None = None,
    logit_bias_strength: float = 0.0,
) -> list[float]:
    # Llama Guard does not support conversation starting with the assistant
    # Therefore, we drop the first utterance if it is from the assistant
    for i, x in enumerate(conversations):
        if x[0]["role"] == "assistant":
            conversations[i] = x[1:]

    assert len(conversations) == 1, "Batch-size must be 1 for now; batching not yet implemented"

    # convert from {"rule": "user", "content": "..."} to multimodal format: {"role": "user", "content": [{"type": "text", "text": "..."}]}
    # right now we assume all inputs are text-only
    mm_conversations = []
    for conv in conversations:
        mm_conv = []
        for turn in conv:
            turn['content'] = [{'type': 'text', 'text': turn['content']}]
            mm_conv.append(turn)
        mm_conversations.append(mm_conv)
    
    # default behavior: request all tokens
    if top_logprobs is None:
        top_logprobs = processor.tokenizer.vocab_size

    logit_bias = None
    if logit_bias_strength and top_logprobs < processor.tokenizer.vocab_size:
        safe_ids = processor(text=safe_keyword, add_special_tokens=False)['input_ids'][0]
        unsafe_ids = processor(text=unsafe_keyword, add_special_tokens=False)['input_ids'][0]
        assert len(safe_ids) == 1, f"{safe_keyword=} must be a single token but got ids={safe_ids}"
        assert len(unsafe_ids) == 1, f"{unsafe_keyword=} must be a single token but got ids={unsafe_ids}"
        # OpenAI API expects token ids as *strings* in the logit_bias map.
        # Adding the SAME bias to both tokens preserves their relative logit difference,
        # but makes it very likely they appear in the returned top_logprobs list.
        logit_bias = {str(safe_ids[0]): float(logit_bias_strength), str(unsafe_ids[0]): float(logit_bias_strength)}

    resp = client.chat.completions.create(
        model=model,
        messages=mm_conversations[0],
        max_tokens=MAX_GEN_TOKENS,
        temperature=0.0,
        logprobs=True,
        top_logprobs=top_logprobs,
        logit_bias=logit_bias,
        # NOTE: prompts are good for this model, I've manually verified with:
        # extra_body={
        #     "echo": True,
        #     "return_token_ids": True,
        #     "prompt_logprobs": 5,
        # }
    )
    
    assert len(resp.choices) == 1, "Model should produce only one generation per prompt"
    resp = resp.choices[0]  

    assert len(resp.logprobs.content) <= MAX_GEN_TOKENS, f"Model should produce at most max_tokens={MAX_GEN_TOKENS} but got {len(resp.logprobs.content)}"
    safe_or_unsafe_idx = None
    for i, generated_token in enumerate(resp.logprobs.content):
        if generated_token.token in [safe_keyword, unsafe_keyword]:
            safe_or_unsafe_idx = i
            break
    assert safe_or_unsafe_idx is not None, f"Model should produce {safe_keyword} or {unsafe_keyword} in the generation but got len(content)={len(resp.logprobs.content)}, content={resp.logprobs.content}"
    generated_token = resp.logprobs.content[safe_or_unsafe_idx] 
    
    if top_logprobs == processor.tokenizer.vocab_size:
        assert len(generated_token.top_logprobs) == processor.tokenizer.vocab_size, "Model should produce logits for all tokens in the vocabulary"

    # .logprob below corresponds to raw logits because the model should be served with 
    # vllm serve <mdl_name> --logprobs-mode raw_logits
    safe_logit = None
    unsafe_logit = None
    for item in generated_token.top_logprobs:
        if item.token == safe_keyword:
            safe_logit = item.logprob
        if item.token == unsafe_keyword:
            unsafe_logit = item.logprob
        
        if safe_logit is not None and unsafe_logit is not None:
            break

    assert safe_logit is not None and unsafe_logit is not None, f"Model should produce logits for {safe_keyword} and {unsafe_keyword}"
    unsafe_prob = torch.softmax(torch.tensor([safe_logit, unsafe_logit]), dim=-1)[1].item()
    return [unsafe_prob]


def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--vllm_port", type=int, default=8000)
    parser.add_argument("--vllm_api_key", type=str, default="EMPTY")
    parser.add_argument("--safe_keyword", type=str, default="safe")
    parser.add_argument("--unsafe_keyword", type=str, default="unsafe")
    parser.add_argument("--top_logprobs", type=int, default=-1, help="If -1, requests full vocab (slow). Try 2 with --logit_bias_strength>0 for speed.")
    parser.add_argument("--logit_bias_strength", type=float, default=0.0, help="If >0 and top_logprobs is small, biases safe/unsafe equally so they appear in top-k.")
    args = parser.parse_args()
    #     [
    #         "--model", "/home/eldarkurtic/meta-llama/Llama-Guard-4-12B",
    #         "--datasets", "advbench_behaviors", 
    #         "--output_dir", "outputs_eldar_sweep",
    #         "--top_logprobs", "10",
    #         "--logit_bias_strength", "0.0",
    #     ]
    # )

    model = args.model
    datasets = args.datasets
    if len(datasets) == 1 and datasets[0] == "all":
        datasets = "all"
    batch_size = args.batch_size
    output_dir = args.output_dir
    vllm_port = args.vllm_port
    vllm_api_key = args.vllm_api_key
    safe_keyword = args.safe_keyword
    unsafe_keyword = args.unsafe_keyword
    top_logprobs = None if args.top_logprobs == -1 else args.top_logprobs
    logit_bias_strength = args.logit_bias_strength

    processor = AutoProcessor.from_pretrained(model)
    client = OpenAI(base_url=f"http://localhost:{vllm_port}/v1", api_key=vllm_api_key)

    benchmark(
        moderate=moderate,
        model_name=model,
        batch_size=batch_size,
        datasets=datasets,
        out_dir=output_dir,
        # Moderate kwargs - the following arguments are given as input to `moderate`
        client=client,
        processor=processor,
        model=model,
        safe_keyword=safe_keyword,
        unsafe_keyword=unsafe_keyword,
        top_logprobs=processor.tokenizer.vocab_size if top_logprobs is None else top_logprobs,
        logit_bias_strength=logit_bias_strength,
    )

if __name__ == "__main__":
    main()