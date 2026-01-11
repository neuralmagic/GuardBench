from argparse import ArgumentParser
import statistics
import time
import torch
from openai import OpenAI
from guardbench import benchmark
from transformers import AutoTokenizer


def moderate(
    conversations: list[list[dict[str, str]]],
    tokenizer: AutoTokenizer,
    client: OpenAI,
    model: str,
    safe_keyword: str,
    unsafe_keyword: str,
    *,
    top_logprobs: int | None = None,
    logit_bias_strength: float = 0.0,
    return_timings: bool = False,
) -> list[float]:
    # Llama Guard does not support conversation starting with the assistant
    # Therefore, we drop the first utterance if it is from the assistant
    t0_total = time.perf_counter()
    for i, x in enumerate(conversations):
        if x[0]["role"] == "assistant":
            conversations[i] = x[1:]

    assert len(conversations) == 1, "Batch-size must be 1 for now; batching not yet implemented"

    # Default behavior (backwards compatible): request all tokens.
    # Faster option: use equal logit_bias on safe/unsafe so both tokens land in top-k, then set top_logprobs=2.
    if top_logprobs is None:
        top_logprobs = tokenizer.vocab_size

    logit_bias = None
    if logit_bias_strength and top_logprobs < tokenizer.vocab_size:
        safe_ids = tokenizer.encode(safe_keyword, add_special_tokens=False)
        unsafe_ids = tokenizer.encode(unsafe_keyword, add_special_tokens=False)
        assert len(safe_ids) == 1, f"{safe_keyword=} must be a single token but got ids={safe_ids}"
        assert len(unsafe_ids) == 1, f"{unsafe_keyword=} must be a single token but got ids={unsafe_ids}"
        # OpenAI API expects token ids as *strings* in the logit_bias map.
        # Adding the SAME bias to both tokens preserves their relative logit difference,
        # but makes it very likely they appear in the returned top_logprobs list.
        logit_bias = {str(safe_ids[0]): float(logit_bias_strength), str(unsafe_ids[0]): float(logit_bias_strength)}

    t0_rpc = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=conversations[0],
        # {"role": "system", "content": "You are helpful."},
        # {"role": "user", "content": "Write one short sentence about Vienna."},
        max_tokens=1,
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
    t1_rpc = time.perf_counter()
    
    assert len(resp.choices) == 1, "Model should produce only one generation per prompt"
    resp = resp.choices[0]  

    assert len(resp.logprobs.content) == 1, "Model should produce only one token due to max_tokens=1 above"
    generated_token = resp.logprobs.content[0] # TODO: currently we don't check if the generated token is safe/unsafe_keyword
    
    if top_logprobs == tokenizer.vocab_size:
        assert len(generated_token.top_logprobs) == tokenizer.vocab_size, "Model should produce logits for all tokens in the vocabulary"

    # .logprob below corresponds to raw logits because the model should be served with 
    # vllm serve <mdl_name> --logprobs-mode raw_logits
    t0_parse = time.perf_counter()
    safe_logit = None
    unsafe_logit = None
    for item in generated_token.top_logprobs:
        if item.token == safe_keyword:
            safe_logit = item.logprob
        if item.token == unsafe_keyword:
            unsafe_logit = item.logprob
        
        if safe_logit is not None and unsafe_logit is not None:
            break
    t1_parse = time.perf_counter()

    assert safe_logit is not None and unsafe_logit is not None, f"Model should produce logits for {safe_keyword} and {unsafe_keyword}"
    unsafe_prob = torch.softmax(torch.tensor([safe_logit, unsafe_logit]), dim=-1)[1].item()
    out = [unsafe_prob]

    if return_timings:
        # Append timings as a second return item (kept out of the core benchmark path unless explicitly enabled).
        out.append(
            {
                "rpc_s": t1_rpc - t0_rpc,
                "parse_s": t1_parse - t0_parse,
                "total_s": time.perf_counter() - t0_total,
                "top_logprobs": top_logprobs,
                "used_logit_bias": bool(logit_bias),
            }
        )
    return out


def microbench_moderate(
    *,
    iters: int,
    warmup: int,
    conversations: list[list[dict[str, str]]],
    tokenizer: AutoTokenizer,
    client: OpenAI,
    model: str,
    safe_keyword: str,
    unsafe_keyword: str,
    top_logprobs: int,
    logit_bias_strength: float,
) -> None:
    # Warmup (lets vLLM load / cache; also warms python/json code paths)
    for _ in range(warmup):
        _ = moderate(
            conversations=[c.copy() for c in conversations],
            tokenizer=tokenizer,
            client=client,
            model=model,
            safe_keyword=safe_keyword,
            unsafe_keyword=unsafe_keyword,
            top_logprobs=top_logprobs,
            logit_bias_strength=logit_bias_strength,
        )

    totals: list[float] = []
    rpcs: list[float] = []
    parses: list[float] = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = moderate(
            conversations=[c.copy() for c in conversations],
            tokenizer=tokenizer,
            client=client,
            model=model,
            safe_keyword=safe_keyword,
            unsafe_keyword=unsafe_keyword,
            top_logprobs=top_logprobs,
            logit_bias_strength=logit_bias_strength,
            return_timings=True,
        )
        t1 = time.perf_counter()
        timings = out[1]
        totals.append(t1 - t0)
        rpcs.append(float(timings["rpc_s"]))
        parses.append(float(timings["parse_s"]))

    def pct(xs: list[float], p: float) -> float:
        xs_sorted = sorted(xs)
        if not xs_sorted:
            return 0.0
        k = int(round((len(xs_sorted) - 1) * p))
        return xs_sorted[k]

    print("microbench_moderate results")
    print(f"  iters={iters} warmup={warmup}")
    print(f"  top_logprobs={top_logprobs} logit_bias_strength={logit_bias_strength}")
    print(f"  total_s: mean={statistics.mean(totals):.4f} p50={pct(totals, 0.50):.4f} p90={pct(totals, 0.90):.4f}")
    print(f"  rpc_s:   mean={statistics.mean(rpcs):.4f} p50={pct(rpcs, 0.50):.4f} p90={pct(rpcs, 0.90):.4f}")
    print(f"  parse_s: mean={statistics.mean(parses):.4f} p50={pct(parses, 0.50):.4f} p90={pct(parses, 0.90):.4f}")

def main():
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vllm_port", type=int, default=8000)
    parser.add_argument("--vllm_api_key", type=str, default="EMPTY")
    parser.add_argument("--safe_keyword", type=str, default="safe")
    parser.add_argument("--unsafe_keyword", type=str, default="unsafe")
    parser.add_argument("--top_logprobs", type=int, default=-1, help="If -1, requests full vocab (slow). Try 2 with --logit_bias_strength>0 for speed.")
    parser.add_argument("--logit_bias_strength", type=float, default=0.0, help="If >0 and top_logprobs is small, biases safe/unsafe equally so they appear in top-k.")
    parser.add_argument("--microbench", action="store_true", help="Run a local timing micro-benchmark of moderate() and exit.")
    parser.add_argument("--microbench_iters", type=int, default=20)
    parser.add_argument("--microbench_warmup", type=int, default=5)
    args = parser.parse_args()
    #     [
    #         "--model", "meta-llama/LlamaGuard-7b",
    #         "--datasets", "all", 
    #         "--batch_size", "1",
    #         "--output_dir", "outputs_eldar_sweep",
    #         "--vllm_port", "8000",
    #         "--vllm_api_key", "EMPTY",
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

    tokenizer = AutoTokenizer.from_pretrained(model)

    # Initialize vLLM client
    client = OpenAI(base_url=f"http://localhost:{vllm_port}/v1", api_key=vllm_api_key)

    if args.microbench:
        microbench_moderate(
            iters=args.microbench_iters,
            warmup=args.microbench_warmup,
            conversations=[[{"role": "user", "content": "Hello, can you help me with something?"}]],
            tokenizer=tokenizer,
            client=client,
            model=model,
            safe_keyword=safe_keyword,
            unsafe_keyword=unsafe_keyword,
            top_logprobs=tokenizer.vocab_size if top_logprobs is None else top_logprobs,
            logit_bias_strength=logit_bias_strength,
        )
        return

    benchmark(
        moderate=moderate,
        model_name=model,
        batch_size=batch_size,
        datasets=datasets,
        out_dir=output_dir,
        # datasets=["advbench_behaviors", "advbench_strings"],
        # Moderate kwargs - the following arguments are given as input to `moderate`
        client=client,
        tokenizer=tokenizer,
        model=model,
        safe_keyword=safe_keyword,
        unsafe_keyword=unsafe_keyword,
        top_logprobs=tokenizer.vocab_size if top_logprobs is None else top_logprobs,
        logit_bias_strength=logit_bias_strength,
    )

if __name__ == "__main__":
    main()