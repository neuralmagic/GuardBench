import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "10000"
os.environ["HF_HUB_CACHE"] = "/home/eldarkurtic/hf_hub_cache"
os.environ["TRANSFORMERS_CACHE"] = "/home/eldarkurtic/transformers_cache"

import torch
from transformers import Llama4ForConditionalGeneration, AutoProcessor
from guardbench import benchmark

model_id = "/home/eldarkurtic/meta-llama/Llama-Guard-4-12B"
processor = AutoProcessor.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model = model.to("cuda")
model = model.eval()


def find_safe_or_unsafe_in_answer(
    sequences: torch.Tensor, # [batch_size, seq_len]
    safe_token_id: int,
    unsafe_token_id: int,
    input_len: int, # [batch_size]
    pad_token_id: int,
) -> torch.Tensor: # [batch_size]
    positions = torch.tensor([-1] * sequences.shape[0])
    for i in range(sequences.shape[0]):
        # tag all safe/unsafe tokens
        mask = (sequences[i] == safe_token_id) | (sequences[i] == unsafe_token_id) 
        # mask out input prompt (and left padding)
        mask = mask & (input_len < torch.arange(sequences.shape[1], device=mask.device))   
        # mask padding tokens
        mask = mask & (sequences[i] != pad_token_id)
        # find first safe/unsafe token
        safe_or_unsafe_positions = mask.nonzero(as_tuple=False)[0]
        if safe_or_unsafe_positions.numel() > 0:
            positions[i] = safe_or_unsafe_positions[0] - input_len
        else:
            positions[i] = -1

    assert all(positions > 0), "Could not find safe or unsafe token in model's output"
    return positions

def moderate(
    conversations: list[list[dict[str, str]]],  # MANDATORY!
    processor: AutoProcessor,
    model: Llama4ForConditionalGeneration,
    safe_token_id: int,
    unsafe_token_id: int,
) -> list[float]:
    # Llama Guard does not support conversation starting with the assistant
    # Therefore, we drop the first utterance if it is from the assistant
    for i, x in enumerate(conversations):
        if x[0]["role"] == "assistant":
            conversations[i] = x[1:]

    # convert from {"rule": "user", "content": "..."} to multimodal format: {"role": "user", "content": [{"type": "text", "text": "..."}]}
    # right now we assume all inputs are text-only
    mm_conversations = []
    for conv in conversations:
        mm_conv = []
        for turn in conv:
            turn['content'] = [{'type': 'text', 'text': turn['content']}]
            mm_conv.append(turn)
        mm_conversations.append(mm_conv)
    
    # Apply Llama Guard's chat template to each conversation
    inputs = processor.apply_chat_template(
        mm_conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,  # padding is on the left
    ).to(model.device)

    # Generate output - Here is where the input moderation happens
    output = model.generate(
        **inputs,
        max_new_tokens=5,
        output_scores=True,
        return_dict_in_generate=True,
    )

    # Take logits for the first generated token of each input sample <-- doesn't work for Llama-Guard-4-12B because the model start generation with \n\n token
    # so we need to run through output tokens and stop on the first token which is either safe or unsafe
    # This way we assume that we will filter out all answer-formatting that the model might produce before
    # generating the actual answer "safe" or "unsafe"
    input_len = inputs['input_ids'].shape[-1] # all inputs have the same length in the batch (via left-padding)
    first_safe_or_unsafe_position_in_answer = find_safe_or_unsafe_in_answer(
        output.sequences, 
        safe_token_id, 
        unsafe_token_id,
        input_len,
        pad_token_id=processor.tokenizer.pad_token_id,
    )
    assert torch.isin(output.sequences[:, input_len + first_safe_or_unsafe_position_in_answer], torch.tensor([safe_token_id, unsafe_token_id], device=output.sequences.device)).all(), "first_safe_or_unsafe_position_in_answer is not a safe or unsafe token"
    # output.scores is a tuple of "max_new_tokens" tensors of shape [batch_size, vocab_size]
    logits = torch.stack([output.scores[out_token_idx][i, [safe_token_id, unsafe_token_id]] for i, out_token_idx in enumerate(first_safe_or_unsafe_position_in_answer.tolist())], dim=0)
    # shape is [batch_size, 2] which corresponds to [safe_logit, unsafe_logit]

    # Compute "unsafe" probabilities
    return torch.softmax(logits, dim=-1)[:, 1].tolist()

safe_token_id = processor(text="safe", add_special_tokens=False)['input_ids'][0]
assert len(safe_token_id) == 1, "processor(text='safe')['input_ids'] should return a list of length 1 but got {}".format(len(safe_token_id))
safe_token_id = safe_token_id[0]

unsafe_token_id = processor(text="unsafe", add_special_tokens=False)['input_ids'][0]
assert len(unsafe_token_id) == 1, "processor(text='unsafe')['input_ids'] should return a list of length 1 but got {}".format(len(unsafe_token_id))
unsafe_token_id = unsafe_token_id[0]

benchmark(
    moderate=moderate,
    model_name="Llama Guard",
    batch_size=1,
    datasets="all",
    # datasets=["advbench_behaviors"],
    # Moderate kwargs - the following arguments are given as input to `moderate`
    processor=processor,
    model=model,
    safe_token_id=safe_token_id,
    unsafe_token_id=unsafe_token_id,
)
