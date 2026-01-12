import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from guardbench import benchmark

model_id = "meta-llama/LlamaGuard-7b"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model = model.to("cuda")
model = model.eval()

def moderate(
    conversations: list[list[dict[str, str]]],  # MANDATORY!
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    safe_token_id: int,
    unsafe_token_id: int,
) -> list[float]:
    # Llama Guard does not support conversation starting with the assistant
    # Therefore, we drop the first utterance if it is from the assistant
    for i, x in enumerate(conversations):
        if x[0]["role"] == "assistant":
            conversations[i] = x[1:]

    # Apply Llama Guard's chat template to each conversation
    input_ids = [tokenizer.apply_chat_template(x) for x in conversations]

    # Convert input IDs to PyTorch tensor
    input_ids = torch.tensor(input_ids, device=model.device)

    # Generate output - Here is where the input moderation happens
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=5,
        output_scores=True,
        return_dict_in_generate=True,
        pad_token_id=0,
    )

    # Take logits for the first generated token of each input sample
    logits = output.scores[0][:, [safe_token_id, unsafe_token_id]]

    # Compute "unsafe" probabilities
    return torch.softmax(logits, dim=-1)[:, 1].tolist()

safe_token_id = tokenizer.encode("safe", add_special_tokens=False)
assert len(safe_token_id) == 1, "tokenizer.encode('safe') should return a list of length 1 but got {}".format(len(safe_token_id))
safe_token_id = safe_token_id[0]

unsafe_token_id = tokenizer.encode("unsafe", add_special_tokens=False)
assert len(unsafe_token_id) == 1, "tokenizer.encode('unsafe') should return a list of length 1 but got {}".format(len(unsafe_token_id))
unsafe_token_id = unsafe_token_id[0]

benchmark(
    moderate=moderate,
    model_name="Llama Guard",
    batch_size=1,
    # datasets=["aart", "advbench_behaviors", "advbench_strings", "beaver_tails_330k", "bot_adversarial_dialogue"],
    datasets="all",
    # datasets=["advbench_behaviors", "advbench_strings"],
    # Moderate kwargs - the following arguments are given as input to `moderate`
    tokenizer=tokenizer,
    model=model,
    safe_token_id=safe_token_id,
    unsafe_token_id=unsafe_token_id,
)