# Lumi portability gate v0.1

Status: **in progress; full-scale training is not approved**

## Current result

| Target | Status | Evidence / remaining work |
|---|---|---|
| Hugging Face | PASS (tiny checkpoint) | Export includes model code and `auto_map`; isolated Python loads `AutoModel` and `AutoModelForCausalLM`, preserves logits, and generates |
| vLLM | CODE READY; runtime test pending | Base model now routes attention through the Transformers attention interface and forwards backend kwargs; test on Linux/CUDA with `model_impl="transformers"` is still required |
| MLX-LM | DESIGN REVIEWED; adapter pending | MLX's Llama implementation supports an independent `head_dim`, including Lumi's wider query projection; Lumi still needs a custom model adapter and real converted-checkpoint test on the M5 Mac |
| GGUF / llama.cpp | NOT IMPLEMENTED | A custom `LumiForCausalLM` converter/architecture mapping and Metal generation test are required |

Passing Hugging Face does not imply that MLX, GGUF, or vLLM passed. The 64x64
attention layout remains a candidate until all three runtime gates execute with
the same tiny checkpoint and produce numerically sensible logits.

## Why the Hugging Face test is isolated

Importing the local `lumi` package registers its classes in the current Python
process, which can hide an incomplete checkpoint. The portability test exports
the code and weights, launches Python with isolated mode, imports only
Transformers, and loads with `trust_remote_code=True`. It also checks the base
`AutoModel` mapping required by vLLM.

Run it with:

```bash
python scripts/check_isolated_hf.py
```

## vLLM runtime gate

Run only after a trained tokenizer and tiny real-data checkpoint exist:

1. start a low-cost Linux/CUDA instance;
2. install the repository and a pinned vLLM version;
3. load the local checkpoint with the Transformers backend and remote code
   explicitly trusted;
4. verify the backend class is `Transformers...`, not ordinary eager PyTorch;
5. compare prompt logits/generation with the Hugging Face reference;
6. exercise one- and multi-request serving through the OpenAI-compatible API;
7. record GPU memory, prefill/decode speed, and exact package versions.

Do not turn on the full H100 cluster for this. The first runtime gate should use
minutes, not days.

## MLX and GGUF gates

Both local formats need the frozen tokenizer files. The next implementation
pass will add a tiny MLX adapter and GGUF conversion mapping, then compare
greedy output on the M5 Mac. If the wider 64x64 attention projection causes a
backend implementation problem, compare the existing 24x128 fallback at the
same pilot scale before changing the 4B architecture.

Official implementation references:

- [Hugging Face custom models](https://huggingface.co/docs/transformers/custom_models)
- [vLLM Transformers backend and custom models](https://docs.vllm.ai/en/latest/models/supported_models/)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [llama.cpp model-architecture guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md)
