"""Exact prompt formats (App. A.2, Table 5) and chat templating.

All prompts are given as chat messages and rendered with each model's own
tokenizer chat template. Three prompt families are used:

* pair representation (Sec. 3.1): ``Q: {question} A: {answer}`` as the user turn,
  preceded by a system message chosen by ``template_id``: each pair is assigned
  one of templates 0-2; the 14B template control (App. D.1) renders a fixed
  200-pair subset under all five;
* correctness judgment (Sec. 3.3): the pair text followed by a Yes/No question,
  with the fixed system message ``You are a helpful assistant.``;
* five-shot short-answer generation (App. D.3–D.4, Table 5).
"""
from __future__ import annotations

from typing import Any

# System messages of the pair representation, indexed by template_id. Pairs are
# assigned templates 0-2 (``data.assign_split_and_templates``); templates 3-4 are
# used only by the 14B template control (App. D.1, Table 21).
TEMPLATE_SYSTEM_MESSAGES: tuple[str, ...] = (
    "You are a helpful assistant.",
    "You are a helpful assistant. Answer the user's question directly.",
    "You are a knowledgeable AI assistant that helps users with various tasks.",
    "Provide concise and factual answers.",
    "Answer carefully using your knowledge; do not add unnecessary explanation.",
)
CANONICAL_TEMPLATE = 0

JUDGMENT_SYSTEM_PROMPT = "You are a helpful assistant."
JUDGMENT_QUESTION = "Is the above answer correct? Respond with only 'Yes' or 'No'."

# Five-shot block for short-answer generation (App. A.2). It ends with a newline.
FEWSHOT_BLOCK = """Answer each question with only the short answer. Do not explain your reasoning.
Q: What is the capital of France?
A: Paris
Q: Who wrote Hamlet?
A: William Shakespeare
Q: What planet is known as the Red Planet?
A: Mars
Q: What is the chemical symbol for gold?
A: Au
Q: In which continent is Brazil?
A: South America
"""

# Table 4: templates 0-1 use instruction S0, template 2 uses S2.
SHORT_ANSWER_SYSTEM_PREFIX = {
    0: "You are a helpful assistant. ",
    1: "You are a helpful assistant. ",
    2: "You are a knowledgeable AI assistant. ",
}

# Models whose short-answer prompt has no system message (Table 5): the five-shot
# block and the question are concatenated into one user message.
NO_SYSTEM_SHORT_ANSWER = ("mistral-7b",)


def pair_text(question: str, answer: str) -> str:
    return f"Q: {question} A: {answer}"


def pair_messages(question: str, answer: str, template_id: int = CANONICAL_TEMPLATE) -> list[dict[str, str]]:
    """Pair representation input x = (q, a) (Sec. 3.1, App. A.2).

    The same system + user messages are used for every model, including
    Mistral-7B and Ministral-3-14B, whose chat templates fold the system message
    into the first user turn.
    """
    return [
        {"role": "system", "content": TEMPLATE_SYSTEM_MESSAGES[int(template_id)]},
        {"role": "user", "content": pair_text(question, answer)},
    ]


def judgment_messages(question: str, answer: str, system: bool = True) -> list[dict[str, str]]:
    """Yes/No correctness-judgment prompt used for the steering readout (Sec. 3.3).

    Pass ``system=False`` to run the prompt ablation without a system message.
    """
    user = {"role": "user", "content": f"{pair_text(question, answer)}\n\n{JUDGMENT_QUESTION}"}
    if not system:
        return [user]
    return [{"role": "system", "content": JUDGMENT_SYSTEM_PROMPT}, user]


def short_answer_messages(question: str, template_id: int = 0,
                          model_key: str = "llama-3.1-8b") -> list[dict[str, str]]:
    """Five-shot short-answer generation prompt (App. A.2, Tables 4-5).

    Llama and Qwen: the five-shot block is the system message and the question is
    a separate user message. Mistral: no system message; block and question are
    one user message.
    """
    user = f"Q: {question} A:"
    if model_key in NO_SYSTEM_SHORT_ANSWER:
        return [{"role": "user", "content": FEWSHOT_BLOCK + user}]
    prefix = SHORT_ANSWER_SYSTEM_PREFIX.get(int(template_id), SHORT_ANSWER_SYSTEM_PREFIX[0])
    return [{"role": "system", "content": prefix + FEWSHOT_BLOCK},
            {"role": "user", "content": user}]


def is_mistral_common(tokenizer: Any) -> bool:
    """True for the ``mistral-common`` tokenizer backend used by Ministral-3-14B."""
    return type(tokenizer).__name__ == "MistralCommonBackend"


def render(tokenizer: Any, model_key: str, messages: list[dict[str, str]],
           add_generation_prompt: bool = False) -> str:
    """Render chat messages to text with the model's chat template.

    The rendered text is then tokenized with ``tokenizer(text)`` (default special
    tokens), exactly as in the original runs; for templates that already emit a
    BOS token this yields the same token sequence the original caches used.
    """
    if is_mistral_common(tokenizer):
        raise ValueError(f"{model_key}: use encode() for the mistral-common tokenizer")
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=add_generation_prompt)


def encode(tokenizer: Any, model_key: str, messages: list[dict[str, str]],
           add_generation_prompt: bool = False) -> list[int]:
    """Token ids of a chat prompt.

    HF tokenizers: ``tokenizer(render(...))["input_ids"]``. Ministral-3-14B
    (mistral-common backend): the backend's own chat-template tokenization,
    which is the canonical encoding for that model (App. B.1).
    """
    if is_mistral_common(tokenizer):
        enc = tokenizer.apply_chat_template(messages, tokenize=True,
                                            add_generation_prompt=add_generation_prompt,
                                            continue_final_message=False,
                                            return_tensors="pt", return_dict=True)
        ids = enc["input_ids"]
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise RuntimeError(f"unexpected input shape {tuple(ids.shape)}")
        return ids[0].tolist()
    return list(tokenizer(render(tokenizer, model_key, messages, add_generation_prompt))["input_ids"])
