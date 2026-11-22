"""Tests for probedir.prompts and the answer-token alignment in probedir.extract."""
import pytest

from probedir import extract, prompts

tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }} <|end|> {% endfor %}"
            "{% if add_generation_prompt %}<|assistant|> {% endif %}")


def tiny_tokenizer(texts):
    """Whitespace word-level fast tokenizer with a simple chat template (offline)."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {"[UNK]": 0}
    for t in texts:
        for w in t.replace("\n", " ").split():
            vocab.setdefault(w, len(vocab))
    for w in ("<|system|>", "<|user|>", "<|end|>", "<|assistant|>"):
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]")
    fast.chat_template = TEMPLATE
    return fast


def test_pair_messages_templates():
    m = prompts.pair_messages("Which city?", "Ottawa", 2)
    assert m[0]["content"] == prompts.TEMPLATE_SYSTEM_MESSAGES[2]
    assert m[1]["content"] == "Q: Which city? A: Ottawa"


def test_judgment_messages():
    m = prompts.judgment_messages("Which city?", "Toronto")
    assert m[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert m[1]["content"].endswith("Is the above answer correct? Respond with only 'Yes' or 'No'.")
    assert len(prompts.judgment_messages("q", "a", system=False)) == 1


def test_short_answer_messages_model_specific():
    llama = prompts.short_answer_messages("Q?", 0, "llama-3.1-8b")
    mistral = prompts.short_answer_messages("Q?", 0, "mistral-7b")
    assert llama[0]["role"] == "system" and llama[0]["content"].endswith("A: South America\n")
    assert len(mistral) == 1 and mistral[0]["content"].endswith("Q: Q? A:")
    assert prompts.short_answer_messages("Q?", 2)[0]["content"].startswith("You are a knowledgeable AI assistant. ")


def test_answer_token_index_points_at_last_answer_token():
    q, a = "Which city is the capital of Canada?", "Ottawa City"
    tok = tiny_tokenizer([prompts.TEMPLATE_SYSTEM_MESSAGES[0], f"Q: {q} A: {a}"])
    text = prompts.render(tok, "test", prompts.pair_messages(q, a, 0))
    ids = tok(text)["input_ids"]
    pos = extract.answer_token_index(tok, "test", text, a)
    assert tok.convert_ids_to_tokens(ids[pos]) == "City"
    assert tok.convert_ids_to_tokens(ids[pos + 1]) == "<|end|>"


def test_ministral_span_by_common_prefix_suffix():
    class MistralCommonBackend:  # stand-in with the mistral-common interface used by encode()
        all_special_ids = [0, 99]

        def __init__(self):
            self.words = {}

        def apply_chat_template(self, messages, **kw):
            import torch
            text = " ".join(m["content"] for m in messages).split()
            ids = [0] + [self.words.setdefault(w, len(self.words) + 1) for w in text] + [99]
            return {"input_ids": torch.tensor([ids])}

        def decode(self, ids, skip_special_tokens=True):
            inv = {v: k for k, v in self.words.items()}
            return " ".join(inv[i] for i in ids if i in inv)

    tok = MistralCommonBackend()
    span = extract.ministral_answer_span(tok, "Which city?", "Ottawa", 0)
    assert tok.decode([span["input_ids"][span["target_pos"]]]) == "Ottawa"
    assert span["input_ids"][span["target_pos"] + 1] == 99


def test_five_template_messages_and_alignment_rules():
    assert len(prompts.TEMPLATE_SYSTEM_MESSAGES) == 5
    assert prompts.pair_messages("q", "a")[0]["content"] == prompts.TEMPLATE_SYSTEM_MESSAGES[0]
    q, a = "Who wrote it ?", "Ann Lee"
    tok = tiny_tokenizer([*prompts.TEMPLATE_SYSTEM_MESSAGES, f"Q: {q} A: {a}"])
    for rule in ("rendered_span", "token_match"):
        ids, pos = extract.locate(tok, "test", q, a, 4, alignment=rule)
        assert tok.convert_ids_to_tokens(ids[pos]) == "Lee"
