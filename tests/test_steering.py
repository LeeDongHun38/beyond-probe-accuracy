import numpy as np
import pytest

from probedir import steering

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def tiny_model():
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(cfg).eval()


def _capture(model, layer):
    seen = {}

    def hook(_m, _i, out):
        seen["h"] = (out[0] if isinstance(out, tuple) else out).detach().clone()
    return seen, model.model.layers[layer].register_forward_hook(hook)


def test_steer_adds_alpha_v_at_nonpadding_positions(tiny_model):
    ids = torch.tensor([[5, 6, 7, 8], [1, 9, 10, 11]])
    mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
    v = np.zeros(16, np.float32)
    v[3] = 1.0
    seen, h = _capture(tiny_model, 1)
    with torch.no_grad():
        tiny_model(input_ids=ids, attention_mask=mask)
    base = seen["h"]
    h.remove()
    with steering.steer(tiny_model, 1, v, 2.5):
        seen, h = _capture(tiny_model, 1)    # registered after the steering hook -> sees steered output
        with torch.no_grad():
            tiny_model(input_ids=ids, attention_mask=mask)
        h.remove()
    diff = seen["h"] - base
    expected = 2.5 * torch.from_numpy(v) * mask.unsqueeze(-1)
    torch.testing.assert_close(diff, expected)


def test_steer_records_final_position_norms_and_removes_hooks(tiny_model):
    ids = torch.tensor([[1, 2, 3, 4]])
    norms = []
    with steering.steer(tiny_model, 0, None, 0.0, record_norms=norms):
        seen, h = _capture(tiny_model, 0)
        with torch.no_grad():
            tiny_model(input_ids=ids)
        h.remove()
    assert len(norms) == 1
    assert norms[0] == pytest.approx(float(seen["h"][0, -1].float().norm()), rel=1e-6)
    assert not tiny_model.model.layers[0]._forward_hooks
    assert not tiny_model._forward_pre_hooks


def test_prefill_detection_during_generation(tiny_model):
    norms = []
    with torch.no_grad(), steering.steer(tiny_model, 1, None, 0.0, record_norms=norms):
        tiny_model.generate(torch.tensor([[3, 4, 5]]), max_new_tokens=4, do_sample=False)
    assert len(norms) == 1       # recorded on the prefill pass only, not on the 3 cached decoding steps


class _StubTokenizer:
    """Maps each candidate string to a fixed id sequence."""
    table = {"Yes": [10], "yes": [11], " Yes": [12], " yes": [12], "YES": [13, 14],
             "No": [20], "no": [21], " No": [22], " no": [23], "NO": [24]}

    def encode(self, text, add_special_tokens=False):
        return self.table.get(text, [99])

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text)}


def test_candidate_token_ids_rules():
    yes, no = steering.candidate_token_ids(_StubTokenizer(), "llama-3.1-8b")
    assert yes == [10, 11, 12, 12, 13] and no == [20, 21, 22, 23, 24]       # first token of each
    yes, no = steering.candidate_token_ids(_StubTokenizer(), "ministral-3-14b")
    assert yes == [10, 11, 12] and no == [20, 21, 22, 23, 24]               # single-token only, sorted


def test_magnitude_helpers():
    assert steering.median_residual_norm([3.0, 1.0, 2.0, 10.0]) == pytest.approx(2.5)
    assert steering.alpha_for_r(0.1, 40.0) == pytest.approx(4.0)
    assert steering.r_for_alpha(4.0, 40.0) == pytest.approx(0.1)


def test_heldout_sample_is_deterministic_and_heldout():
    a = steering.heldout_sample("triviaqa", 42)
    b = steering.heldout_sample("triviaqa", 42)
    assert [p["id"] for p in a] == [p["id"] for p in b]
    assert len(a) == 200 and len({p["id"] for p in a}) == 200
    assert all(p["split"] == "test" for p in a)
    assert a[0]["id"] == "triviaqa_00947"        # first pair of the paper's seed-42 sample
