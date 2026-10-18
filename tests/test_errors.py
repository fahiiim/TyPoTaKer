import random

from typotaker.data.errors import corrupt_sentence


def test_identity_rate_and_noise():
    rng = random.Random(0)
    text = "The server listens on port 8000 by default."
    n_same = sum(corrupt_sentence(text, rng, p=0.7) == text for _ in range(200))
    assert 20 < n_same < 120
    rng = random.Random(1)
    noisy = [corrupt_sentence(text, rng, p=1.0) for _ in range(20)]
    assert any(s != text for s in noisy)
