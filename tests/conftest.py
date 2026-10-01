from __future__ import annotations

import numpy as np
import pytest

from nanoserve.config import ModelConfig
from nanoserve.model import GPT2


@pytest.fixture(scope="session")
def tiny_model() -> GPT2:
    return GPT2.random(ModelConfig.tiny(), seed=0, dtype=np.float64)
