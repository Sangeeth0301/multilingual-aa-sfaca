import sys
import os
import tempfile
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "04_pipeline_backend_frontend", "backend")))

from inference import InferenceEngine, detect_script_cluster, detect_script
from contrastive_loss import supervised_contrastive_loss
from sanitizer import TextSanitizer
from file_parser import parse_uploaded_file
from model import SFACAModel
import app
import api

# ── Router Tests ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected_cluster, expected_supported, expected_mixed", [
    ("", "latin", False, False),
    ("123 456", "latin", False, False),
    ("😊👍", "latin", False, False),
    ("Café au lait", "latin", True, False),
    ("Привет мир", "cyrillic", True, False),
    ("γειά σου", "greek", True, False),
    ("مرحبا", "arabic", True, False),
    ("سلام", "arabic", True, False), # Persian
    ("ہیلو", "arabic", True, False), # Urdu
    ("你好", "hanzi", True, False),
    ("こんにちは世界", "hanzi", True, False),
    ("안녕하세요", "hanzi", True, False), # Hangul
    ("Hello world مرحبا", "latin", True, True),
    ("שלום", "latin", False, False), # Hebrew (unsupported -> fallback latin)
    ("नमस्ते", "latin", False, False), # Devanagari (unsupported -> fallback latin)
])
def test_router_cases(text, expected_cluster, expected_supported, expected_mixed):
    res = detect_script(text)
    assert res["cluster"] == expected_cluster
    assert res["supported"] == expected_supported
    assert res["mixed"] == expected_mixed
    assert detect_script_cluster(text) == expected_cluster

# ── Fake Backbone & Mocks ───────────────────────────────────────────────────

class FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

class FakePeftModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.active_adapter = "default"
    def set_adapter(self, name):
        self.active_adapter = name
    def add_adapter(self, name, config):
        pass
    def load_adapter(self, path, adapter_name):
        pass
    def forward(self, input_ids, attention_mask):
        return self.base_model(input_ids, attention_mask)

class FakeBaseModelOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state

class FakeBaseModel(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.config = MagicMock(hidden_size=hidden_size)
        self.hidden = nn.Parameter(torch.randn(1, 3, hidden_size))
    def forward(self, input_ids, attention_mask):
        return FakeBaseModelOutput(self.hidden)

@pytest.fixture
def mock_transformers():
    with patch("model.AutoModel.from_pretrained") as mock_model, \
         patch("model.get_peft_model") as mock_peft, \
         patch("inference.build_tokenizer") as mock_tok:
        
        base_model = FakeBaseModel()
        mock_model.return_value = base_model
        mock_peft.side_effect = lambda m, c, adapter_name: FakePeftModel(m)
        mock_tok.return_value = FakeTokenizer()
        
        yield mock_model, mock_peft, mock_tok

# ── Head Isolation Test ─────────────────────────────────────────────────────

def test_head_isolation(mock_transformers):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 2 fake heads
        cluster1, cluster2 = "latin", "cyrillic"
        os.makedirs(os.path.join(tmpdir, cluster1))
        os.makedirs(os.path.join(tmpdir, cluster2))
        
        head1 = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 8))
        head2 = nn.Sequential(nn.Linear(32, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 8))
        
        # Ensure they are different
        with torch.no_grad():
            head2[0].weight.fill_(1.0)
            head1[0].weight.fill_(0.0)
            
        torch.save(head1.state_dict(), os.path.join(tmpdir, cluster1, "classifier_head.pt"))
        torch.save(head2.state_dict(), os.path.join(tmpdir, cluster2, "classifier_head.pt"))
        
        engine = InferenceEngine(device=torch.device("cpu"))
        engine.load_from_dir(tmpdir)
        
        # The fake backbone returns the exact same hidden state regardless of cluster (because FakePeftModel ignores set_adapter for actual weights).
        # But the heads are different, so the outputs MUST be different.
        res1 = engine.predict("test", cluster1)
        res2 = engine.predict("test", cluster2)
        
        # Logits should be different
        assert res1["logits"] != res2["logits"]
        
        # Verify engine used head1 for cluster1
        with torch.no_grad():
            head1.eval()
            head2.eval()
            pooled = engine.model.backbone(None, None).last_hidden_state[:, 0, :]
            expected1 = head1(pooled).squeeze().cpu().tolist()
            expected2 = head2(pooled).squeeze().cpu().tolist()
        
        # Almost equal due to float precision
        for a, b in zip(res1["logits"], expected1): assert abs(a-b) < 1e-4
        for a, b in zip(res2["logits"], expected2): assert abs(a-b) < 1e-4

# ── Parity Test ─────────────────────────────────────────────────────────────

def test_engine_parity(mock_transformers):
    with tempfile.TemporaryDirectory() as tmpdir:
        cluster = "latin"
        os.makedirs(os.path.join(tmpdir, cluster))
        
        # Load a real SFACAModel with the mock to generate a head
        model = SFACAModel(clusters=[cluster])
        torch.save(model.classifier.state_dict(), os.path.join(tmpdir, cluster, "classifier_head.pt"))
        
        engine = InferenceEngine(model=model, tokenizer=FakeTokenizer(), device=torch.device("cpu"))
        # We manually loaded model/tokenizer, now load heads
        engine.load_from_dir(tmpdir)
        
        res = engine.predict("hello", cluster)
        engine_logits = res["logits"]
        
        with torch.no_grad():
            model.eval()
            model_logits = model(torch.tensor([[1,2,3]]), torch.tensor([[1,1,1]]), cluster).squeeze().cpu().tolist()
            
        for a, b in zip(engine_logits, model_logits):
            assert abs(a-b) < 1e-4

# ── Error Test ──────────────────────────────────────────────────────────────

def test_missing_adapter_dir():
    engine = InferenceEngine(device=torch.device("cpu"))
    os.environ.pop("ADAPTER_DIR", None)
    with pytest.raises(ValueError, match="ADAPTER_DIR"):
        engine.load_from_dir()

# ── Loss Tests ──────────────────────────────────────────────────────────────

def test_contrastive_loss():
    # Identical embeddings -> similarity is high
    emb = torch.ones(4, 32)
    labels = torch.tensor([0, 0, 1, 1])
    loss = supervised_contrastive_loss(emb, labels)
    assert not torch.isnan(loss)
    
    # No positives -> loss should be 0
    emb2 = torch.randn(4, 32)
    labels2 = torch.tensor([0, 1, 2, 3])
    loss2 = supervised_contrastive_loss(emb2, labels2)
    assert loss2.item() == 0.0

    # Normal batch
    loss3 = supervised_contrastive_loss(torch.randn(8, 32), torch.tensor([0,0,0,1,1,2,2,3]))
    assert loss3.item() > 0.0

# ── Sanitizer Tests ─────────────────────────────────────────────────────────

def test_sanitizer():
    # Changes in Phase E
    san = TextSanitizer()
    text, meta = san.sanitize("Here is some math: $x = 1$")
    # Current behavior only. Let's just assert it runs and returns text.
    assert isinstance(text, str)
    assert isinstance(meta, dict)

# ── Parser Tests ────────────────────────────────────────────────────────────

def test_parser_txt_md_docx():
    # txt
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"Hello txt")
        tmp_txt = f.name
    
    # md
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
        f.write(b"# Hello md")
        tmp_md = f.name
        
    try:
        t_txt, _ = parse_uploaded_file(tmp_txt)
        assert "Hello txt" in t_txt
        
        t_md, _ = parse_uploaded_file(tmp_md)
        assert "Hello md" in t_md
        
        import docx
        doc = docx.Document()
        doc.add_paragraph("Hello docx")
        tmp_docx = tmp_txt + ".docx"
        doc.save(tmp_docx)
        t_docx, _ = parse_uploaded_file(tmp_docx)
        assert "Hello docx" in t_docx
        os.remove(tmp_docx)
    finally:
        os.remove(tmp_txt)
        os.remove(tmp_md)
