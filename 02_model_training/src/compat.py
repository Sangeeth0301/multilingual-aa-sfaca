# ── PEFT torchao compatibility patch ─────────────────────────────────────────
# PEFT 0.19.x raises ImportError when torchao < 0.16.0 (Kaggle locks 0.10.0).
# Monkey-patch is_torchao_available to return False instead of crashing.
# Replace dispatch_torchao directly (avoids recursive is_torchao_available wrapping)

try:
    import peft.import_utils as _peft_utils
    _orig_torchao = _peft_utils.is_torchao_available
    def _safe_torchao():
        try:
            return _orig_torchao()
        except ImportError:
            return False
    _peft_utils.is_torchao_available = _safe_torchao
    try:
        import peft.tuners.lora.torchao as _lora_torchao
        _lora_torchao.is_torchao_available = _safe_torchao
        _lora_torchao.dispatch_torchao = lambda *a, **kw: None
    except Exception:
        pass
except Exception:
    pass
