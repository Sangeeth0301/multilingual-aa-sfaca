import io
import os
import sys

os.environ["SFACA_STUB"] = "1"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

import pytest
from fastapi.testclient import TestClient

import api
from analyzer import analyze, segment
from stub_scorer import StubScorer

TEXT = ("Photosynthesis is the process by which plants convert light into chemical energy. "
        "I remember my grandmother watering her tomatoes every dawn, grumbling about the heat. "
        "Moreover, it is crucial to understand that this process plays a pivotal role. "
        "The reaction is $6CO_2 + 6H_2O \\rightarrow C_6H_{12}O_6$ in total. "
        "Furthermore, researchers continue to delve into the multifaceted mechanisms. "
        "Honestly, the lab results we got last week were a mess.")


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


def test_segments_reproduce_text():
    segs = segment(TEXT)
    assert "".join(s["text"] for s in segs) == TEXT
    assert any(s["type"] == "notation" for s in segs)


def test_analyze_schema_and_math_excluded():
    res = analyze(TEXT, StubScorer())
    assert abs(res["ai_pct"] + res["human_pct"] - 100) < 0.05
    assert abs(sum(res["generator_probs"].values()) - 1) < 0.01
    for s in res["segments"]:
        if s["type"] == "notation":
            assert s["ai_prob"] is None and not s["scored"]
    assert res["script"] == "Latin"


def test_empty_and_notation_only():
    assert analyze("", StubScorer())["low_evidence"]
    assert analyze("$$E=mc^2$$", StubScorer())["low_evidence"]


def test_mixed_script_flag():
    t = ("This is a long enough English sentence to be scored properly by the model. "
         "Это достаточно длинное предложение на русском языке для проверки маршрутизации.")
    res = analyze(t, StubScorer())
    assert res["mixed_script"] is True


def test_long_doc_cap():
    with pytest.raises(ValueError):
        analyze("This is a sentence that is long enough to count. " * 20000, StubScorer())


def test_health(client):
    assert client.get("/health").json()["model"] == "stub"


def test_analyze_text(client):
    r = client.post("/api/analyze", data={"text": TEXT})
    assert r.status_code == 200
    assert "segments" in r.json()


def test_rejects_short_and_bad_type(client):
    assert client.post("/api/analyze", data={"text": "hi"}).status_code == 400
    r = client.post("/api/analyze", files={"file": ("x.exe", io.BytesIO(b"abc"), "application/octet-stream")})
    assert r.status_code == 415


def test_upload_txt_and_docx(client):
    r = client.post("/api/analyze", files={"file": ("a.txt", io.BytesIO(TEXT.encode()), "text/plain")})
    assert r.status_code == 200 and r.json()["filename"] == "a.txt"
    import docx
    d = docx.Document()
    d.add_paragraph(TEXT)
    buf = io.BytesIO()
    d.save(buf)
    buf.seek(0)
    r = client.post("/api/analyze", files={"file": ("a.docx", buf, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
    assert r.status_code == 200


def test_corrupt_pdf_is_clean_error(client):
    r = client.post("/api/analyze", files={"file": ("a.pdf", io.BytesIO(b"not a pdf"), "application/pdf")})
    assert r.status_code == 422


def test_export_html(client):
    r = client.post("/api/export", data={"text": TEXT})
    assert r.status_code == 200 and "AI-attribution report" in r.text
