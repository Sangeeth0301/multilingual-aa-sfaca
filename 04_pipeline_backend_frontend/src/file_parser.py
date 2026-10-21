"""
file_parser.py
--------------
Multi-Format File Parser for SFA-CA Web App.

Supports reading text content from:
- Plain text (.txt, .md, .csv)
- PDF Documents (.pdf) via pypdf / pdfplumber fallback
- Word Documents (.docx) via python-docx / zipfile fallback
"""

import os
from typing import Tuple


def parse_uploaded_file(file_path: str) -> Tuple[str, str]:
    """
    Parses an uploaded file and returns (extracted_text, file_type).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".txt", ".md", ".csv", ".json", ".log"]:
        return parse_text_file(file_path), ext[1:].upper()
    elif ext == ".pdf":
        return parse_pdf_file(file_path), "PDF"
    elif ext == ".docx":
        return parse_docx_file(file_path), "DOCX"
    else:
        raise ValueError(f"Unsupported file format '{ext}'. Supported formats: .txt, .pdf, .docx, .md")


def parse_text_file(file_path: str) -> str:
    """Reads plain text files with UTF-8 or Latin-1 fallback."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()


def parse_pdf_file(file_path: str) -> str:
    """Parses PDF document using pypdf or PyPDF2."""
    try:
        import pypdf
        reader = pypdf.PdfReader(file_path)
        text_runs = [page.extract_text() for page in reader.pages if page.extract_text()]
        return "\n\n".join(text_runs)
    except ImportError:
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(file_path)
            text_runs = [page.extract_text() for page in reader.pages if page.extract_text()]
            return "\n\n".join(text_runs)
        except ImportError:
            raise ImportError("Please install 'pypdf' or 'PyPDF2' to parse PDF files: pip install pypdf")


def parse_docx_file(file_path: str) -> str:
    """Parses DOCX document using python-docx or zipfile fallback."""
    try:
        import docx
        doc = docx.Document(file_path)
        return "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except ImportError:
        # Fallback to standard library zipfile + xml parsing if python-docx isn't installed
        import zipfile
        import xml.etree.ElementTree as ET
        try:
            with zipfile.ZipFile(file_path) as docx_zip:
                xml_content = docx_zip.read("word/document.xml")
                root = ET.fromstring(xml_content)
                text_runs = []
                for p in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
                    p_text = "".join([t.text for t in p.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t") if t.text])
                    if p_text.strip():
                        text_runs.append(p_text)
                return "\n\n".join(text_runs)
        except Exception as e:
            raise ValueError(f"Failed to parse DOCX file: {e}")


if __name__ == "__main__":
    # Smoke test
    print("File parser module loaded successfully.")
