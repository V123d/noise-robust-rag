import os
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx
except ImportError:
    docx = None

def extract_pdf(pdf_path, out_path):
    text = ""
    if fitz:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
    elif PdfReader:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text()
    else:
        text = "No PDF library found. Please install pymupdf or pypdf2."
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

def extract_docx(docx_path, out_path):
    text = ""
    if docx:
        doc = docx.Document(docx_path)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        text = "No DOCX library found. Please install python-docx."
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    extract_pdf(r"d:\毕业论文实验\开题报告1.pdf", r"d:\毕业论文实验\report.txt")
    extract_docx(r"d:\毕业论文实验\第一二章（未修改版）.docx", r"d:\毕业论文实验\thesis.txt")
    print("Extraction complete.")
