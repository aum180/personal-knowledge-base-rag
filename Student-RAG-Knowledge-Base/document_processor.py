"""
document_processor.py
=====================
Enhanced Document Processing Module for Personal RAG Knowledge Base.
Supports PDF, DOCX, TXT, and Markdown documents.

Key capabilities:
- Robust multi-format text extraction (page-by-page for PDF using PyPDF).
- Detection of empty, corrupted, or scanned/image-based PDFs requiring OCR.
- Deep Unicode cleaning (standardizing dashes, quotes, whitespace, and invalid characters).
- Boundary-aware text chunking (500-800 chars, 100-150 overlap) without cutting words.
- Rich metadata preservation: filename, doc_type, page_number, chunk_id, section_tag, char_length.
- Detailed extraction inspection reports for Streamlit UI preview.
"""

import os
import re
import hashlib
import unicodedata
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

# Document parsing libraries
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None


class DocumentProcessor:
    """
    Handles robust text extraction, preprocessing, inspection,
    and metadata-rich chunking for personal documents.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

    # Common section keywords in resumes and academic documents
    SECTION_PATTERNS = {
        "HEADER": r"^(curriculum vitae|resume|profile|summary|contact)",
        "EDUCATION": r"(education|academic background|qualifications|degree|coursework|university|college)",
        "EXPERIENCE": r"(experience|internship|work experience|employment history|professional experience)",
        "SKILLS": r"(technical skills|skills|competencies|technologies|tools|programming languages)",
        "PROJECTS": r"(projects|academic projects|capstone|key projects|personal projects)",
        "PUBLICATIONS": r"(publications?|research\s+publications?|papers?|conferences?|accepted\s+at|ieee|springer|scopus|proceedings|journal|arxiv|lnns|lncs|iccisd|indea)",
        "CERTIFICATIONS": r"(certifications|certificates|licenses|courses|awards)",
    }

    def __init__(self, chunk_size: int = 650, chunk_overlap: int = 120):
        """
        Args:
            chunk_size: Target character length per chunk (default: 650 chars, range 500-800).
            chunk_overlap: Overlap in characters between adjacent chunks (default: 120 chars).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def compute_file_hash(file_bytes: bytes) -> str:
        """Computes SHA-256 hash to detect changes and duplicates."""
        return hashlib.sha256(file_bytes).hexdigest()

    def clean_text(self, text: str) -> str:
        """
        Cleans and normalizes text:
        - Normalizes Unicode (NFKD)
        - Replaces non-standard dashes, quotes, and replacement characters (like )
        - Preserves important headings, casing, dates, and names
        - Normalizes excessive whitespace and standardized newlines
        """
        if not text:
            return ""

        # Normalize unicode characters
        text = unicodedata.normalize("NFKC", text)

        # Replace non-breaking spaces and invalid replacement characters
        text = text.replace("\xa0", " ").replace("\ufffd", " - ")

        # Standardize typographic quotes and dashes
        text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015]", " - ", text)
        text = re.sub(r"[\u2018\u2019\u201a\u201b]", "'", text)
        text = re.sub(r"[\u201c\u201d\u201e\u201f]", '"', text)

        # Standardize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse 3+ newlines into 2
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Normalize repeated horizontal spaces/tabs
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    def inspect_and_extract_file(self, file_path: str) -> Dict[str, Any]:
        """
        Reads a document from path and produces an inspection report with full extracted text.
        """
        path = Path(file_path)
        if not path.exists():
            return {
                "filename": path.name,
                "path": str(path),
                "doc_type": path.suffix.lower(),
                "num_pages": 0,
                "char_count": 0,
                "word_count": 0,
                "preview": "",
                "text": "",
                "is_empty": True,
                "is_scanned": False,
                "error": f"File not found: {file_path}",
                "file_hash": ""
            }

        with open(path, "rb") as f:
            content = f.read()

        return self.inspect_and_extract_bytes(content, filename=path.name)

    def inspect_and_extract_bytes(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """
        Extracts text from byte content and returns a comprehensive diagnostic report:
        - filename, doc_type, num_pages, char_count, word_count
        - text, preview
        - is_empty, is_scanned (requiring OCR warning)
        - error (if any)
        """
        ext = Path(filename).suffix.lower()
        file_hash = self.compute_file_hash(file_bytes)

        report = {
            "filename": filename,
            "doc_type": ext,
            "num_pages": 1,
            "char_count": 0,
            "word_count": 0,
            "preview": "",
            "text": "",
            "is_empty": False,
            "is_scanned": False,
            "error": None,
            "file_hash": file_hash,
            "pages_text": []
        }

        if ext not in self.SUPPORTED_EXTENSIONS:
            report["is_empty"] = True
            report["error"] = f"Unsupported file format '{ext}'. Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            return report

        if len(file_bytes) == 0:
            report["is_empty"] = True
            report["error"] = f"The file '{filename}' is completely empty (0 bytes)."
            return report

        try:
            if ext in {".txt", ".md"}:
                text = self._extract_txt_or_md(file_bytes, filename)
                clean = self.clean_text(text)
                report["text"] = clean
                report["pages_text"] = [clean]
            elif ext == ".docx":
                text = self._extract_docx(file_bytes, filename)
                clean = self.clean_text(text)
                report["text"] = clean
                report["pages_text"] = [clean]
            elif ext == ".pdf":
                pages, raw_text = self._extract_pdf_pages(file_bytes, filename)
                report["num_pages"] = len(pages)
                clean_pages = [self.clean_text(p) for p in pages]
                report["pages_text"] = clean_pages
                report["text"] = self.clean_text(raw_text)

                # Scanned PDF detection
                if report["num_pages"] > 0 and len(report["text"]) < 40:
                    report["is_scanned"] = True
                    report["error"] = (
                        f"PDF '{filename}' contains {report['num_pages']} page(s) but has zero or suspiciously little extractable text "
                        f"({len(report['text'])} characters). This document appears to be scanned or image-based and requires OCR."
                    )
        except Exception as e:
            report["error"] = f"Error reading '{filename}': {str(e)}"
            report["is_empty"] = True
            return report

        char_len = len(report["text"])
        report["char_count"] = char_len
        report["word_count"] = len(report["text"].split()) if char_len > 0 else 0
        report["preview"] = report["text"][:350] + ("..." if char_len > 350 else "")

        if char_len == 0 and not report["is_scanned"]:
            report["is_empty"] = True
            report["error"] = f"No readable text could be extracted from '{filename}'."

        return report

    def extract_text_from_file(self, file_path: str) -> str:
        """Helper to extract text and raise on error."""
        report = self.inspect_and_extract_file(file_path)
        if report["error"]:
            raise ValueError(report["error"])
        return report["text"]

    def extract_text_from_bytes(self, file_bytes: bytes, filename: str) -> str:
        """Helper to extract text from bytes and raise on error."""
        report = self.inspect_and_extract_bytes(file_bytes, filename)
        if report["error"]:
            raise ValueError(report["error"])
        return report["text"]

    def _extract_txt_or_md(self, file_bytes: bytes, filename: str) -> str:
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]
        for enc in encodings:
            try:
                return file_bytes.decode(enc)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Failed to decode text file '{filename}' with standard encodings.")

    def _extract_docx(self, file_bytes: bytes, filename: str) -> str:
        """
        Layout-aware DOCX text extraction:
        - Extracts text from headers, footers, body paragraphs, tables, and sidebars.
        - Robust handling of Word textboxes (<w:txbxContent>) inside drawings and shapes (<w:drawing>).
        - Correctly filters out <mc:Fallback> elements from <mc:AlternateContent> to prevent duplicate text.
        - Preserves logical reading order across multi-column, sidebar, and tabular layouts.
        """
        if docx is None:
            raise ImportError("python-docx is not installed. Please install it using: pip install python-docx")

        import io
        import copy

        try:
            doc = docx.Document(io.BytesIO(file_bytes))
            extracted_blocks = []
            seen_blocks = set()

            def add_block(txt: str):
                cleaned = txt.strip()
                if cleaned and cleaned not in seen_blocks:
                    extracted_blocks.append(cleaned)
                    seen_blocks.add(cleaned)

            # 1. Header paragraphs and tables
            for sec in doc.sections:
                try:
                    for p in sec.header.paragraphs:
                        if p.text.strip():
                            add_block(p.text.strip())
                    for tbl in sec.header.tables:
                        for row in tbl.rows:
                            cells = [c.text.strip() for c in row.cells if c.text.strip()]
                            if cells:
                                add_block(" | ".join(cells))
                except Exception:
                    pass

            # 2. Body traversal (paragraphs, textboxes/shapes, tables in order)
            body = copy.deepcopy(doc._element.body)

            # Prune mc:Fallback to prevent duplicate text from AlternateContent (drawingML vs VML)
            for ac in body.findall(".//{http://schemas.openxmlformats.org/markup-compatibility/2006}AlternateContent"):
                choices = ac.findall("{http://schemas.openxmlformats.org/markup-compatibility/2006}Choice")
                fallbacks = ac.findall("{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback")
                if choices:
                    for fb in fallbacks:
                        ac.remove(fb)

            for elem in body:
                tag = elem.tag.split("}")[-1]

                if tag == "p":
                    # Check for text boxes / floating shapes (w:txbxContent)
                    txbx_elements = elem.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}txbxContent")
                    if txbx_elements:
                        for txbx in txbx_elements:
                            t_paras = txbx.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p")
                            lines = []
                            for tp in t_paras:
                                line = "".join([
                                    t.text for t in tp.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
                                    if t.text
                                ]).strip()
                                if line:
                                    lines.append(line)
                            if lines:
                                add_block("\n".join(lines))

                    # Direct paragraph runs outside drawing textboxes
                    direct_runs = elem.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}r")
                    direct_text = "".join([
                        t.text for r in direct_runs
                        for t in r.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
                        if t.text
                    ]).strip()
                    if direct_text:
                        add_block(direct_text)

                elif tag == "tbl":
                    # Tables in body
                    rows_text = []
                    for row in elem.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tr"):
                        cell_texts = []
                        for cell in row.findall("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tc"):
                            ct = " ".join([
                                t.text for t in cell.findall(".//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t")
                                if t.text
                            ]).strip()
                            if ct:
                                cell_texts.append(ct)
                        if cell_texts:
                            rows_text.append(" | ".join(cell_texts))
                    if rows_text:
                        add_block("\n".join(rows_text))

            # 3. Footer paragraphs
            for sec in doc.sections:
                try:
                    for p in sec.footer.paragraphs:
                        if p.text.strip():
                            add_block(p.text.strip())
                except Exception:
                    pass

            return "\n\n".join(extracted_blocks)
        except Exception as e:
            raise ValueError(f"Could not read DOCX '{filename}': {str(e)}")

    def _extract_pdf_pages(self, file_bytes: bytes, filename: str) -> Tuple[List[str], str]:
        """
        Layout-aware PDF extraction:
        - Uses pypdf layout extraction mode to preserve multi-column reading order.
        - Falls back gracefully to standard plain text extraction if layout mode is empty or fails.
        """
        if pypdf is None:
            raise ImportError("pypdf is not installed. Please install it using: pip install pypdf")

        import io
        try:
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            pages = []
            for page in reader.pages:
                extracted = ""
                try:
                    # Attempt layout mode for multi-column preservation
                    extracted = page.extract_text(extraction_mode="layout") or ""
                except Exception:
                    extracted = ""

                # Fallback to plain mode if layout mode returned empty or failed
                if not extracted.strip():
                    try:
                        extracted = page.extract_text() or ""
                    except Exception:
                        extracted = ""

                pages.append(extracted)
            return pages, "\n\n".join(pages)
        except Exception as e:
            raise ValueError(f"Could not read PDF '{filename}': {str(e)}")

    def detect_section_hint(self, text: str) -> str:
        """
        Identifies which resume/document section or topic the text chunk belongs to,
        using both explicit headings AND contextual content patterns.
        """
        text_lower = text.lower()

        # 1. Heading patterns
        for section, pattern in self.SECTION_PATTERNS.items():
            if re.search(pattern, text_lower):
                return section

        # 2. Content-based patterns
        if re.search(r"\b(b\.?f\.?a|b\.?tech|b\.?s\.?|bachelor|master|ph\.?d|degree|university|college|school|academic|coursework)\b", text_lower):
            return "EDUCATION"
        if re.search(r"\b(20\d\d|19\d\d)\s*[-–—to]+\s*(present|current|20\d\d|\d{4})\b", text_lower) or any(w in text_lower for w in ["worked with", "collaborated with", "designed over", "developed original", "responsibilities", "job title", "graphic designer", "junior graphic designer", "internship"]):
            return "EXPERIENCE"
        if re.search(r"\b(photoshop|illustrator|after effects|indesign|premiere|python|java|javascript|react|html|css|sql|figma|docker|git)\b", text_lower):
            return "SKILLS"
        if any(w in text_lower for w in ["accepted at", "ieee", "springer", "proceedings", "conference", "journal"]):
            return "PUBLICATIONS"

        return "GENERAL"

    def chunk_text(
        self,
        text: str,
        source_filename: str,
        doc_type: Optional[str] = None,
        page_number: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Splits text into chunks of approximately 500-800 characters with 100-150 overlap.
        Preserves complete words and sentence boundaries.
        Tags metadata: source, doc_type, chunk_id, page_number, section_hint, is_header.
        """
        if not text:
            return []

        if doc_type is None:
            doc_type = Path(source_filename).suffix.lower()

        chunks: List[Dict[str, Any]] = []
        n = len(text)
        start = 0
        chunk_idx = 0

        while start < n:
            target_end = min(start + self.chunk_size, n)

            # If not at the end of text, find the nearest clean boundary
            if target_end < n:
                window = text[start:target_end]
                boundary = -1

                # 1. Prefer paragraph break (\n\n)
                para_pos = window.rfind("\n\n")
                if para_pos != -1 and para_pos > int(self.chunk_size * 0.45):
                    boundary = start + para_pos + 2

                # 2. Prefer single newline (\n)
                if boundary == -1:
                    nl_pos = window.rfind("\n")
                    if nl_pos != -1 and nl_pos > int(self.chunk_size * 0.45):
                        boundary = start + nl_pos + 1

                # 3. Prefer sentence ending punctuation (. ? !)
                if boundary == -1:
                    punct_matches = list(re.finditer(r"[\.\?\!]\s+", window))
                    if punct_matches:
                        last_punct = punct_matches[-1]
                        if last_punct.end() > int(self.chunk_size * 0.4):
                            boundary = start + last_punct.end()

                # 4. Fallback to whitespace to never cut words in half
                if boundary == -1:
                    space_pos = window.rfind(" ")
                    if space_pos != -1 and space_pos > int(self.chunk_size * 0.3):
                        boundary = start + space_pos + 1
                    else:
                        boundary = target_end

                target_end = boundary

            chunk_content = text[start:target_end].strip()
            if chunk_content:
                section_tag = self.detect_section_hint(chunk_content)
                is_header = (chunk_idx == 0 or section_tag == "HEADER")

                chunks.append({
                    "chunk_id": chunk_idx,
                    "text": chunk_content,
                    "source": source_filename,
                    "doc_type": doc_type,
                    "page_number": page_number,
                    "char_length": len(chunk_content),
                    "word_count": len(chunk_content.split()),
                    "section_hint": section_tag,
                    "is_header": is_header,
                    "start_idx": start,
                    "end_idx": target_end
                })
                chunk_idx += 1

            if target_end >= n:
                break

            # Step forward with overlap, ensuring forward progress
            step = max(1, (target_end - start) - self.chunk_overlap)
            start = start + step

        return chunks

    def process_directory(self, directory_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, str]]]:
        """
        Scans a directory, extracts all supported files, generates chunks with rich metadata,
        and returns:
            (all_chunks, successful_file_reports, failed_file_reports)
        """
        all_chunks = []
        successful_reports = []
        failed_reports = []

        dir_path = Path(directory_path)
        if not dir_path.exists() or not dir_path.is_dir():
            return [], [], [{"filename": str(directory_path), "error": "Directory does not exist"}]

        files = sorted(dir_path.iterdir())
        for file in files:
            if file.is_file() and file.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                if file.name.lower() in {"readme.txt", "readme.md"}:
                    continue

                report = self.inspect_and_extract_file(str(file))
                if report["error"] or report["is_empty"]:
                    failed_reports.append({
                        "filename": file.name,
                        "error": report["error"] or "File is empty",
                        "is_scanned": report.get("is_scanned", False)
                    })
                else:
                    # If PDF with pages, chunk per page to retain page numbers
                    if report["doc_type"] == ".pdf" and len(report["pages_text"]) > 1:
                        file_chunks = []
                        for page_idx, page_txt in enumerate(report["pages_text"], 1):
                            if page_txt.strip():
                                p_chunks = self.chunk_text(
                                    page_txt,
                                    source_filename=file.name,
                                    doc_type=report["doc_type"],
                                    page_number=page_idx
                                )
                                file_chunks.extend(p_chunks)
                    else:
                        file_chunks = self.chunk_text(
                            report["text"],
                            source_filename=file.name,
                            doc_type=report["doc_type"]
                        )

                    # Re-index chunk IDs across the document
                    for idx, c in enumerate(file_chunks):
                        c["chunk_id"] = idx

                    all_chunks.extend(file_chunks)
                    successful_reports.append(report)

        return all_chunks, successful_reports, failed_reports
