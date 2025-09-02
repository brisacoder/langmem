#!/usr/bin/env python3
"""PDF section extraction helpers.

This module contains small, importable helper functions to locate and extract
sections from PDFs. It focuses on two common strategies:

- heading detection using layout cues (font sizes and position), and
- mapping printed/Table-of-Contents page numbers to physical PDF page indices
    (useful when a document shows logical page numbers in the TOC).

All functions are intended to be imported and used from notebooks or other
programs; there is intentionally no command-line interface in this module.

Example
-------
>>> from pathlib import Path
>>> from scripts.extract_pdf_section import (
...     extract_section_by_heading, extract_pages_from_printed_range,
... )
>>> repo = Path('.')
>>> inp = str(repo / 'examples' / 'intro_videos' / '25_U717_Expedition_OM_ENG_V2.pdf')
>>> # Preview matches for the heading
>>> res = extract_section_by_heading(inp, r'(?i)Data Privacy', '/tmp/out.pdf', preview=True)
>>> print(res['raw_matches'][0])

The functions rely on PyMuPDF (imported as ``fitz``). At runtime PyMuPDF must
be installed in the active Python environment.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import fitz  # type: ignore


def looks_like_toc(page_text: str) -> bool:
    """Heuristically detect whether a page contains a Table-of-Contents.

    The heuristic checks for repeated dot-leaders ("....... 24") and for many
    short lines that end with a small integer (likely TOC entries).

    Args:
        page_text: Full page text as returned by PyMuPDF's text extraction.

    Returns:
        True if the page looks like a TOC, False otherwise.

    Example:
        >>> looks_like_toc('1. Introduction ............ 3\n2. Usage ........ 5')
        True
    """
    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    dot_leader_lines = sum(1 for ln in lines if re.search(r"\.{2,}\s*\d+\s*$", ln))
    short_page_number_lines = sum(
        1
        for ln in lines
        if re.search(r"\b\d{1,3}\s*$", ln) and len(ln.split()) <= 8
    )
    return (dot_leader_lines >= 2) or (short_page_number_lines >= 6)


def compute_median_font_size(doc: Any, sample_pages: int = 20) -> float:
    """Compute a robust median font size for the document.

    The median font size is useful to identify headings which usually have
    larger font sizes than the document body. The function samples pages
    across the document to collect span sizes and returns the median.

    Args:
        doc: A PyMuPDF ``Document`` object (typed as ``Any`` to avoid strict
            static-analysis errors).
        sample_pages: Maximum number of pages to sample (spread evenly).

    Returns:
        The median font size observed, or 0.0 if no spans were found.

    Example:
        >>> compute_median_font_size(doc, sample_pages=10)
        10.0
    """
    sizes: List[float] = []
    step = max(1, doc.page_count // sample_pages) if doc.page_count > sample_pages else 1
    for i in range(0, doc.page_count, step):
        page = doc.load_page(i)
        try:
            # PyMuPDF's Page.get_text is not visible to static type checkers here
            data = page.get_text("dict")  # type: ignore[attr-defined]
        except RuntimeError:
            continue
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    sizes.append(float(span.get("size", 0)))
    if not sizes:
        return 0.0
    sizes.sort()
    return sizes[len(sizes) // 2]


def preview_matches(doc: Any, heading_re: str, context_chars: int = 120) -> List[Tuple[int, str]]:
    """Scan the document for textual regex matches and return context snippets.

    Args:
        doc: PyMuPDF Document object.
        heading_re: Regular expression to search for (string form).
        context_chars: Number of characters of context to include around the match.

    Returns:
        A list of tuples (page_index, snippet) where page_index is 0-based.

    Example:
        >>> preview_matches(doc, r"(?i)Data Privacy", context_chars=80)
        [(27, '...Data Privacy ...')]
    """
    matches: List[Tuple[int, str]] = []
    cre = re.compile(heading_re, re.IGNORECASE)
    for i in range(doc.page_count):
        # static checkers do not know about Page.get_text
        text = doc.load_page(i).get_text("text")  # type: ignore[attr-defined]
        m = cre.search(text)
        if m:
            start = max(0, m.start() - context_chars)
            end = min(len(text), m.end() + context_chars)
            snippet = text[start:end].replace("\n", " ")
            matches.append((i, snippet))
    return matches


def build_printed_page_map(doc: Any, bottom_margin: int = 80, top_margin: int = 80) -> Dict[str, int]:
    """Build a mapping from printed/logical page numbers to physical pages.

    The function first attempts to read PDF page labels (if supported by the
    PyMuPDF version). If labels are not available or not human-readable it
    falls back to scanning visible text spans near the top/bottom of each
    physical page and extracting integer-looking page numbers.

    Args:
        doc: PyMuPDF Document object.
        bottom_margin: Vertical margin from bottom (in points) to consider a
            printed page number as a footer.
        top_margin: Vertical margin from top to consider a printed page number
            as a header.

    Returns:
        A dict mapping printed-page strings (e.g. '24') to the first physical
        0-based page index where that printed number was observed.

    Example:
        >>> mapping = build_printed_page_map(doc)
        >>> mapping.get('24')
        27
    """
    # Try PDF page labels first (PyMuPDF provides get_page_labels in recent versions)
    mapping: Dict[str, int] = {}
    try:
        if hasattr(doc, "get_page_labels"):
            labels = doc.get_page_labels()
            # Some PyMuPDF versions return strings; others may return dicts.
            # Only use labels if they are human-readable strings.
            if labels and all(isinstance(lab, str) for lab in labels):
                for i, lab in enumerate(labels):
                    if lab:
                        key = str(lab).strip()
                        if key and key not in mapping:
                            mapping[key] = i
                if mapping:
                    return mapping
            # otherwise fall through to scanning visible text spans
    except Exception:
        # Fall back to scanning spans
        pass

    num_re = re.compile(r"^\s*(\d{1,5})\s*$")
    for i in range(doc.page_count):
        page = doc.load_page(i)
        try:
            # PyMuPDF page.get_text is a runtime-only attribute
            data = page.get_text("dict")  # type: ignore[attr-defined]
        except Exception:
            continue
        ph = page.rect.height
        found = False
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    m = num_re.match(txt)
                    if not m:
                        continue
                    bbox = span.get("bbox") or (0, 0, 0, 0)
                    y0, y1 = bbox[1], bbox[3]
                    # Accept numbers near top or bottom of page
                    if (ph - y1) <= bottom_margin or y0 <= top_margin:
                        key = m.group(1)
                        if key not in mapping:
                            mapping[key] = i
                        found = True
                        break
                if found:
                    break
            if found:
                break
    return mapping


def map_printed_to_physical(
    doc: fitz.Document, printed: str, mapping: Optional[Dict[str, int]] = None
) -> Optional[int]:
    """Map a printed page number string (e.g. '24') to a physical 0-based page index.

    Returns None if not found.
    """
    if mapping is None:
        mapping = build_printed_page_map(doc)
    return mapping.get(str(printed).strip())


def find_layout_matches(
    doc: fitz.Document,
    heading_re: str,
    size_threshold_delta: float = 2.0,
    top_y_threshold: int = 150,
) -> List[int]:
    """Return list of 0-based page indices where heading_re appears as a layout-style heading."""
    start_search = 0
    for i in range(min(12, doc.page_count)):
        txt = doc.load_page(i).get_text("text")  # type: ignore[attr-defined]
        if re.search(r"Table\s+of\s+Contents|Contents", txt, re.IGNORECASE):
            start_search = i + 1
            break
    while start_search < doc.page_count and looks_like_toc(
        doc.load_page(start_search).get_text("text")  # type: ignore[attr-defined]
    ):
        start_search += 1

    median_size = compute_median_font_size(doc)
    if median_size <= 0:
        median_size = 10.0

    cre = re.compile(heading_re, re.IGNORECASE)
    found_pages: List[int] = []
    for i in range(start_search, doc.page_count):
        page = doc.load_page(i)
        try:
            data = page.get_text("dict")  # type: ignore[attr-defined]
        except RuntimeError:
            data = {"blocks": []}
        page_has_heading = False
        for block in data.get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    size = float(span.get("size", 0))
                    bbox = span.get("bbox")
                    y0 = bbox[1] if bbox else 0
                    if cre.search(text):
                        if size >= median_size + size_threshold_delta and y0 <= top_y_threshold:
                            page_has_heading = True
                            break
                if page_has_heading:
                    break
            if page_has_heading:
                break
        if page_has_heading:
            found_pages.append(i)
    return found_pages


def extract_pages(input_pdf: str, start: int, end: int, output_pdf: str) -> None:
    """Extract a contiguous range of physical pages and write to a new PDF.

    Args:
        input_pdf: Path to source PDF file.
        start: 0-based start page index (inclusive).
        end: 0-based end page index (inclusive).
        output_pdf: Path to write the resulting PDF.

    Example:
        >>> extract_pages('in.pdf', 23, 27, 'out.pdf')
    """
    doc = fitz.open(input_pdf)
    new = fitz.open()
    for p in range(start, end + 1):
        new.insert_pdf(doc, from_page=p, to_page=p)
    new.save(output_pdf)


def extract_section_by_heading(
    input_pdf: str,
    heading_re: str,
    out_pdf: str,
    occurrence: int = 1,
    size_delta: float = 2.0,
    top_y: int = 150,
    preview: bool = False,
    context_chars: int = 120,
) -> dict:
    """Locate a section by heading (using layout rules) and extract it.

    Returns a dict with either candidate lists when preview=True or the
    extraction result (start_page, end_page, output).
    """
    doc = fitz.open(input_pdf)
    raw = preview_matches(doc, heading_re, context_chars=context_chars)
    layout = find_layout_matches(doc, heading_re, size_threshold_delta=size_delta, top_y_threshold=top_y)
    # try to build printed->physical mapping once for preview
    printed_map = build_printed_page_map(doc)
    if preview:
        # enrich raw matches with any trailing TOC numbers and mapping
        enriched = []
        for p, snippet in raw:
            nums = re.findall(r"(\d{1,5})", snippet)
            toc_num = nums[-1] if nums else None
            mapped = map_printed_to_physical(doc, toc_num, printed_map) if toc_num else None
            enriched.append(
                {
                    "page_index": p + 1,
                    "snippet": snippet,
                    "toc_number": toc_num,
                    "mapped_physical": (mapped + 1) if mapped is not None else None,
                }
            )
        return {
            "raw_matches": enriched,
            "layout_candidates": [p + 1 for p in layout],
            "printed_map_sample": dict(list(printed_map.items())[:10]),
        }
    if not layout:
        raise ValueError("No layout-based heading found; try a different regex or relax thresholds.")
    if occurrence < 1 or occurrence > len(layout):
        raise ValueError("occurrence out of range")
    idx = occurrence - 1
    start0 = layout[idx]
    end0 = (layout[idx + 1] - 1) if (idx + 1) < len(layout) else (doc.page_count - 1)
    extract_pages(input_pdf, start0, end0, out_pdf)
    return {"start_page": start0 + 1, "end_page": end0 + 1, "output": out_pdf}


def extract_pages_from_printed_range(input_pdf: str, start_printed: str, end_printed: str, output_pdf: str) -> dict:
    """Convert printed page numbers to physical pages and extract that range.

    Args:
        input_pdf: Path to source PDF.
        start_printed: Printed start page (string or int).
        end_printed: Printed end page (string or int).
        output_pdf: Output path.

    Returns:
        Dict with keys 'start_page', 'end_page', 'output' (1-based pages in result).
    """
    doc = fitz.open(input_pdf)
    mapping = build_printed_page_map(doc)
    sp = str(start_printed).strip()
    ep = str(end_printed).strip()
    s_phys = mapping.get(sp)
    e_phys = mapping.get(ep)
    if s_phys is None or e_phys is None:
        raise ValueError(
            "Could not map printed range "
            f"{start_printed}..{end_printed} to physical pages; mapping keys sample: "
            f"{list(mapping.keys())[:10]}"
        )
    extract_pages(input_pdf, s_phys, e_phys, output_pdf)
    return {"start_page": s_phys + 1, "end_page": e_phys + 1, "output": output_pdf}


def extract_text_from_pages(input_pdf: str, start: int, end: int) -> str:
    """Extract plain text from a contiguous range of physical pages.

    Args:
        input_pdf: Path to source PDF.
        start: 0-based start page index (inclusive).
        end: 0-based end page index (inclusive).

    Returns:
        A single string with the concatenated page texts.

    Example:
        >>> s = extract_text_from_pages('in.pdf', 27, 27)
    """
    doc = fitz.open(input_pdf)
    parts: List[str] = []
    for p in range(start, end + 1):
        txt = doc.load_page(p).get_text("text")  # type: ignore[attr-defined]
        parts.append(txt)
    return "\n\n".join(parts).strip()


def extract_text_from_printed_range(input_pdf: str, start_printed: str, end_printed: str) -> str:
    """Map printed page numbers to physical pages and return the combined text.

    Args:
        input_pdf: Path to source PDF.
        start_printed: Printed start page (string or int).
        end_printed: Printed end page (string or int).

    Returns:
        Concatenated text content for the mapped physical page range.

    Example:
        >>> text = extract_text_from_printed_range('in.pdf', '24', '28')
    """
    doc = fitz.open(input_pdf)
    mapping = build_printed_page_map(doc)
    sp = str(start_printed).strip()
    ep = str(end_printed).strip()
    s_phys = mapping.get(sp)
    e_phys = mapping.get(ep)
    if s_phys is None or e_phys is None:
        raise ValueError(
            "Could not map printed range "
            f"{start_printed}..{end_printed} to physical pages; mapping keys sample: "
            f"{list(mapping.keys())[:10]}"
        )
    return extract_text_from_pages(input_pdf, s_phys, e_phys)


def extract_text_from_section_by_heading(
    input_pdf: str,
    heading_re: str,
    occurrence: int = 1,
    size_delta: float = 2.0,
    top_y: int = 150,
) -> str:
    """Locate a section by heading (layout rules) and return its plain text.

    This function uses the same layout-based heading detection as
    ``extract_section_by_heading`` but returns the extracted text rather than
    writing a new PDF.

    Args:
        input_pdf: Path to the PDF.
        heading_re: Heading regex.
        occurrence: Which detected layout occurrence to use (1-based).
        size_delta: Font-size delta above median to treat as heading.
        top_y: Maximum y-coordinate for heading spans (top of page).

    Returns:
        Extracted plain text for the section (concatenated pages).

    Example:
        >>> txt = extract_text_from_section_by_heading('in.pdf', r'(?i)Data Privacy')
    """
    doc = fitz.open(input_pdf)
    layout = find_layout_matches(doc, heading_re, size_threshold_delta=size_delta, top_y_threshold=top_y)
    if not layout:
        raise ValueError("No layout-based heading found; try a different regex or preview first.")
    if occurrence < 1 or occurrence > len(layout):
        raise ValueError("occurrence out of range")
    idx = occurrence - 1
    start0 = layout[idx]
    end0 = (layout[idx + 1] - 1) if (idx + 1) < len(layout) else (doc.page_count - 1)
    return extract_text_from_pages(input_pdf, start0, end0)


# (duplicate helper removed – the documented version is defined earlier)


__all__ = [
    "looks_like_toc",
    "compute_median_font_size",
    "preview_matches",
    "build_printed_page_map",
    "map_printed_to_physical",
    "find_layout_matches",
    "extract_pages",
    "extract_section_by_heading",
    "extract_pages_from_printed_range",
]
