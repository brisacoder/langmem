# Package marker for scripts
# Intentionally small — scripts exposes utility functions for notebooks and demos.
from .extract_pdf_section import (
    looks_like_toc,
    compute_median_font_size,
    preview_matches,
    find_layout_matches,
    extract_pages,
    extract_section_by_heading,
)
