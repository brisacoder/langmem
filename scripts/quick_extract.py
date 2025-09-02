#!/usr/bin/env python3
import sys
import fitz

def extract_pages(input_pdf, start_1based, end_1based, output_pdf):
    start = start_1based - 1
    end = end_1based - 1
    doc = fitz.open(input_pdf)
    if start < 0 or end >= doc.page_count or start > end:
        raise ValueError(f"Invalid page range {start_1based}-{end_1based} for document with {doc.page_count} pages")
    new = fitz.open()
    for p in range(start, end + 1):
        new.insert_pdf(doc, from_page=p, to_page=p)
    new.save(output_pdf)

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: quick_extract.py input.pdf start_page end_page output.pdf")
        sys.exit(2)
    inp = sys.argv[1]
    try:
        s = int(sys.argv[2])
        e = int(sys.argv[3])
    except ValueError:
        print("start_page and end_page must be integers (1-based)")
        sys.exit(2)
    outp = sys.argv[4]
    print(f"Extracting pages {s}..{e} from {inp} to {outp}")
    extract_pages(inp, s, e, outp)
    print("Done")
