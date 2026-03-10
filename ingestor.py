"""
ingestor.py - URL Scraper for GOIES
Fetches an article from a URL and extracts its text content.
"""

import urllib.request
import urllib.error
from html.parser import HTMLParser

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_content = []
        self.ignore_tags = {"script", "style", "nav", "footer", "header", "aside", "noscript", "meta", "link"}
        self.in_ignored = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.ignore_tags:
            self.in_ignored += 1

    def handle_endtag(self, tag):
        if tag in self.ignore_tags:
            self.in_ignored = max(0, self.in_ignored - 1)

    def handle_data(self, data):
        if self.in_ignored == 0:
            text = data.strip()
            if text:
                self.text_content.append(text)

def fetch_url_text(url: str) -> str:
    """
    Fetches HTML from a URL and parses out the readable text content.
    """
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8', errors='ignore')
            parser = TextExtractor()
            parser.feed(html)
            
            # Simple heuristic: join paragraphs with double newlines
            return "\n\n".join(parser.text_content)
    except Exception as e:
        raise ValueError(f"Failed to fetch or parse URL: {e}")

def parse_pdf(file_bytes: bytes) -> str:
    import io
    import PyPDF2
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = []
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text.append(extracted)
    return "\n\n".join(text)

def parse_docx(file_bytes: bytes) -> str:
    import io
    import docx
    doc = docx.Document(io.BytesIO(file_bytes))
    return "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
