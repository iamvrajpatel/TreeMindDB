from __future__ import annotations
import csv,io,json

class SafeTextExtractor:
    allowed={".txt",".md",".markdown",".csv",".json",".pdf"}
    def extract(self, content: bytes, filename: str):
        suffix="."+filename.rsplit(".",1)[-1].lower() if "." in filename else ".txt"
        if suffix not in self.allowed: raise ValueError("Unsupported file type")
        if suffix==".pdf":
            try:
                from pypdf import PdfReader
            except ImportError as exc:
                raise ValueError("PDF support requires the pypdf optional dependency") from exc
            try:
                reader=PdfReader(io.BytesIO(content), strict=False)
                return [(page.extract_text() or "", index + 1) for index,page in enumerate(reader.pages) if page.extract_text()]
            except Exception as exc:
                raise ValueError("Unable to safely extract text from PDF") from exc
        text=content.decode("utf-8",errors="replace")
        if suffix==".json": text=json.dumps(json.loads(text),ensure_ascii=False,indent=2)
        if suffix==".csv": text="\n".join(" | ".join(row) for row in csv.reader(io.StringIO(text)))
        return [(text,None)]

def chunk_pages(pages:list[tuple[str,int|None]], chunk_size:int=800, overlap:int=120):
    chunks=[]
    for text,page in pages:
        words=text.split(); start=0
        while start<len(words):
            part=" ".join(words[start:start+chunk_size]); chunks.append((part,page,len(part.split())))
            if start+chunk_size>=len(words):break
            start+=chunk_size-overlap
    return chunks
