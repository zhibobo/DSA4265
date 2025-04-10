#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import fitz 
import re
from typing import List, Dict
import os


class MASPreprocessor:
    def __init__(self):
        self.notice_path = "mas_documents/MAS Notice 626 dated 28 March 2024.pdf"
        self.guideline_path = "mas_documents/Guidelines to MAS Notice 626 March 2024 - Final.pdf"
        self.fair_dealing_path = "mas_documents/Fair Dealing Guidelines 30 May 2024.pdf"

    def get_chunks(self) -> List[Dict]:
        chunks_notice = self.chunk_mas_notice(self.notice_path)
        chunks_notice = self.add_references_to_notice(chunks_notice)

        chunks_guidelines = self.chunk_mas_guidelines(self.guideline_path)
        chunks_guidelines = self.add_references_to_guidelines(chunks_guidelines, chunks_notice)

        chunks_fair_dealing = self.chunk_fair_dealing(self.fair_dealing_path)

        return chunks_notice + chunks_guidelines + chunks_fair_dealing

    def chunk_mas_notice(self, pdf_path: str, start_page: int = 0, end_page: int = None) -> List[Dict]:
        doc = fitz.open(pdf_path)
        if end_page is None:
            end_page = len(doc)

        chunks = []
        section_id_pattern = re.compile(r"^(\d+[A-Z]?\.\d+[A-Z]?)\b")

        current_part_id = None
        current_part_title = None
        current_section_id = None
        current_section_title = None
        current_text = []

        last_bold_line = None

        def flush_section():
            if current_section_id and current_text:
                chunks.append({
                    "id": f"mas-notice-626-{current_section_id}",
                    "text": " ".join(current_text).strip(),
                    "metadata": {
                        "part_id": current_part_id,
                        "part_title": current_part_title,
                        "section_id": current_section_id,
                        "section_title": current_section_title or "",
                        "regulation": "MAS Notice 626",
                        "references": []
                    }
                })

        for page_num in range(start_page, end_page):
            page = doc[page_num]
            lines_raw = page.get_text("dict")["blocks"]

            layout_lines = []
            for block in lines_raw:
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    text_combined = " ".join(span["text"].strip() for span in spans if span["text"].strip())
                    if not text_combined:
                        continue
                    layout_lines.append({
                        "text": text_combined,
                        "bold": any("Bold" in span["font"] for span in spans),
                        "x0": spans[0]["bbox"][0],
                    })

            plain_lines = page.get_text("text").splitlines()

            for i, line in enumerate(plain_lines):
                stripped = line.strip()
                matching_layout = next((l for l in layout_lines if l["text"] == stripped), None)
                if (
                    matching_layout and
                    matching_layout["bold"] and
                    stripped != stripped.upper() and
                    not re.match(r"^\d+$", stripped)
                ):
                    last_bold_line = stripped
                    continue

                if re.match(r"^\d+[A-Z]?$", stripped) and i + 1 < len(plain_lines) and plain_lines[i + 1].strip().isupper():
                    flush_section()
                    current_part_id = stripped
                    current_part_title = plain_lines[i + 1].strip()
                    current_section_id = None
                    current_section_title = None
                    current_text = []
                    last_bold_line = None
                    continue

                match = section_id_pattern.match(stripped)
                if match:
                    flush_section()
                    current_section_id = match.group(1)
                    if last_bold_line:
                        current_section_title = last_bold_line
                        last_bold_line = None
                    else:
                        current_section_title = (
                            chunks[-1]["metadata"]["section_title"]
                            if chunks and chunks[-1]["metadata"]["part_id"] == current_part_id
                            else ""
                        )
                    current_text = [stripped]
                    continue

                if current_section_id:
                    current_text.append(stripped)

        flush_section()
        return chunks

    def chunk_mas_guidelines(self, pdf_path: str, start_page: int = 2) -> List[Dict]:
        doc = fitz.open(pdf_path)
        chunks = []

        section_id_pattern = re.compile(r"^(\d+-\d+)\b")
        subsection_pattern = re.compile(r"^(\d+-\d+-\d+)\b")
        part_id_pattern = re.compile(r"^\d+$")
        page_header_phrases = [
            "GUIDELINES TO MAS NOTICE 626",
            "COUNTERING THE FINANCING OF TERRORISM"
        ]

        all_lines = []
        for page_num in range(start_page, len(doc)):
            page = doc[page_num]
            lines = [line.strip() for line in page.get_text("text").splitlines()]
            lines = [
                line for line in lines
                if line and not any(header.lower() in line.lower() for header in page_header_phrases)
            ]
            all_lines.extend(lines)

        part_positions = {}
        for i in range(len(all_lines) - 1):
            if part_id_pattern.match(all_lines[i]):
                next_line = all_lines[i + 1]
                if (
                    next_line[0].isupper()
                    and len(next_line.split()) > 1
                    and not any(header.lower() in next_line.lower() for header in page_header_phrases)
                ):
                    part_positions[i] = (all_lines[i], next_line)

        last_seen_part_id = "1"
        last_seen_part_title = "Introduction"

        current_section_id = None
        current_section_title = None
        inherited_section_title = ""
        current_text = []
        potential_title = None

        def flush():
            nonlocal current_section_id, current_section_title, inherited_section_title, current_text
            if current_section_id and current_text:
                section_title_to_use = current_section_title or inherited_section_title or ""
                if section_title_to_use == last_seen_part_title:
                    section_title_to_use = ""
                chunks.append({
                    "id": f"mas-guidelines-626-{current_section_id}",
                    "text": " ".join(current_text).strip(),
                    "metadata": {
                        "part_id": last_seen_part_id,
                        "part_title": last_seen_part_title,
                        "section_id": current_section_id,
                        "section_title": section_title_to_use,
                        "regulation": "MAS Guidelines to Notice 626",
                        "references": []
                    }
                })
            inherited_section_title = ""
            current_section_id = None
            current_section_title = None
            current_text = []

        i = 0
        while i < len(all_lines):
            if i in part_positions:
                flush()
                last_seen_part_id, last_seen_part_title = part_positions[i]
                inherited_section_title = ""

            line = all_lines[i]

            if (
                not section_id_pattern.match(line)
                and not subsection_pattern.match(line)
                and not part_id_pattern.match(line)
                and len(line.split()) <= 8
                and line[0].isupper()
            ):
                for lookahead in range(1, 4):
                    if i + lookahead < len(all_lines) and section_id_pattern.match(all_lines[i + lookahead]):
                        if line == last_seen_part_title:
                            potential_title = ""
                            inherited_section_title = ""
                        else:
                            potential_title = line
                        break
                i += 1
                continue

            match_section = section_id_pattern.match(line)
            if match_section and not subsection_pattern.match(line):
                flush()
                current_section_id = match_section.group(1)
                current_section_title = potential_title or ""
                inherited_section_title = current_section_title if current_section_title else ""
                potential_title = None
                current_text = [line]
                i += 1
                continue

            if current_section_id:
                current_text.append(line)

            i += 1

        flush()
        return chunks

    def chunk_fair_dealing(self, pdf_path: str,  start_page: int = 5) -> List[Dict]:
        import copy
        
        doc = fitz.open(pdf_path)
        chunks = []
        
        # Patterns
        part_pattern = re.compile(r"^(\d+)\s+Fair Dealing Outcome", re.IGNORECASE)
        section_title_pattern = re.compile(r"^(\d+\.\d+)\s+(.+)")
        section_id_pattern = re.compile(r"^(\d+\.\d+\.\d+)\b")
        practice_start_pattern = re.compile(r"^(Good|Poor) practice (\d+\.\d+)", re.IGNORECASE)
        
        current_part_id = None
        current_part_title = None
        current_section_id = None
        current_section_title = None
        current_text = []
        current_section_type = "main"
        temp_title_tracker = None
        expecting_part_subtitle = False
        active_chunk = None
        
        def flush_active_chunk():
            nonlocal active_chunk, current_text
            if active_chunk:
                active_chunk["text"] = " ".join(current_text).strip()
                chunks.append(active_chunk)
                active_chunk = None
                current_text = []
        
        def start_new_chunk(section_id, title=""):
            return {
                "id": f"mas-fair-dealing-{section_id}",
                "text": "",
                "metadata": {
                    "part_id": current_part_id,
                    "part_title": current_part_title,
                    "section_id": section_id,
                    "section_title": title,
                    "regulation": "MAS Fair Dealing Guidelines",
                    "references": []
                }
            }
        
        for page_num in range(start_page, len(doc)):
            page = doc[page_num]
            blocks = [b for b in page.get_text("dict")["blocks"] if "lines" in b]
        
            for block in blocks:
                line_text = " ".join(
                    span["text"].strip()
                    for line in block["lines"]
                    for span in line.get("spans", [])
                    if span["text"].strip()
                ).strip()
        
                if not line_text:
                    continue
        
                # Detect Part Header
                match_part = part_pattern.match(line_text)
                if match_part:
                    flush_active_chunk()
                    current_part_id = match_part.group(1)
                    current_part_title = line_text
                    expecting_part_subtitle = True
                    continue
        
                # Detect subtitle after part
                if expecting_part_subtitle:
                    if any("bold" in span["font"].lower() for line in block["lines"] for span in line["spans"]):
                        current_part_title += " " + line_text
                        expecting_part_subtitle = False
                        continue
        
                # Section title (e.g., 4.2 Title)
                match_title = section_title_pattern.match(line_text)
                if match_title:
                    temp_title_tracker = line_text.strip()
                    continue
        
                # Section ID (e.g., 4.2.1)
                match_section = section_id_pattern.match(line_text)
                if match_section:
                    flush_active_chunk()
                    current_section_id = match_section.group(1)
                    section_prefix = ".".join(current_section_id.split(".")[:2])
                    if temp_title_tracker and temp_title_tracker.startswith(section_prefix):
                        current_section_title = temp_title_tracker
                    active_chunk = start_new_chunk(current_section_id, current_section_title)
                    current_text = [line_text]
                    current_section_type = "main"
                    continue
        
                # Practice block (Good or Poor)
                match_practice = practice_start_pattern.match(line_text)
                if match_practice:
                    flush_active_chunk()
                    kind, number = match_practice.groups()
                    practice_id = f"{kind.lower()}-practice-{number}"
                    current_section_id = practice_id
                    current_section_type = kind.lower()
                    active_chunk = start_new_chunk(practice_id, current_section_title)
                    current_text = [line_text]
                    continue
        
                # Regular paragraph inside active chunk
                if active_chunk:
                    current_text.append(line_text)
        
        flush_active_chunk()
        return chunks

    def add_references_to_notice(self, notice_chunks: List[Dict]) -> List[Dict]:
        id_set = {c["id"] for c in notice_chunks}
        part_map = {}
        for c in notice_chunks:
            pid = c["metadata"].get("part_id")
            if pid:
                part_map.setdefault(pid, []).append(c["id"])
    
        def get_references(text: str, chunk_id: str) -> List[str]:
            refs = set()
    
            # (1) paragraphs 6, 7 and 8 or 6.1, 6.2 and 6.3 FIRST
            for match in re.findall(r"paragraphs?\s+((?:\d+(?:\.\d+[A-Z]?)?|\d+)(?:[ ,and]+(?:\d+(?:\.\d+[A-Z]?)?|\d+))*)", text, re.IGNORECASE):
                all_nums = re.split(r"[,\s]+and\s+|,\s*|\s+and\s+", match)
                for num in all_nums:
                    num = num.strip()
                    if not num:
                        continue
                    if re.match(r"\d+\.\d+[A-Z]?$", num):
                        ref_id = f"mas-notice-626-{num}"
                        if ref_id in id_set:
                            refs.add(ref_id)
                    elif num in part_map:
                        refs.update(part_map[num])
    
            # (2) paragraph x.y or x.yA (no bracket)
            for match in re.findall(r"paragraphs?\s+(\d+\.\d+[A-Z]?)(?!\(|[A-Z])\b", text, re.IGNORECASE):
                ref_id = f"mas-notice-626-{match}"
                if ref_id in id_set:
                    refs.add(ref_id)
    
            # (3) paragraph x.yA(a) → only match x.yA
            for match in re.findall(r"paragraphs?\s+(\d+\.\d+[A-Z]?)\([a-z]\)", text, re.IGNORECASE):
                ref_id = f"mas-notice-626-{match}"
                if ref_id in id_set:
                    refs.add(ref_id)
    
            # (4) paragraph x (single number → full part)
            for match in re.findall(r"paragraphs?\s+(\d+)(?![\.-])", text, re.IGNORECASE):
                if match in part_map:
                    refs.update(part_map[match])
    
            # (5) paragraphs x.y to x.z (ignore letters for range)
            for match in re.findall(r"paragraphs?\s+(\d+)\.(\d+)\s+to\s+(\d+)\.(\d+)", text, re.IGNORECASE):
                part1, start, part2, end = match
                if part1 == part2:
                    for i in range(int(start), int(end) + 1):
                        sec_id = f"{part1}.{i}"
                        ref_id = f"mas-notice-626-{sec_id}"
                        if ref_id in id_set:
                            refs.add(ref_id)
    
            refs.discard(chunk_id)
            return sorted(refs)
    
        for chunk in notice_chunks:
            chunk["metadata"]["references"] = get_references(chunk["text"], chunk["id"])
    
        return notice_chunks


    def add_references_to_guidelines(self, guideline_chunks: List[Dict], notice_chunks: List[Dict]) -> List[Dict]:
        id_set_notice = {c["id"] for c in notice_chunks}
        id_set_guideline = {c["id"] for c in guideline_chunks}
        part_map_notice = {}
        for c in notice_chunks:
            part_id = c["metadata"].get("part_id")
            if part_id:
                part_map_notice.setdefault(part_id, []).append(c["id"])
        
        def get_references(chunk: Dict) -> List[str]:
            refs = set()
        
            section_id = chunk["metadata"]["section_id"]
            section_title = chunk["metadata"].get("section_title", "")
            text = chunk["text"]
            full_text = f"{section_title} {text}"
        
            # Match MAS Notice section by direct conversion
            dot_section_id = section_id.replace("-", ".")
            direct_id = f"mas-notice-626-{dot_section_id}"
            if direct_id in id_set_notice:
                refs.add(direct_id)
        
            # Paragraph patterns
            # (1) paragraph 4.1
            for match in re.findall(r"paragraphs? (\d+\.\d+)(?!\()", full_text, re.IGNORECASE):
                ref_id = f"mas-notice-626-{match}"
                if ref_id in id_set_notice:
                    refs.add(ref_id)
        
            # (2) paragraph 4.1(a)
            for match in re.findall(r"paragraphs? (\d+\.\d+)\([a-z]\)", full_text, re.IGNORECASE):
                ref_id = f"mas-notice-626-{match}"
                if ref_id in id_set_notice:
                    refs.add(ref_id)
        
            # (3) paragraph 4-1
            for match in re.findall(r"paragraphs? (\d+-\d+)(?!-\d)", full_text, re.IGNORECASE):
                ref_id = f"mas-guidelines-626-{match}"
                if ref_id in id_set_guideline:
                    refs.add(ref_id)
        
            # (4) paragraph 4-1-2 → 4-1
            for match in re.findall(r"paragraphs? (\d+-\d+)-\d+", full_text, re.IGNORECASE):
                ref_id = f"mas-guidelines-626-{match}"
                if ref_id in id_set_guideline:
                    refs.add(ref_id)
        
            # (5) paragraph 6 → whole part
            for match in re.findall(r"paragraphs? (\d+)(?![\.-])", full_text, re.IGNORECASE):
                part_id = match
                refs.update(part_map_notice.get(part_id, []))
        
            # (6) paragraphs 6, 7 and 8
            for match in re.findall(r"paragraphs? ((?:\d+(?:\.\d+)?(?:, )?)+(?:and \d+(?:\.\d+)?))", full_text, re.IGNORECASE):
                nums = re.findall(r"\d+(?:\.\d+)?", match)
                for num in nums:
                    if "." in num:
                        ref_id = f"mas-notice-626-{num}"
                        if ref_id in id_set_notice:
                            refs.add(ref_id)
                    else:
                        refs.update(part_map_notice.get(num, []))
        
            # (7) paragraphs 6.1 to 6.5
            for match in re.findall(r"paragraphs? (\d+)\.(\d+) to (\d+)\.(\d+)", full_text, re.IGNORECASE):
                part1, start, part2, end = match
                if part1 == part2:
                    for i in range(int(start), int(end) + 1):
                        sec_id = f"{part1}.{i}"
                        ref_id = f"mas-notice-626-{sec_id}"
                        if ref_id in id_set_notice:
                            refs.add(ref_id)
        
            refs.discard(chunk["id"])
            return sorted(refs)
        
        # Apply to all
        for chunk in guideline_chunks:
            chunk["metadata"]["references"] = get_references(chunk)
        
        return guideline_chunks

