import re
import pymupdf
import copy

DEFAULT_FILENAME = r"ba_data_extraction/ba_documents/Banking Act 1970.pdf"
FOOTER_PATTERN = r"Banking Act 1970 (\d+) 2020 Ed. Informal Consolidation – version in force from \d{1,2}/\d{1,2}/\d{4}|Banking Act 1970 2020 Ed. (\d+) Informal Consolidation – version in force from \d{1,2}/\d{1,2}/\d{4}|Informal Consolidation – version in force from \d{1,2}/\d{1,2}/\d{4} (\d+) 2020 Ed."
PREFIX = 'ba-1970'

class BankingActPreprocessor:
    def __init__(self, filename=None):
        self.filename = filename if filename else DEFAULT_FILENAME
        self.doc = None
        if self.filename:
            self.doc = pymupdf.open(self.filename)
        self.footer_pattern = FOOTER_PATTERN
        self.prefix = PREFIX
        self.page_count = self.doc.page_count

    def get_full_text(self, page_start, page_end):
        full_text = ''
        for page_num in range(page_start, page_end):  
            # Load the page
            page = self.doc.load_page(page_num)  
            # Load text 
            text = page.get_text("text")  
            # Remove line break
            text = re.sub(r'\n+', ' ', text)
            # Remove footer 
            text = re.sub(self.footer_pattern, "", text)
            text = text.strip() + " "

            full_text = full_text + text
        return full_text

    def parse_toc(self, text):
        # Split the string into parts
        parts = re.split(r'PART \d+[A-Z]?', text)[1:]
        part_titles_matches = re.findall(r'PART (\d+[A-Z]?) (.+?)(?=PART \d+[A-Z]?|$)', text, re.DOTALL)
        
        parsed_toc = []
        
        for part_match, part_content in zip(part_titles_matches, parts):
            part_id, part_title = part_match
            
            # Regex to handle complex section IDs like 48AA
            sections = re.findall(r'(\d+[A-Z]*)\.\s*(.+?)(?=\d+[A-Z]*\.|$)', part_content.strip(), re.DOTALL)
            # Extract only the uppercase letters which denote the part title 
            match = re.search(r'\b[A-Z\s,.]+\b', part_title)
            part_title = match.group() if match else part_title
            
            for section_id, section_title in sections:
                section_title = section_title.strip()
                # Remove 'Section' placeholder at the end of section title 
                section_title = re.sub(r'\bSection\b$', '', section_title)
                # Remove Division from section_title
                section_title = re.sub(r'Division \d+ —.*', '', section_title)
                # Clean last section to exclude unnecessary text
                if section_id == '79':
                    section_title = re.split(r' First Schedule', section_title)[0]
                
                parsed_toc.append({
                    "part_id": part_id.strip(),
                    "part_title": part_title.strip(),
                    "section_id": section_id.strip(),
                    "section_title": section_title.strip()
                })
        
        return parsed_toc
    
    def split_subsections(self, text, toc_list):
        # Split by parts 
        part_pattern = re.split(r'(?=PART \d+)', text)

        output = {'chunks': [], 
                'chunks_length': [], 
                'ids': [], 
                'section_dict': {}}

        id_prefix = self.prefix
        regulation = 'Banking Act 1970'

        for part_text in part_pattern[1:12]:
            part_id = part_text.split(' ')[1]
            toc_sections = [item for item in toc_list if item['part_id'] == part_id]
            n_sections = len(toc_sections)
            
            # Split by sections
            for id, toc_section in enumerate(toc_sections):
                section_id = toc_section["section_id"]
                section_title = toc_section["section_title"]

                if id+1 < n_sections: 
                    next_section_id = toc_sections[id+1]["section_id"]
                    next_section_title = toc_sections[id+1]["section_title"]

                    if ('Repealed' not in section_title) and ('Repealed' in next_section_title):
                        match = re.search(rf'{section_title} {section_id}\.(.*?){next_section_id}\.', part_text)
                    elif ('Repealed' in section_title) and ('Repealed' not in next_section_title):
                        match = re.search(rf'{section_id}\.(.*?){next_section_title} {next_section_id}\.', part_text)
                    elif ('Repealed' in section_title) and ('Repealed' in next_section_title):
                        match = re.search(rf'{section_id}\.(.*?){next_section_id}\.', part_text)
                    else: 
                        match = re.search(rf'{section_title} {section_id}\.(.*?){next_section_title} {next_section_id}\.', part_text)

                # handle last section        
                else:  
                    if ('Repealed' in section_title):
                        match = re.search(rf'{section_id}\.(.*)', part_text)
                    else: 
                        match = re.search(rf'{section_title} {section_id}\.(.*)', part_text)

                section_text = match.group(1).strip()
                section_text = f'{section_id}. {section_text}'

                if section_id == '79':
                        section_text = re.split(r' FIRST SCHEDULE', section_text)[0]

                delimiters = re.findall(r'(?:\. |—|] )\(\d+[A-Za-z]?\)', section_text)
                subsections_list = [x.replace('—','').replace('.','').replace(']', '').strip() for x in delimiters]

                # Chunk by subsection
                toc_section['regulation'] = regulation
                toc_section['references'] = []
                toc_section['part_title'] = toc_section['part_title'].capitalize()
                toc_section['section_id'] = section_id.lower()

                #If no subsection, take section as the smallest chunk
                if len(subsections_list) == 0: 
                    chunk_id = (id_prefix + '-' + section_id).lower()
                    text = f"Part {toc_section['part_id']}. {toc_section['part_title']}\nSection {toc_section['section_id']}. {toc_section['section_title']}\n{section_text}"
                    chunk = {
                        "id": chunk_id, 
                        "text": text,
                        "metadata": toc_section
                    }

                    raw_sid = chunk['metadata']['section_id']
                    formatted_sid = f'{id_prefix}-{raw_sid}'
                    if formatted_sid not in output['section_dict'].keys():
                        output['section_dict'][formatted_sid] = []

                    output['section_dict'][formatted_sid].append(chunk['id'])
                    output['chunks'].append(chunk)
                    output['chunks_length'].append(len(section_text))
                    output['ids'].append(chunk_id)
                else:
                    for i in range(len(subsections_list)):
                        delimiter_start = delimiters[i]
                        subsection_start = subsections_list[i]

                        if i+1 < len(subsections_list):
                            delimiter_end = delimiters[i+1]
                            subsection_end = subsections_list[i+1]

                            match = re.search(rf'({re.escape(delimiter_start)}.*?{re.escape(delimiter_end)})', section_text)
                            subsection_text = match.group(1).strip()
                            
                            subsection_text = subsection_text.replace(delimiter_start, subsection_start)
                            subsection_text = re.sub(rf'\s*{re.escape(subsection_end)}$', '', subsection_text).strip() 
                        else: 
                            match = re.search(rf'{re.escape(delimiter_start)}(.*)', section_text)
                            subsection_text = match.group(1).strip()
                            subsection_text = f'{subsection_start} {subsection_text}'
                        
                        subsection_text = section_id + subsection_text
                        subsection_id = subsection_start.replace("(", "").replace(")", "")
                        chunk_id = (f'{id_prefix}-{section_id}.{subsection_id}').lower()

                        text = f"Part {toc_section['part_id']}. {toc_section['part_title']}\nSection {toc_section['section_id']}. {toc_section['section_title']}\n{subsection_text}"
                        chunk = {
                            "id": chunk_id, 
                            "text": text,
                            "metadata": toc_section
                        }

                        raw_sid = chunk['metadata']['section_id']
                        formatted_sid = f'{id_prefix}-{raw_sid}'
                        if formatted_sid not in output['section_dict'].keys():
                            output['section_dict'][formatted_sid] = []

                        output['section_dict'][formatted_sid].append(chunk['id'])
                        output['chunks'].append(chunk)
                        output['chunks_length'].append(len(subsection_text))
                        output['ids'].append(chunk_id)
        return output
    
    def extract_references(self, text):
        keywords = ['section', 'sections', 'subsection', 'subsections']
        split_text = text.split(' ')
        references = []
        for i, word in enumerate(split_text): 
            if word.lower() in keywords:
                # Check if next word is a section/subsection number 
                save_reference = word
                for j, word in enumerate(split_text[i+1:]):
                    pattern = r"^(?:\d+[A-Z]{0,2}(\(\d+[A-Z]{0,2}\))?(\([a-z]\))?(\(\b[ivxlcdm]+\b\))?(\(\b[A-Z]+\b\))?$)|((\(\d+[A-Z]{0,2}\))(\([a-z]\))?(\(\b[ivxlcdm]+\b\))?(\(\b[A-Z]+\b\))?$)|((\([a-z]\))(\(\b[ivxlcdm]+\b\))?(\(\b[A-Z]+\b\))?$)|^(and|or|to)$"
                    cleaned_word = re.sub(r"(,|;|\.)?\s*$", "", word).strip()
                    if bool(re.fullmatch(pattern, cleaned_word)):
                        save_reference = ' '.join([save_reference, word])
                    elif j == 0: # if the first next word doesn't match, it's standalone keyword (e.g: 'in this section')
                        save_reference = ''
                        break
                    elif cleaned_word == 'of': # reference to other Act 
                        save_reference = ''
                    else: 
                        break
                save_reference = re.sub(r"(,|, or|;|\.|or|and|to|, to)?\s*$", "", save_reference, flags=re.IGNORECASE).strip()
                if save_reference.lower() not in [''] + keywords:
                    references.append(save_reference)
        references = list(dict.fromkeys(references))
        return references
    
    def truncate_references(self, references):
        output = []
        for text in references:
            # only keep until subsection e.g: 15B(1)(a)(i)(A) -> 15B(1)
            cleaned_text = re.sub(r'\(\b[a-z]+\b\)', '', text)  # Remove lowercase letters in brackets
            cleaned_text = re.sub(r'\(\b[ivxlcdm]+\b\)', '', cleaned_text, flags=re.IGNORECASE)  # Remove Roman numerals in brackets
            cleaned_text = re.sub(r'\(\b[A-Z]+\b\)', '', cleaned_text, flags=re.IGNORECASE)  # Remove capital letter in brackets
            # Clean trailing commas and whitespaces
            cleaned_text = re.sub(r'( ,)', '', cleaned_text)
            cleaned_text = re.sub(r'(,|,\s+(and|or|to)|;|\.|or|and|to|, to)?\s*$', '', cleaned_text, flags=re.IGNORECASE)
            cleaned_text = cleaned_text.replace('  ', ' ')
            output.append(cleaned_text.strip())  # Remove extra spaces
        return output
    
    def insert_sections(self, references):
        expanded_section = []
        for reference in references: 
            keywords = ['section', 'sections'] 
            split_text = reference.split(' ')

            # skip if it is not a section 
            if split_text[0].lower() not in keywords:
                expanded_section.append(reference)
                continue

            section_id = '' # store the section_id to be inserted to standalone subsection_id 
            expanded_text = [split_text[0]]
            for word in split_text[1:]:
                cleaned_word = re.sub(r"(,|;|\.)?\s*$", "", word).strip()
                pattern_section = r"^(?:\d+[A-Z]{0,2}|(\d+[A-Z]{0,2})(\(\d+[A-Z]{0,2}\))?(\([a-z]\))?(\(\b[ivxlcdm]+\b\))?(\(\b[A-Z]+\b\))?$)"
                match_section = re.match(pattern_section, cleaned_word)

                pattern_subsection = r"(?:\(\d+[A-Z]{0,2}\)|\(\d+[A-Z]{0,2}\)(\([a-z]\))?(\(\b[ivxlcdm]+\b\))?(\(\b[A-Z]+\b\))?$)" 
                match_subsection = re.match(pattern_subsection, cleaned_word)

                if match_section: 
                    section_id = match_section.group(0)
                    expanded_text.append(word) #section doesn't change 
                elif match_subsection:
                    subsection_id = match_subsection.group(0)
                    subsection_id = section_id + subsection_id
                    expanded_text.append(subsection_id) #inserted section_id 
                else: 
                    expanded_text.append(word)
            expanded_text = ' '.join(expanded_text)
            expanded_section.append(expanded_text)
        return expanded_section
    
    def reformat_references(self, references, section_id):
        result = []
        prefix = self.prefix
        
        for reference in references:
            # Find all patterns that resemble section/subsection identifiers
            reference = reference.lower()

            # Reformat section 15b(1) -> 15b.1
            if 'subsection' not in reference: 
                pattern = r'(\d+[a-z]*)(?:\((\d+[a-z]*)\))?'
                matches = re.findall(pattern, reference)
                
                for match in matches:
                    section_part = match[0].lower()  # Convert section part to lowercase
                    subsection_part = match[1] if match[1] else None 

                    # If subsection exists, format accordingly
                    if subsection_part:
                        result.append(f'{prefix}-{section_part}.{subsection_part.lower()}')
                    else:
                        result.append(f'{prefix}-{section_part}')
            else: 
                subsections_id = re.findall(r'\((.*?)\)', reference)
                for sid in subsections_id:
                    result.append(f'{prefix}-{section_id}.{sid}'.lower())
        return result
    
    def map_sections(self, list_references, section_dict):
        final_ref = []
        for ref in list_references:
            if ref in section_dict.keys():
                for id in section_dict[ref]:
                    final_ref.append(id)
            else: 
                final_ref.append(ref)
        final_ref = list(dict.fromkeys(final_ref)) # remove duplicates 
        return final_ref
    
    def get_references(self, raw_chunks, section_dict):
        chunks = []
        for chunk in raw_chunks: 
            text = chunk['text']
            id = chunk['id']
            section_id = chunk['metadata']['section_id']

            # Process references
            raw_references = self.extract_references(text)
            truncated_references = self.truncate_references(raw_references)
            expanded_references = self.insert_sections(truncated_references)
            formatted_references = self.reformat_references(expanded_references, section_id=section_id)
            final_references = self.map_sections(formatted_references, section_dict)
            
            # Ensure the updated chunk is correctly stored
            chunk['metadata']['references'] = final_references
            chunks.append(copy.deepcopy(chunk))  
        return chunks
    
    def get_toc(self):
        toc_text = self.get_full_text(1, 8)
        toc_list = self.parse_toc(toc_text)
        return toc_list

    def get_chunks(self):
        toc_list = self.get_toc()
        contents_text = self.get_full_text(8, self.page_count)

        subsections_dict = self.split_subsections(contents_text, toc_list)
        raw_chunks = subsections_dict['chunks']
        section_dict = subsections_dict['section_dict']

        final_chunks = self.get_references(raw_chunks, section_dict)
        return final_chunks
        
if __name__ == "__main__":
    preprocessor_ba_1970 = BankingActPreprocessor()
    final_chunks = preprocessor_ba_1970.get_chunks() 
        