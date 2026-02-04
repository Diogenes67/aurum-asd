#!/usr/bin/env python3
"""
ASD Workflow Backend Server v0.27 - Flask Production Version
Serves the web app and handles API calls
- Llama 3.3 70B via HuggingFace API
- Full context (128K window)
"""

VERSION = "0.27"

import json
import urllib.request
import os
import re
import time
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.', static_url_path='')

# Get HF token from environment
HF_TOKEN = os.environ.get('HF_TOKEN', '')

# ============== Static File Serving ==============

@app.route('/')
def index():
    return send_from_directory('.', 'asd_workflow_aurum.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)

# ============== API Endpoints ==============

@app.route('/api/status', methods=['POST'])
def handle_status():
    """Check if Ollama is running and which models are available"""
    try:
        req = urllib.request.Request('http://localhost:11434/api/tags')
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m['name'] for m in data.get('models', [])]
            return jsonify({'status': 'ok', 'models': models})
    except Exception as e:
        return jsonify({'status': 'error', 'error': str(e)})

@app.route('/api/prescan-batch', methods=['POST'])
def handle_prescan_batch():
    """Hybrid prescan: filename matching first, then AI for unresolved items"""
    try:
        data = request.get_json()
        documents = data.get('documents', {})
        use_cloud = data.get("useCloud", False)

        print(f"\n[PreScan] {len(documents)} documents")
        for fname in documents.keys():
            print(f"  - {fname}")

        # Initialize metadata
        metadata = {
            'gp_referral': {'status': 'missing', 'source': None},
            'hearing_test': {'status': 'not_done', 'source': None},
            'dev_history': {'status': 'missing', 'source': None},
            'teacher_input': {'status': 'missing', 'source': None},
            'previous_asd': {'status': 'none', 'source': None}
        }

        # PASS 1: Filename + content keyword matching (instant)
        print("[PreScan] Pass 1: Filename matching...")

        for filename, text in documents.items():
            lower_name = filename.lower()
            lower_text = text.lower()

            # GP Referral
            if any(kw in lower_name for kw in ['gp', 'referral', 'paed']):
                metadata['gp_referral'] = {'status': 'present', 'source': filename}

                # Check for hearing test in GP referral content
                if any(kw in lower_text for kw in ['bera', 'abr', 'audiometry', 'audiolog', 'hearing screen', 'peripheral hearing', 'hearing test', 'brainstem auditory']):
                    if any(kw in lower_text for kw in ['normal', 'passed', 'within normal', 'no concerns']):
                        metadata['hearing_test'] = {'status': 'normal', 'source': filename}
                    else:
                        metadata['hearing_test'] = {'status': 'concerns', 'source': filename}

                # Check for developmental history
                if any(kw in lower_text for kw in ['milestone', 'developmental history', 'birth history', 'pregnancy', 'early development', 'developmental concerns']):
                    if metadata['dev_history']['status'] == 'missing':
                        metadata['dev_history'] = {'status': 'present', 'source': filename}

            # Teacher/School input
            if any(kw in lower_name for kw in ['teacher', 'school', 'educator', 'kindergarten']):
                metadata['teacher_input'] = {'status': 'present', 'source': filename}

            # Other reports with developmental history
            if any(kw in lower_name for kw in ['speech', 'psych', 'social', 'worker', 'ot', 'occupational']):
                if any(kw in lower_text for kw in ['developmental history', 'milestone', 'birth', 'early development', 'background information']):
                    if metadata['dev_history']['status'] == 'missing':
                        metadata['dev_history'] = {'status': 'present', 'source': filename}

            # Previous ASD assessment mentioned
            if any(kw in lower_text for kw in ['previous autism', 'prior asd', 'previously assessed for autism', 'earlier autism assessment']):
                if metadata['previous_asd']['status'] == 'none':
                    metadata['previous_asd'] = {'status': 'missing', 'source': None}

        # Check what's still missing after Pass 1
        missing_items = []
        if metadata['gp_referral']['status'] == 'missing':
            missing_items.append('gp_referral')
        if metadata['hearing_test']['status'] == 'not_done':
            missing_items.append('hearing_test')
        if metadata['dev_history']['status'] == 'missing':
            missing_items.append('dev_history')
        if metadata['teacher_input']['status'] == 'missing':
            missing_items.append('teacher_input')

        print(f"[PreScan] Pass 1 results: {len(missing_items)} items unresolved: {missing_items}")

        # PASS 2: AI analysis only if critical items still missing
        if missing_items and len(documents) > 0:
            print(f"[PreScan] Pass 2: AI analysis for unresolved items...")

            # Build combined text
            combined = ""
            filenames_list = list(documents.keys())
            for filename, text in documents.items():
                combined += f"\n\n=== {filename} ===\n{text[:4000]}\n"

            prompt = f"""<start_of_turn>user
I have these clinical documents for an autism assessment:
{combined}

The filenames are: {', '.join(filenames_list)}

I need to find these specific items that were not detected by filename:
{', '.join(missing_items)}

For each item, tell me:
- gp_referral: Is there a GP or paediatrician referral letter?
- hearing_test: Is there an audiological hearing test result (BERA, ABR, audiometry)?
- dev_history: Is there developmental history (milestones, birth history, early concerns)?
- teacher_input: Is there a teacher report or school observation?

Return JSON with ONLY the items I asked about. Use EXACT filenames from the list above.
Example: {{"teacher_input": {{"status": "present", "source": "Exact_Filename.docx"}}}}

Status options: "present", "missing", "normal" (for hearing), "not_done" (for hearing)
<end_of_turn>
<start_of_turn>model
"""

            try:
                start = time.time()
                print("[PreScan] Using Llama 3.3 70B...")
                response = call_huggingface(prompt, timeout=120)
                elapsed = time.time() - start
                print(f"[PreScan] AI response in {elapsed:.1f}s")

                # Parse AI response
                clean = response.replace('```json', '').replace('```', '').strip()
                start_idx = clean.find('{')
                end_idx = clean.rfind('}') + 1

                if start_idx >= 0 and end_idx > start_idx:
                    ai_result = json.loads(clean[start_idx:end_idx])
                    print(f"[PreScan] AI found: {ai_result}")

                    # Merge AI results, but validate filenames
                    for key, val in ai_result.items():
                        if key in missing_items and isinstance(val, dict):
                            source = val.get('source')
                            # Validate source is an actual filename
                            if source and source in filenames_list:
                                metadata[key] = val
                            elif val.get('status') in ['present', 'normal']:
                                # AI says present but wrong filename - try to find correct one
                                for fname in filenames_list:
                                    if key == 'teacher_input' and any(kw in fname.lower() for kw in ['report', 'observation', 'assessment']):
                                        metadata[key] = {'status': val.get('status', 'present'), 'source': fname}
                                        break
                            else:
                                metadata[key] = {'status': val.get('status', 'missing'), 'source': None}

            except Exception as e:
                print(f"[PreScan] AI pass failed: {e}")

        print(f"[PreScan] Final results:")
        for key, val in metadata.items():
            print(f"  {key}: {val['status']} ({val['source']})")

        metadata = fix_red_flags(metadata)
        print(f"[PreScan] Red flags: {metadata.get('red_flags', [])}")

        return jsonify({'success': True, 'metadata': metadata, 'time': 0})

    except Exception as e:
        print(f"[PreScan Error] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/extract', methods=['POST'])
def handle_extract():
    """Extract from each document separately, then merge results"""
    try:
        data = request.get_json()
        model = data.get('model', 'llama-3.3-70b')
        text = data.get('text', '')

        print(f"\n[Extract] Model: {model}, Text length: {len(text)}")

        # Split into individual documents
        docs = []
        current_doc = ""
        current_name = ""
        for line in text.split('\n'):
            if line.startswith('--- ') and line.endswith(' ---'):
                if current_doc.strip():
                    docs.append((current_name, current_doc.strip()))
                current_name = line[4:-4]
                current_doc = ""
            else:
                current_doc += line + "\n"
        if current_doc.strip():
            docs.append((current_name, current_doc.strip()))

        print(f"[Extract] Split into {len(docs)} documents")

        # Process each document separately
        merged = {"A1":[],"A2":[],"A3":[],"B1":[],"B2":[],"B3":[],"B4":[],"C":[],"D":[],"E":[]}

        for doc_name, doc_text in docs:
            print(f"[Extract] Processing: {doc_name} ({len(doc_text)} chars)")

            prompt = build_extraction_prompt(doc_text)
            result = call_huggingface(prompt, timeout=120)

            # Clean markdown
            clean = result.replace('```json', '').replace('```', '').strip()

            start_idx = clean.find('{')
            end_idx = clean.rfind('}') + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = clean[start_idx:end_idx]

                # Fix inconsistent quoting from model
                json_str = fix_json_quotes(json_str)

                try:
                    doc_evidence = json.loads(json_str)
                    for key in merged:
                        if key in doc_evidence and isinstance(doc_evidence[key], list):
                            for quote in doc_evidence[key]:
                                q = quote.strip().strip('"')
                                if q and len(q) > 25 and not is_prompt_echo(q) and not any(existing["text"] == q for existing in merged[key]):
                                    merged[key].append({"text": q, "source": doc_name})
                    print(f"[Extract]   Got {sum(len(doc_evidence.get(k,[])) for k in merged)} quotes")
                except json.JSONDecodeError as e:
                    print(f"[Extract]   JSON error: {e}")
                    print(f"[Extract]   Raw response (first 500 chars): {json_str[:500]}")
            else:
                print(f"[Extract]   No JSON found in response")

        response = json.dumps(merged)
        print(f"[Extract] Final: {sum(len(v) for v in merged.values())} total quotes")

        return jsonify({'success': True, 'response': response})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/extract_twostage', methods=['POST'])
def handle_extract_twostage():
    """Two-stage extraction: Llama 3.3 for quotes and categorization"""
    try:
        data = request.get_json()
        documents = data.get('documents', {})

        print(f"\n[Two-Stage] Processing {len(documents)} documents")

        print("[Stage 1] Extracting quotes with Llama 3.3...")
        all_quotes = []
        stage1_start = time.time()

        for doc_name, doc_text in documents.items():
            print(f"  Processing {doc_name}...", end=" ", flush=True)

            prompt = f"""You must respond with ONLY a JSON object. No other text.

Extract all sentences related to autism assessment from this document.

Categories:
- social: social interaction, engagement, relationships
- communication: language, speech, gestures, pointing
- repetitive: repetitive movements, stereotypies, routines
- sensory: sensory responses, pain sensitivity
- development: developmental milestones, early concerns

Document:
{doc_text[:3000]}

Respond with ONLY this JSON filled with quotes:
{{"social":[],"communication":[],"repetitive":[],"sensory":[],"development":[]}}"""

            try:
                response = call_huggingface(prompt, timeout=120)
                quotes = parse_stage1_response(response, doc_name)
                all_quotes.extend(quotes)
                print(f"{len(quotes)} quotes")
            except Exception as e:
                print(f"Error: {e}")

        stage1_time = time.time() - stage1_start

        seen = set()
        unique_quotes = []
        for q in all_quotes:
            key = q["text"][:50]
            if key not in seen:
                seen.add(key)
                unique_quotes.append(q)

        print(f"[Stage 1] Complete: {len(unique_quotes)} unique quotes in {stage1_time:.1f}s")

        print("[Stage 2] Categorizing with Llama 3.3...")
        stage2_start = time.time()

        quotes_for_stage2 = unique_quotes[:30]
        quotes_text = "\n".join([f'"{q["text"]}" [{q["source"]}]' for q in quotes_for_stage2])

        prompt = f"""You must respond with ONLY a JSON object. No other text.

Categorize these clinical quotes into DSM-5 autism criteria.

CRITERIA:
A1 = Social reciprocity: back-and-forth interaction, responding to name, sharing enjoyment
A2 = Nonverbal: eye contact, gestures, pointing, facial expressions
A3 = Relationships: peer interest, friendships, imaginative play
B1 = Repetitive: stereotyped movements (rocking, flapping, toe walking), echolalia
B2 = Routines: insistence on sameness, distress at changes
B3 = Interests: restricted intense interests, fixations
B4 = Sensory: over/under-reactive to sensory input
C = Early onset: symptoms in early developmental period
D = Impact: functional impairment
E = Rule-outs: hearing/cognitive testing

RULES:
- "supporting" = quote shows the autism feature IS present
- "contradicting" = quote shows the feature is NOT present or is typical
- Include source document in brackets

QUOTES:
{quotes_text}

Respond with ONLY this JSON:
{{"A1":{{"supporting":[],"contradicting":[]}},"A2":{{"supporting":[],"contradicting":[]}},"A3":{{"supporting":[],"contradicting":[]}},"B1":{{"supporting":[],"contradicting":[]}},"B2":{{"supporting":[],"contradicting":[]}},"B3":{{"supporting":[],"contradicting":[]}},"B4":{{"supporting":[],"contradicting":[]}},"C":{{"supporting":[],"contradicting":[]}},"D":{{"supporting":[],"contradicting":[]}},"E":{{"supporting":[],"contradicting":[]}}}}"""

        response = call_huggingface(prompt, timeout=180)
        result = parse_stage2_response(response)

        stage2_time = time.time() - stage2_start
        print(f"[Stage 2] Complete in {stage2_time:.1f}s")

        return jsonify({
            'success': True,
            'evidence': result,
            'stats': {
                'stage1_time': stage1_time,
                'stage2_time': stage2_time,
                'total_quotes': len(unique_quotes),
                'quotes_categorized': len(quotes_for_stage2)
            }
        })

    except Exception as e:
        print(f"[Two-Stage Error] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/extract-functional', methods=['POST'])
def handle_extract_functional():
    """Extract functional assessment information from documents"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        model = data.get('model', 'llama-3.3-70b')

        print(f"\n[Functional] Extracting functional assessment...")
        print(f"[Functional] Document length: {len(text)} chars")

        prompt = build_functional_prompt(text)
        result = call_huggingface(prompt, timeout=180)

        clean = result.replace('```json', '').replace('```', '').strip()
        start_idx = clean.find('{')
        end_idx = clean.rfind('}') + 1

        if start_idx >= 0 and end_idx > start_idx:
            json_str = clean[start_idx:end_idx]
            parsed = json.loads(json_str)
            print(f"[Functional] Extracted {len([k for k,v in parsed.items() if v])} domains with content")
            return jsonify({'success': True, 'response': json_str})
        else:
            print(f"[Functional] No valid JSON found")
            return jsonify({'success': False, 'error': 'No valid JSON in response'})

    except Exception as e:
        print(f"[Functional Error] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/extract-hf', methods=['POST'])
def handle_extract_hf():
    """Extract using HuggingFace API (Llama 3.3 70B) - faster but cloud-based"""
    try:
        data = request.get_json()
        text = data.get('text', '')

        print(f"\n[HF Extract] Text length: {len(text)}")

        # Split into documents
        docs = []
        current_doc = ""
        current_name = ""
        for line in text.split('\n'):
            if line.startswith('--- ') and line.endswith(' ---'):
                if current_doc.strip():
                    docs.append((current_name, current_doc.strip()))
                current_name = line[4:-4]
                current_doc = ""
            else:
                current_doc += line + "\n"
        if current_doc.strip():
            docs.append((current_name, current_doc.strip()))

        print(f"[HF Extract] Split into {len(docs)} documents")

        merged = {"A1":[],"A2":[],"A3":[],"B1":[],"B2":[],"B3":[],"B4":[],"C":[],"D":[],"E":[]}

        for doc_name, doc_text in docs:
            print(f"[HF Extract] Processing: {doc_name} ({len(doc_text)} chars)")

            prompt = build_extraction_prompt(doc_text)
            result = call_huggingface(prompt, timeout=120)

            # Parse JSON response
            clean = result.replace('```json', '').replace('```', '').strip()
            start_idx = clean.find('{')
            end_idx = clean.rfind('}') + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = clean[start_idx:end_idx]
                json_str = fix_json_quotes(json_str)

                try:
                    doc_evidence = json.loads(json_str)
                    for key in merged:
                        if key in doc_evidence and isinstance(doc_evidence[key], list):
                            for quote in doc_evidence[key]:
                                q = quote.strip().strip('"') if isinstance(quote, str) else str(quote)
                                if q and len(q) > 20 and not any(existing["text"] == q for existing in merged[key]):
                                    merged[key].append({"text": q, "source": doc_name})
                    print(f"[HF Extract]   Got {sum(len(doc_evidence.get(k,[])) for k in merged)} quotes")
                except json.JSONDecodeError as e:
                    print(f"[HF Extract]   JSON error: {e}")
            else:
                print(f"[HF Extract]   No JSON found")

        response = json.dumps(merged)
        print(f"[HF Extract] Final: {sum(len(v) for v in merged.values())} total quotes")

        return jsonify({'success': True, 'response': response})

    except Exception as e:
        print(f"[HF Extract Error] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/generate-report', methods=['POST'])
def handle_generate_report():
    """Generate tailored reports for different audiences"""
    try:
        data = request.get_json()

        report_type = data.get('reportType')
        client_info = data.get('clientInfo', {})
        evidence = data.get('evidence', {})
        functional = data.get('functionalAssessment', {})
        diagnostic = data.get('diagnosticDecisions', {})
        case_note = data.get('caseNote', '')
        documents = data.get('documents', {})
        model = data.get('model', 'llama-3.3-70b')

        print(f"\n[Report] Evidence keys: {list(evidence.keys())}")
        print(f"[Report] Documents: {list(documents.keys())}")
        print(f"[Report] Generating {report_type} report for {client_info.get('name', 'Unknown')}")

        prompt = build_report_prompt(report_type, client_info, evidence, functional, diagnostic, case_note, documents)

        # Longer reports need more tokens for the full template
        if report_type == 'ndis':
            max_tokens = 8000
        elif report_type in ['teacher', 'gp', 'caregiver']:
            max_tokens = 6000
        else:
            max_tokens = 2000

        # NDIS needs longer timeout due to comprehensive template
        timeout = 300 if report_type == 'ndis' else 180
        result = call_huggingface(prompt, timeout=timeout, max_tokens=max_tokens)

        print(f"[Report] Generated {len(result)} chars")
        return jsonify({'success': True, 'report': result})

    except Exception as e:
        print(f"[Report Error] {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/export-docx', methods=['POST'])
def handle_export_docx():
    """Export case note as Word document"""
    try:
        return jsonify({'success': False, 'error': 'DOCX export not yet implemented'})
    except Exception as e:
        print(f"[Export Error] {e}")
        return jsonify({'success': False, 'error': str(e)})

# ============== Helper Functions ==============

def fix_red_flags(metadata):
    """Post-process to ensure red_flags only contains genuinely missing critical items"""
    red_flags = []

    items = {
        'gp_referral': 'No referral letter - REQUEST before assessment',
        'hearing_test': 'No hearing test - REFER NOW, cannot diagnose without ruling out hearing loss',
        'dev_history': 'No developmental history - SEND TO PARENT or extend interview',
        'teacher_input': 'No teacher/school input - FOLLOW UP, need multi-context evidence',
    }

    for key, message in items.items():
        status = metadata.get(key, {}).get('status', 'missing')
        if status == 'missing':
            red_flags.append(message)
        elif key == 'hearing_test' and status == 'not_done':
            red_flags.append(message)

    prev_status = metadata.get('previous_asd', {}).get('status', 'none_mentioned')
    if prev_status == 'missing':
        red_flags.append('Previous ASD assessment mentioned but not obtained - REQUEST')

    metadata['red_flags'] = red_flags
    return metadata

def is_prompt_echo(text):
    """Filter out quotes that are just echoes of prompt criteria"""
    prompt_fragments = [
        'response to name', 'sharing enjoyment', 'back-and-forth interaction',
        'eye contact', 'pointing', 'gestures', 'facial expressions', 'joint attention',
        'peer interest', 'friendships', 'imaginative play', 'cooperative play',
        'hand flapping', 'rocking', 'toe walking', 'head banging', 'echolalia', 'lining up',
        'distress at changes', 'rigid routines', 'need for sameness', 'transitions',
        'intense interests', 'fixations', 'preoccupations', 'perseverative focus',
        'sound sensitivity', 'texture aversion', 'pain response', 'food selectivity', 'mouthing',
        'early milestones', 'regression', 'when concerns first noted',
        'school difficulties', 'social difficulties', 'daily living impact',
        'hearing tests', 'vision tests', 'cognitive assessment', 'other diagnoses'
    ]
    lower = text.lower().strip()
    if len(text) < 30:
        for frag in prompt_fragments:
            if lower == frag or lower == frag + 's':
                return True
    return False

def fix_json_quotes(json_str):
    """Leniently repair malformed JSON output from an LLM"""
    s = (json_str or "").strip()

    # Quick normalizations
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r'""([^"]*?)""', r'"\1"', s)

    prev = None
    while prev != s:
        prev = s
        s = re.sub(r'"\\\"(.*?)\\\""', r'"\1"', s)

    def strip_outer_quotes(token):
        t = token.strip()
        if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
            t = t[1:-1]
        if len(t) >= 2 and t[0] in '""' and t[-1] in '""':
            t = t[1:-1]
        return t.strip()

    def unescape_model_wrappers(t):
        t = t.strip()
        t = strip_outer_quotes(t)
        t = re.sub(r'^\\"(.*)\\"$', r'\1', t)
        t = re.sub(r'^"+(.*)"+$', r'\1', t)
        return t.strip()

    result = {}
    key_iter = list(re.finditer(r'"([^"]+)"\s*:\s*\[', s))
    if not key_iter:
        key_iter = list(re.finditer(r'([A-Za-z][A-Za-z0-9_]*)\s*:\s*\[', s))

    n = len(s)

    def find_matching_bracket(start_idx):
        depth = 0
        in_str = False
        esc = False
        i = start_idx
        while i < n:
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        return i
            i += 1
        return -1

    def split_array_items(array_body):
        items = []
        buf = []
        in_str = False
        esc = False
        for ch in array_body:
            if in_str:
                buf.append(ch)
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                    buf.append(ch)
                elif ch == ',':
                    token = "".join(buf).strip()
                    items.append(token)
                    buf = []
                else:
                    buf.append(ch)
        token = "".join(buf).strip()
        if token:
            items.append(token)
        return items

    for m in key_iter:
        key = m.group(1)
        bracket_start = m.end() - 1
        if bracket_start < 0 or bracket_start >= n or s[bracket_start] != '[':
            continue

        bracket_end = find_matching_bracket(bracket_start)
        if bracket_end == -1:
            bracket_end = n - 1

        body = s[bracket_start + 1: bracket_end]
        raw_tokens = split_array_items(body)

        cleaned_items = []
        for tok in raw_tokens:
            t = tok.strip()
            if not t:
                continue

            t = re.sub(r'^[\s\]]+', '', t)
            t = re.sub(r'[\s\[]+$', '', t).strip()

            if t.startswith('"') or t.endswith('"') or t.startswith('"') or t.endswith('"'):
                t2 = unescape_model_wrappers(t)
                if t2:
                    cleaned_items.append(t2)
                continue

            bare = unescape_model_wrappers(t)
            if bare:
                if cleaned_items:
                    if cleaned_items[-1].endswith((',', ';', ':')):
                        cleaned_items[-1] = cleaned_items[-1] + " " + bare
                    else:
                        cleaned_items[-1] = cleaned_items[-1] + ", " + bare
                else:
                    cleaned_items.append(bare)

        final_items = []
        for it in cleaned_items:
            x = re.sub(r'\s+', ' ', it).strip()
            x = x.replace('\u201c', '"').replace('\u201d', '"')
            final_items.append(x)

        result[key] = final_items

    if not result:
        s = re.sub(r'""([^"]*?)""', r'"\1"', s)
        s = re.sub(r'"\\\"(.*?)\\\""', r'"\1"', s)
        return s

    return json.dumps(result, ensure_ascii=False)

def build_extraction_prompt(text):
    """Build DSM-5 extraction prompt"""
    return f"""You must respond with ONLY a JSON object. No explanations. No markdown.

TASK: Extract EXACT word-for-word quotes from this clinical document for autism assessment.

CRITICAL INSTRUCTION: You MUST extract BOTH types of evidence:
1. Evidence that a feature IS PRESENT (supports ASD)
2. Evidence that a feature is ABSENT, INTACT, or TYPICAL (contradicts ASD)

=== CRITERIA ===

A1 - SOCIAL-EMOTIONAL RECIPROCITY
A2 - NONVERBAL COMMUNICATION
A3 - RELATIONSHIPS
B1 - STEREOTYPED/REPETITIVE MOVEMENTS OR SPEECH
B2 - INSISTENCE ON SAMENESS / ROUTINES
B3 - RESTRICTED/FIXATED INTERESTS
B4 - SENSORY HYPER/HYPOREACTIVITY
C - EARLY DEVELOPMENTAL PERIOD
D - FUNCTIONAL IMPAIRMENT
E - DIFFERENTIAL DIAGNOSIS

=== RULES ===
1. Copy EXACT quotes - do not paraphrase
2. Each quote should be 5-40 words
3. Extract ALL relevant quotes - do not limit
4. "Explicitly absent" statements ARE evidence - extract them

=== DOCUMENT ===
{text}

Return ONLY this JSON with actual quotes:
{{"A1":[],"A2":[],"A3":[],"B1":[],"B2":[],"B3":[],"B4":[],"C":[],"D":[],"E":[]}}

Each array contains simple quote strings only - no objects, no source field.
Empty array [] if no relevant quotes. Extract comprehensively."""

def build_functional_prompt(text):
    """Build functional assessment extraction prompt"""
    return f"""You must respond with ONLY a JSON object. No explanations. No markdown. Just JSON.

Extract information for a functional assessment from the clinical documents below.
For each domain, extract relevant quotes and summarize key findings.

DOMAINS TO EXTRACT:
- strengths: Skills, interests, abilities, things the person enjoys or is good at
- medical: Health conditions, physical development, vision, hearing, medications
- cognitive: Thinking, learning, problem-solving, reasoning, academic performance
- speech: Language development, communication skills, receptive/expressive language
- motor: Gross motor, fine motor, coordination, handwriting, physical skills
- social: Peer relationships, social understanding, friendships, play skills
- emotional: Emotional regulation, anxiety, mood, behavioral concerns
- attention: Focus, concentration, executive function, planning, organization
- adaptive: Self-care, daily living skills, independence, practical skills
- background: Developmental history, family history, services involved, psychosocial factors

DOCUMENT TEXT:
{text}

Respond with ONLY this JSON structure:
{{
  "strengths": "Summary of strengths and interests found...",
  "medical": "Summary of medical/health information...",
  "cognitive": "Summary of cognitive/learning information...",
  "speech": "Summary of speech/language/communication...",
  "motor": "Summary of motor skills...",
  "social": "Summary of social functioning...",
  "emotional": "Summary of emotional regulation/behaviour...",
  "attention": "Summary of attention/executive function...",
  "adaptive": "Summary of adaptive/daily living skills...",
  "background": "Summary of history and background..."
}}

If no information is found for a domain, use an empty string ""."""

def build_report_prompt(report_type, client_info, evidence, functional, diagnostic, case_note, documents=None):
    """Build audience-specific report prompt"""
    documents = documents or {}
    name = client_info.get('name', '[Name]')
    age = client_info.get('age', '')
    pronouns = client_info.get('pronouns', 'they/them')

    functional_summary = "\n".join([f"- {k}: {v[:200]}..." if len(str(v)) > 200 else f"- {k}: {v}"
                                    for k, v in functional.items() if v])

    # Format evidence for report
    evidence_text = ""
    for criterion, data in evidence.items():
        if isinstance(data, dict) and 'quotes' in data:
            quotes = data['quotes']
        elif isinstance(data, list):
            quotes = data
        else:
            continue

        if quotes:
            quote_list = ", ".join([f'"{q}"' if isinstance(q, str) else f'"{q.get("text", "")}"' for q in quotes[:5]])
            evidence_text += f"- {criterion}: {quote_list}\n"

    diagnosis_met = diagnostic.get('asdMet', False)
    severity = diagnostic.get('severityLevel', 'Level 1')

    base_context = f"""CLIENT: {name}, Age {age}
PRONOUNS: {pronouns}
DIAGNOSIS: {'ASD confirmed' if diagnosis_met else 'ASD not confirmed'} - {severity}

EXTRACTED EVIDENCE:
{evidence_text}

FUNCTIONAL SUMMARY:
{functional_summary}

CASE NOTE EXCERPT:
{case_note[:2000]}"""

    if report_type == 'caregiver':
        caregiver_template = get_caregiver_template()
        docs_text = ""
        for doc_name, doc_content in documents.items():
            docs_text += f"\n--- {doc_name} ---\n{doc_content[:6000]}\n"

        return f"""Fill in this caregiver report template using the clinical documents provided.

CHILD: {name}
AGE: {age}
PRONOUNS: {pronouns}
DIAGNOSIS: {'ASD confirmed' if diagnosis_met else 'ASD not confirmed'} - {severity}

=== ORIGINAL CLINICAL DOCUMENTS ===
{docs_text}

=== CAREGIVER REPORT TEMPLATE ===
{caregiver_template}

=== INSTRUCTIONS ===
1. Read through ALL the clinical documents above carefully
2. Fill in EVERY section of the template using actual information from the documents
3. Replace all {{placeholders}} with real data from the clinical reports
4. Use warm, supportive, plain language - no clinical jargon
5. Use neurodiversity-affirming language (differences not deficits)
6. If information is not in documents, write "[To be discussed with family]" - do not invent

Write the complete filled-in Caregiver Report now in Markdown format:"""

    elif report_type == 'teacher':
        teacher_template = get_teacher_template()
        docs_text = ""
        for doc_name, doc_content in documents.items():
            docs_text += f"\n--- {doc_name} ---\n{doc_content[:4000]}\n"

        return f"""Fill in this teacher letter template using the clinical reports provided.

STUDENT: {name}
AGE: {age}
PRONOUNS: {pronouns}
DIAGNOSIS: {'ASD confirmed' if diagnosis_met else 'ASD not confirmed'} - {severity}

=== ORIGINAL CLINICAL REPORTS ===
{docs_text}

=== TEMPLATE TO COMPLETE ===
{teacher_template}
=== END TEMPLATE ===

Output the completed letter:"""

    elif report_type == 'gp':
        gp_template = get_gp_template()
        docs_text = ""
        for doc_name, doc_content in documents.items():
            docs_text += f"\n--- {doc_name} ---\n{doc_content[:4000]}\n"

        return f"""Fill in this GP letter template using the clinical reports provided.

PATIENT: {name}
AGE: {age}
PRONOUNS: {pronouns}
DIAGNOSIS: {'ASD confirmed' if diagnosis_met else 'ASD not confirmed'} - {severity}

=== ORIGINAL CLINICAL REPORTS ===
{docs_text}

=== TEMPLATE TO COMPLETE ===
{gp_template}
=== END TEMPLATE ===

Output the completed letter:"""

    elif report_type == 'ndis':
        ndis_template = get_ndis_template()
        docs_text = ""
        for doc_name, doc_content in documents.items():
            docs_text += f"\n--- {doc_name} ---\n{doc_content[:6000]}\n"

        return f"""Fill in this NDIS Supporting Evidence template using the clinical documents provided.

CHILD: {name}
AGE: {age}
PRONOUNS: {pronouns}
DIAGNOSIS: {'ASD confirmed' if diagnosis_met else 'ASD not confirmed'} - {severity}

=== ORIGINAL CLINICAL DOCUMENTS ===
{docs_text}

=== NDIS SUPPORTING EVIDENCE TEMPLATE ===
{ndis_template}

Write the complete filled-in NDIS Supporting Evidence Report now in Markdown format:"""

    else:
        return f"Generate a summary report for {name}."

def get_gp_template():
    """Load GP letter template"""
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'gp_letter_template.md')
    if os.path.exists(template_path):
        try:
            with open(template_path, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"[Template] Error loading GP template: {e}")

    return """**GP Letter Template - Fallback**"""

def get_teacher_template():
    """Load teacher letter template"""
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'teacher_letter_template.md')
    if os.path.exists(template_path):
        try:
            with open(template_path, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"[Template] Error loading teacher template: {e}")

    return """**Teacher Letter Template - Fallback**"""

def get_caregiver_template():
    """Load caregiver report template"""
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'caregiver_template.md')
    if os.path.exists(template_path):
        try:
            with open(template_path, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"[Template] Error loading caregiver template: {e}")

    return """**Caregiver Report Template - Fallback**"""

def get_ndis_template():
    """Load NDIS template"""
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'ndis_template.md')
    if os.path.exists(template_path):
        try:
            with open(template_path, 'r') as f:
                return f.read()
        except Exception as e:
            print(f"[Template] Error loading NDIS template: {e}")

    return """**NDIS Supporting Evidence Template - Fallback**"""

def call_huggingface(prompt, timeout=120, max_tokens=2000):
    """Call HuggingFace API via Together provider"""
    token = HF_TOKEN or os.environ.get('HF_TOKEN', '')

    if not token:
        raise ValueError("HF_TOKEN environment variable not set")

    payload = json.dumps({
        "model": "meta-llama/llama-3.3-70b-instruct",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens
    })

    req = urllib.request.Request(
        'https://router.huggingface.co/novita/v3/openai/chat/completions',
        data=payload.encode(),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {token}',
            'User-Agent': 'Mozilla/5.0'
        }
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
        return result['choices'][0]['message']['content']

def parse_stage1_response(response, doc_name):
    """Parse Stage 1 response into quote list"""
    quotes = []
    clean = response.replace('```json', '').replace('```', '').strip()
    start_idx = clean.find('{')
    end_idx = clean.rfind('}') + 1

    if start_idx >= 0 and end_idx > start_idx:
        try:
            data = json.loads(clean[start_idx:end_idx])
            for category, quote_list in data.items():
                if isinstance(quote_list, list):
                    for q in quote_list:
                        if isinstance(q, str) and len(q) > 15:
                            quotes.append({
                                "text": q,
                                "source": doc_name,
                                "category": category
                            })
        except json.JSONDecodeError:
            pass

    return quotes

def parse_stage2_response(response):
    """Parse Stage 2 response into evidence dict"""
    clean = response.replace('```json', '').replace('```', '').strip()
    start_idx = clean.find('{')
    end_idx = clean.rfind('}') + 1

    if start_idx >= 0 and end_idx > start_idx:
        try:
            data = json.loads(clean[start_idx:end_idx])
            for crit in data:
                if 'supporting' in data[crit]:
                    data[crit]['supporting'] = list(dict.fromkeys(data[crit]['supporting']))
                if 'contradicting' in data[crit]:
                    data[crit]['contradicting'] = list(dict.fromkeys(data[crit]['contradicting']))
            return data
        except json.JSONDecodeError:
            pass

    return {}

# ============== Main ==============

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))

    print("=" * 60)
    print(f"ASD Workflow Server v{VERSION} (Flask)")
    print("=" * 60)
    print(f"Server running at: http://localhost:{port}")
    print(f"Open: http://localhost:{port}/")
    print("=" * 60)

    app.run(host='0.0.0.0', port=port, debug=False)
