import os, sys, time, zipfile, shutil

print('=' * 75)
print(' STREAMING VALIDATOR & PACKAGER FOR AMAZON ML SUBMISSION')
print('=' * 75)

t0 = time.time()
test_s1_path = 'dataset/test/test_source1.tsv'
matching_path = 'output/matching_results.tsv'
candidate_path = 'output/candidate_pairs.tsv'

if not os.path.exists(test_s1_path):
    print(f'ERROR: {test_s1_path} not found!')
    sys.exit(1)

# Step 1: Read all expected S1 IDs in order
print('[1/4] Reading required S1 IDs from test_source1.tsv...')
expected_ids = []
with open(test_s1_path, 'r', encoding='utf-8') as f:
    header = next(f)
    for line in f:
        parts = line.split('\t', 1)
        expected_ids.append(parts[0])

num_expected = len(expected_ids)
print(f'  Required S1 entities: {num_expected:,}')

# Step 2: Stream validate matching_results.tsv
print('\n[2/4] Validating matching_results.tsv in streaming mode...')
matching_map = {}
total_links = 0
singletons = 0
match_errors = []

with open(matching_path, 'r', encoding='utf-8') as f:
    hdr = f.readline().rstrip('\r\n')
    if hdr != 'source1_entity_id\tmatched_entity_ids':
        match_errors.append(f'Invalid header: {hdr}')
    
    idx = 0
    for line_idx, line in enumerate(f, start=2):
        line = line.rstrip('\r\n')
        parts = line.split('\t')
        if len(parts) != 2:
            match_errors.append(f'Line {line_idx}: invalid column count ({len(parts)})')
            break
        s1, mids_str = parts
        if idx < num_expected and s1 != expected_ids[idx]:
            match_errors.append(f'Line {line_idx}: S1 mismatch. Expected {expected_ids[idx]}, got {s1}')
            if len(match_errors) > 5: break
        idx += 1
        
        mids = mids_str.split(',') if mids_str else []
        if not mids:
            singletons += 1
        else:
            if len(mids) != len(set(mids)):
                match_errors.append(f'Line {line_idx}: duplicate IDs inside matched list: {s1}')
            for mid in mids:
                if mid.startswith('S1-'):
                    match_errors.append(f'Line {line_idx}: contains S1 self-match: {mid}')
                elif not (mid.startswith('S2-') or mid.startswith('S3-')):
                    match_errors.append(f'Line {line_idx}: invalid ID prefix: {mid}')
            total_links += len(mids)
        if mids:
            matching_map[s1] = set(mids)

if idx != num_expected:
    match_errors.append(f'Total row count mismatch: found {idx}, expected {num_expected}')

if match_errors:
    print('  FAIL on matching_results.tsv:')
    for err in match_errors[:10]:
        print(f'    - {err}')
    sys.exit(1)
else:
    print(f'  PASS matching_results.tsv! Rows: {idx:,}, Singletons: {singletons:,} ({singletons/idx*100:.2f}%), Total links: {total_links:,}')

# Step 3: Stream validate candidate_pairs.tsv and check candidate superset
print('\n[3/4] Validating candidate_pairs.tsv in streaming mode...')
cand_errors = []
cand_singletons = 0
total_candidates = 0
missing_candidates = 0

with open(candidate_path, 'r', encoding='utf-8') as f:
    hdr = f.readline().rstrip('\r\n')
    if hdr != 'source1_entity_id\tcandidate_entity_ids':
        cand_errors.append(f'Invalid header: {hdr}')
        
    cidx = 0
    for line_idx, line in enumerate(f, start=2):
        line = line.rstrip('\r\n')
        parts = line.split('\t')
        if len(parts) != 2:
            cand_errors.append(f'Line {line_idx}: invalid column count ({len(parts)})')
            break
        s1, cids_str = parts
        if cidx < num_expected and s1 != expected_ids[cidx]:
            cand_errors.append(f'Line {line_idx}: S1 mismatch. Expected {expected_ids[cidx]}, got {s1}')
            if len(cand_errors) > 5: break
        cidx += 1
        
        cids = cids_str.split(',') if cids_str else []
        if not cids:
            cand_singletons += 1
        else:
            if len(cids) != len(set(cids)):
                cand_errors.append(f'Line {line_idx}: duplicate IDs inside candidate list: {s1}')
            total_candidates += len(cids)
            
        # Verify that all final matched entities exist in candidate pairs
        if s1 in matching_map:
            cid_set = set(cids)
            diff = matching_map[s1] - cid_set
            if diff:
                missing_candidates += len(diff)
                if missing_candidates <= 5:
                    print(f'  WARNING: S1 {s1} has match {diff} not in candidates!')

if cidx != num_expected:
    cand_errors.append(f'Total row count mismatch: found {cidx}, expected {num_expected}')

if cand_errors:
    print('  FAIL on candidate_pairs.tsv:')
    for err in cand_errors[:10]:
        print(f'    - {err}')
    sys.exit(1)
else:
    print(f'  PASS candidate_pairs.tsv! Rows: {cidx:,}, Total candidates: {total_candidates:,}')
    if missing_candidates == 0:
        print('  PERFECT CONCORDANCE: 100% of matched IDs are verified present in candidate_pairs.tsv!')
    else:
        print(f'  WARNING: {missing_candidates} matched IDs were not in candidate pairs.')

# Step 4: Assemble TopRankTeam_submission.zip
print('\n[4/4] Assembling TopRankTeam_submission.zip...')
staging_dir = 'submission_staging'
if os.path.exists(staging_dir):
    shutil.rmtree(staging_dir)

out_stage = os.path.join(staging_dir, 'output')
code_stage = os.path.join(staging_dir, 'code', 'business_entity_resolution')
src_stage = os.path.join(code_stage, 'src')
os.makedirs(out_stage, exist_ok=True)
os.makedirs(src_stage, exist_ok=True)

shutil.copy2(matching_path, out_stage)
shutil.copy2(candidate_path, out_stage)

for f in os.listdir('src'):
    if f.endswith('.py'):
        shutil.copy2(os.path.join('src', f), src_stage)

if os.path.exists('requirements.txt'):
    shutil.copy2('requirements.txt', code_stage)

if os.path.exists('Documentation_template.md'):
    shutil.copy2('Documentation_template.md', staging_dir)

readme_content = """# Business Entity Resolution Pipeline

## Reproducing End-to-End Results

1. Setup environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Place raw datasets in `dataset/train/` and `dataset/test/`.

3. Run end-to-end pipeline:
   ```bash
   python3 src/run_robust_production.py
   ```

This pipeline executes 7-route candidate blocking, rapid C++ fuzzy & token concordance,
mutual-best 1-to-1 global bipartite matching, and produces the verified submission TSVs.
"""
with open(os.path.join(code_stage, 'README.md'), 'w', encoding='utf-8') as f:
    f.write(readme_content)

zip_filename = 'TopRankTeam_submission.zip'
print(f'Compressing into {zip_filename}...')
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(staging_dir):
        for file in files:
            abs_path = os.path.join(root, file)
            rel_path = os.path.relpath(abs_path, staging_dir)
            zipf.write(abs_path, rel_path)

shutil.rmtree(staging_dir)
zip_size_mb = os.path.getsize(zip_filename) / (1024 * 1024)
print(f'\n[SUCCESS] Final submission package ready: {zip_filename} ({zip_size_mb:.2f} MB)')
print(f'Total validation and packaging time: {time.time() - t0:.1f}s')
