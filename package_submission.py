#!/usr/bin/env python3
"""Submission Packaging Script for Amazon Business Entity Resolution Challenge.

Verifies outputs with utils/validate_submission.py and packages the final archive:
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import argparse
import os
import shutil
import subprocess
import sys
import zipfile


def main():
    parser = argparse.ArgumentParser(description="Package final submission archive.")
    parser.add_argument("--team-name", default="TopRankTeam", help="Your team name.")
    parser.add_argument("--output-dir", default="output", help="Directory containing output TSVs.")
    parser.add_argument("--test-dir", default="dataset/test", help="Directory containing test dataset.")
    args = parser.parse_args()

    team_name = args.team_name.replace(" ", "_")
    zip_filename = f"{team_name}_submission.zip"
    staging_dir = "submission_staging"

    match_file = os.path.join(args.output_dir, "matching_results.tsv")
    cand_file = os.path.join(args.output_dir, "candidate_pairs.tsv")

    print("=" * 75)
    print(f" Packaging Submission for Team: {team_name}")
    print("=" * 75)

    # 1. Run Validator First
    val_script = "utils/validate_submission.py"
    if os.path.exists(val_script) and os.path.exists(match_file) and os.path.exists(cand_file):
        print("\n[Step 1/3] Running official submission validator...")
        val_cmd = [
            sys.executable, val_script,
            "--matching", match_file,
            "--candidate", cand_file,
        ]
        if os.path.exists(args.test_dir):
            val_cmd.extend(["--test-dir", args.test_dir])

        res = subprocess.run(val_cmd)
        if res.returncode != 0:
            print("[ERROR] Submission validation failed! Fix errors before packaging.", file=sys.stderr)
            sys.exit(1)
    else:
        print("[WARN] Skipping validator: output files or validator script not found.")

    # 2. Build staging structure
    print("\n[Step 2/3] Assembling submission directory structure...")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)

    out_stage = os.path.join(staging_dir, "output")
    code_stage = os.path.join(staging_dir, "code", "business_entity_resolution")
    src_stage = os.path.join(code_stage, "src")
    os.makedirs(out_stage, exist_ok=True)
    os.makedirs(src_stage, exist_ok=True)

    # Copy output files
    if os.path.exists(match_file):
        shutil.copy2(match_file, out_stage)
    if os.path.exists(cand_file):
        shutil.copy2(cand_file, out_stage)

    # Copy src files
    for f in os.listdir("src"):
        if f.endswith(".py"):
            shutil.copy2(os.path.join("src", f), src_stage)

    # Copy requirements & README
    if os.path.exists("requirements.txt"):
        shutil.copy2("requirements.txt", code_stage)
    
    # Create code README
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
   python3 run_pipeline.py
   ```

This will run data auditing, 5-fold CV splitting, multi-route candidate generation,
calibrated GBDT pairwise modeling with dedicated singleton gating, and generate both
`output/matching_results.tsv` and `output/candidate_pairs.tsv`.
"""
    with open(os.path.join(code_stage, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)

    # Copy Documentation_template.md
    if os.path.exists("Documentation_template.md"):
        shutil.copy2("Documentation_template.md", staging_dir)

    # 3. Create zip archive
    print(f"\n[Step 3/3] Creating zip archive: {zip_filename}...")
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(staging_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, staging_dir)
                zipf.write(abs_path, rel_path)

    # Cleanup staging
    shutil.rmtree(staging_dir)
    print(f"\n[SUCCESS] Submission package ready: {zip_filename}")
    print(f"Archive Size: {os.path.getsize(zip_filename) / 1024:.2f} KB")


if __name__ == "__main__":
    main()
