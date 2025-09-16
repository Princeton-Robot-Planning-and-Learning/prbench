#!/usr/bin/env python3
"""Generate videos for all demo files."""

import subprocess
from pathlib import Path

def generate_all_videos():
    """Generate videos for all .p demo files."""
    demo_dir = Path("demos")
    demo_files = list(demo_dir.rglob("*.p"))
    
    print(f"Found {len(demo_files)} demo files")
    
    for i, demo_file in enumerate(demo_files, 1):
        print(f"[{i}/{len(demo_files)}] Generating video for: {demo_file}")
        try:
            result = subprocess.run([
                "python", "scripts/generate_demo_video.py", 
                str(demo_file)
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"  ✓ Success")
            else:
                print(f"  ✗ Failed: {result.stderr}")
                
        except Exception as e:
            print(f"  ✗ Error: {e}")

if __name__ == "__main__":
    generate_all_videos()
