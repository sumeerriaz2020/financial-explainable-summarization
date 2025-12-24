#!/usr/bin/env python3
"""
Download Model Checkpoints
==========================

Download pre-trained model checkpoints from HuggingFace Hub or alternative sources.

Usage:
    python download_checkpoints.py --model fully_trained
    python download_checkpoints.py --all
    python download_checkpoints.py --model fully_trained --verify
    
Author: Sumeer Riaz, Dr. M. Bilal Bashir, Syed Ali Hassan Naqvi
"""

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Optional, Dict
import json

try:
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Install with: pip install huggingface_hub")


# Configuration
REPO_ID = "your-username/financial-explainable-summarization"
CACHE_DIR = "checkpoints/"

CHECKPOINTS = {
    'base': {
        'filename': 'base_model.pt',
        'size': '1.6 GB',
        'size_bytes': 1717986918,
        'description': 'Base model with initialized components',
        'sha256': None,  # Add after upload
        'alternative_url': None
    },
    'fully_trained': {
        'filename': 'fully_trained_model.pt',
        'size': '1.7 GB',
        'size_bytes': 1825361100,
        'description': 'Complete trained model (10 epochs) - RECOMMENDED',
        'sha256': None,
        'alternative_url': None,
        'metrics': {
            'rouge_l': 0.487,
            'bertscore': 0.686,
            'factual_consistency': 87.6,
            'ssi': 0.74,
            'cps': 0.51,
            'tcc': 0.54
        }
    },
    'finbert': {
        'filename': 'finbert_ner.pt',
        'size': '438 MB',
        'size_bytes': 459276800,
        'description': 'Fine-tuned FinBERT for financial NER',
        'sha256': None,
        'alternative_url': None
    },
    'stage1': {
        'filename': 'stage1_final.pt',
        'size': '1.6 GB',
        'size_bytes': 1717986918,
        'description': 'Stage 1 checkpoint (warm-up phase)',
        'sha256': None,
        'alternative_url': None
    },
    'stage2': {
        'filename': 'stage2_final.pt',
        'size': '1.6 GB',
        'size_bytes': 1717986918,
        'description': 'Stage 2 checkpoint (KG integration)',
        'sha256': None,
        'alternative_url': None
    },
    'stage3': {
        'filename': 'stage3_final.pt',
        'size': '1.7 GB',
        'size_bytes': 1825361100,
        'description': 'Stage 3 checkpoint (end-to-end) - same as fully_trained',
        'sha256': None,
        'alternative_url': None
    }
}


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{'=' * 70}")
    print(f"{text}")
    print(f"{'=' * 70}\n")


def print_info(checkpoint_name: str):
    """Print checkpoint information"""
    info = CHECKPOINTS[checkpoint_name]
    print(f"Checkpoint: {checkpoint_name}")
    print(f"Description: {info['description']}")
    print(f"Filename: {info['filename']}")
    print(f"Size: {info['size']}")
    
    if 'metrics' in info:
        print(f"\nExpected Performance:")
        for metric, value in info['metrics'].items():
            print(f"  {metric}: {value}")


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of file"""
    sha256 = hashlib.sha256()
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    
    return sha256.hexdigest()


def verify_checkpoint(checkpoint_path: str, expected_hash: Optional[str] = None) -> bool:
    """Verify checkpoint integrity"""
    print_header("Checkpoint Verification")
    
    try:
        import torch
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"\n✓ Checkpoint loaded successfully!")
        
        # Display contents
        print(f"\nCheckpoint contents:")
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"  • {key}: {len(checkpoint[key])} items")
            elif isinstance(checkpoint[key], list):
                print(f"  • {key}: {len(checkpoint[key])} elements")
            else:
                print(f"  • {key}: {type(checkpoint[key]).__name__}")
        
        # Display configuration
        if 'config' in checkpoint:
            print(f"\nModel Configuration:")
            config = checkpoint['config']
            if isinstance(config, dict):
                for key, value in list(config.items())[:5]:
                    print(f"  • {key}: {value}")
        
        # Display metrics
        if 'metrics' in checkpoint:
            print(f"\nPerformance Metrics:")
            for metric, value in checkpoint['metrics'].items():
                print(f"  • {metric}: {value}")
        
        # Verify hash if provided
        if expected_hash:
            print(f"\nVerifying file integrity...")
            actual_hash = compute_file_hash(checkpoint_path)
            
            if actual_hash == expected_hash:
                print(f"✓ Hash verified: {actual_hash[:16]}...")
                return True
            else:
                print(f"✗ Hash mismatch!")
                print(f"  Expected: {expected_hash[:16]}...")
                print(f"  Actual:   {actual_hash[:16]}...")
                return False
        
        return True
    
    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        return False


def download_from_huggingface(
    checkpoint_name: str,
    cache_dir: str = CACHE_DIR
) -> Optional[str]:
    """Download checkpoint from HuggingFace Hub"""
    
    if not HF_AVAILABLE:
        print("Error: huggingface_hub not installed")
        print("Install with: pip install huggingface_hub")
        return None
    
    info = CHECKPOINTS[checkpoint_name]
    
    print_header(f"Downloading from HuggingFace: {checkpoint_name}")
    print_info(checkpoint_name)
    
    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"\nDownloading from: {REPO_ID}")
        print(f"This may take a while ({info['size']})...")
        
        checkpoint_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=info['filename'],
            cache_dir=cache_dir,
            resume_download=True,
            local_dir=cache_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"\n✓ Download completed!")
        print(f"Location: {checkpoint_path}")
        
        return checkpoint_path
    
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"1. Check internet connection")
        print(f"2. Verify repository exists: https://huggingface.co/{REPO_ID}")
        print(f"3. Try alternative download methods (see README)")
        
        return None


def download_from_url(url: str, output_path: str) -> bool:
    """Download file from URL with progress bar"""
    try:
        import requests
        
        print(f"Downloading from: {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        print(f"✓ Download completed: {output_path}")
        return True
    
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def download_checkpoint(
    checkpoint_name: str,
    cache_dir: str = CACHE_DIR,
    verify: bool = False
) -> Optional[str]:
    """Download a specific checkpoint"""
    
    if checkpoint_name not in CHECKPOINTS:
        print(f"Error: Unknown checkpoint '{checkpoint_name}'")
        print(f"Available checkpoints: {', '.join(CHECKPOINTS.keys())}")
        return None
    
    info = CHECKPOINTS[checkpoint_name]
    
    # Try HuggingFace first
    checkpoint_path = download_from_huggingface(checkpoint_name, cache_dir)
    
    # Try alternative URL if HuggingFace fails
    if not checkpoint_path and info.get('alternative_url'):
        output_path = Path(cache_dir) / info['filename']
        if download_from_url(info['alternative_url'], str(output_path)):
            checkpoint_path = str(output_path)
    
    # Verify if requested
    if checkpoint_path and verify:
        if not verify_checkpoint(checkpoint_path, info.get('sha256')):
            print("\nWarning: Verification failed, but file was downloaded")
    
    return checkpoint_path


def download_all(cache_dir: str = CACHE_DIR, verify: bool = False):
    """Download all checkpoints"""
    
    print_header("Downloading All Checkpoints")
    
    total_size = sum(c['size_bytes'] for c in CHECKPOINTS.values())
    total_size_gb = total_size / (1024**3)
    
    print(f"Total size: ~{total_size_gb:.1f} GB")
    print(f"Checkpoints: {len(CHECKPOINTS)}")
    print(f"\nThis will download:")
    
    for name, info in CHECKPOINTS.items():
        print(f"  • {name}: {info['size']}")
    
    response = input("\nContinue? [y/N]: ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    success_count = 0
    failed = []
    
    for name in CHECKPOINTS.keys():
        checkpoint_path = download_checkpoint(name, cache_dir, verify)
        
        if checkpoint_path:
            success_count += 1
        else:
            failed.append(name)
    
    # Summary
    print_header("Download Summary")
    print(f"Successful: {success_count}/{len(CHECKPOINTS)}")
    
    if failed:
        print(f"\nFailed downloads:")
        for name in failed:
            print(f"  • {name}")
    else:
        print(f"\n✓ All checkpoints downloaded successfully!")


def list_checkpoints():
    """List all available checkpoints"""
    
    print_header("Available Checkpoints")
    
    for name, info in CHECKPOINTS.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Size: {info['size']}")
        print(f"  Filename: {info['filename']}")
        
        if 'metrics' in info:
            print(f"  Performance:")
            for metric, value in info['metrics'].items():
                print(f"    • {metric}: {value}")


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained model checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download fully trained model (recommended)
  python download_checkpoints.py --model fully_trained
  
  # Download all checkpoints
  python download_checkpoints.py --all
  
  # Download and verify
  python download_checkpoints.py --model fully_trained --verify
  
  # List available checkpoints
  python download_checkpoints.py --list
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=list(CHECKPOINTS.keys()),
        help='Specific checkpoint to download'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Download all checkpoints (~5.3 GB total)'
    )
    
    parser.add_argument(
        '--cache_dir',
        type=str,
        default=CACHE_DIR,
        help=f'Directory to save checkpoints (default: {CACHE_DIR})'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify checkpoint integrity after download'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available checkpoints'
    )
    
    args = parser.parse_args()
    
    # List checkpoints
    if args.list:
        list_checkpoints()
        return
    
    # Download all
    if args.all:
        download_all(args.cache_dir, args.verify)
        return
    
    # Download specific checkpoint
    if args.model:
        checkpoint_path = download_checkpoint(args.model, args.cache_dir, args.verify)
        
        if checkpoint_path:
            print_header("Download Complete")
            print(f"Checkpoint ready at: {checkpoint_path}")
            print(f"\nNext steps:")
            print(f"1. Load in code: model = load_model('{checkpoint_path}')")
            print(f"2. Run evaluation: python evaluate.py --checkpoint {checkpoint_path}")
            print(f"3. Start inference: python inference.py --model {checkpoint_path}")
        else:
            print("\nDownload failed. See troubleshooting in checkpoints/README.md")
            sys.exit(1)
        
        return
    
    # No arguments - show help
    parser.print_help()
    print("\nRecommended: python download_checkpoints.py --model fully_trained")


if __name__ == "__main__":
    main()
