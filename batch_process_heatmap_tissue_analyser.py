#!/usr/bin/env python3
"""
Batch process slides through complete pipeline:
1. Find TIFF files from slide IDs
2. Run CLAM superpixel analysis
3. Run intact tissue analysis with WM masks
4. Generate comprehensive reports

Input format: slide_DI_123456_ABC-01-LFB+CV
File location: <ARCHIVE_ROOT>/INDD123456/histo_raw/DI_123456_ABC-01-LFB+CV.tif
# """
# when you have the WM files
# python batch_process_heatmap_tissue_analyser.py --input cases.txt --intact_only --percentile 2
# when you want to get the attention scores
# python batch_process_heatmap_tissue_analyser.py --input cases.txt --clam_only
# python batch_process_heatmap_tissue_analyser.py --input cases.txt --out_dir phas_clam_outputs/evaluation2/  --clam_only

# TO GET EXACTLY 5 SUPERPIXELS WITH LOWEST ATTENTION:
# python batch_process_heatmap_tissue_analyser.py --input cases.txt --intact_only --n_tiles 5


# conda activate clam_latest
# export CC=/usr/bin/gcc
# export CXX=/usr/bin/g++


import glob
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Base paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ARCHIVE_ROOT = "/path/to/archive"  # Set to your archive root directory
CLAM_SCRIPT = os.path.join(_SCRIPT_DIR, "ClamResumeSuperpixels.py")
INTACT_SCRIPT = os.path.join(_SCRIPT_DIR, "intact_tissue_analyzer.py")

# Default parameters
DEFAULT_CKPT = os.path.join(_SCRIPT_DIR, "weights", "s_2_checkpoint.pt")
DEFAULT_RESNET50 = os.path.join(_SCRIPT_DIR, "weights", "resnet50-11ad3fa6.pth")
DEFAULT_OUT_DIR = "./phas_clam_outputs"
DEFAULT_PERCENTILE = 10

class SlideInfo:
    """Parse and store slide information"""
    def __init__(self, slide_id, archive_root):
        """
        Parse slide_id formats:
        - slide_DI_123456_ABC-01-LFB+CV  (original format)
        - slide_DI_000099R_04S_07_LFBCV  (new format)
        """
        self.archive_root = archive_root
        
        # Remove 'slide_' prefix if present
        if slide_id.startswith("slide_"):
            slide_id = slide_id[6:]
        
        self.full_id = slide_id
        
        # Parse: DI_...
        if not slide_id.startswith("DI_"):
            raise ValueError(f"Invalid slide ID format: {slide_id} (must start with DI_)")
        
        # Remove DI_ prefix
        remainder = slide_id[3:]
        
        # Split by underscore to get parts
        parts = remainder.split("_")
        
        if len(parts) < 2:
            raise ValueError(f"Cannot parse slide ID: {slide_id} (need at least 2 parts after DI_)")
        
        # Detect format based on structure
        # Format 1: 123456_ABC-01-LFB+CV (has hyphens)
        # Format 2: 000099R_04S_07_LFBCV (no hyphens, multiple underscores)
        
        if '-' in remainder:
            # Original format: DI_123456_ABC-01-LFB+CV
            self.sample_number = parts[0]  # 123456
            suffix_with_hyphens = "_".join(parts[1:])  # ABC-01-LFB+CV
            
            suffix_parts = suffix_with_hyphens.split("-")
            if len(suffix_parts) < 2:
                raise ValueError(f"Cannot parse suffix: {suffix_with_hyphens}")
            
            self.location = suffix_parts[0]  # ABC
            self.slide_number = suffix_parts[1]  # 01
            self.stain = "-".join(suffix_parts[2:]) if len(suffix_parts) > 2 else "LFB+CV"
            self.suffix = suffix_with_hyphens
            
        else:
            # New format: DI_000099R_04S_07_LFBCV
            # Parts: [000099R, 04S, 07, LFBCV]
            if len(parts) < 4:
                raise ValueError(f"Cannot parse new format: {slide_id} (expected 4+ parts)")
            
            self.sample_number = parts[0]  # 000099R
            self.location = parts[1]       # 04S
            self.slide_number = parts[2]   # 07
            self.stain = "_".join(parts[3:])  # LFBCV (or LFBCV_extra if more parts)
            self.suffix = "_".join(parts[1:])  # 04S_07_LFBCV
        
        # Construct paths
        # Remove any letter suffix from sample number for directory (000099R -> 000099)
        sample_num_clean = ''.join(c for c in self.sample_number if c.isdigit())
        self.indd_dir = f"INDD{sample_num_clean}"
        self.tiff_filename = f"DI_{self.sample_number}_{self.suffix}.tif"
        
    def get_tiff_path(self):
        """Get full path to TIFF file"""
        return os.path.join(
            self.archive_root,
            self.indd_dir,
            "histo_raw",
            self.tiff_filename
        )
    
    def __repr__(self):
        return f"SlideInfo({self.full_id})"

def find_and_validate_files(slide_ids, wm_mask_dir, archive_root, out_dir, skip_wm_mask=False, is_intact_only=False):
    """
    Parse slide IDs and validate files exist
    
    Args:
        slide_ids: List of slide ID strings
        wm_mask_dir: Optional directory containing WM masks
        archive_root: Root directory for archives
        out_dir: CLAM output directory (to search for WM masks)
        skip_wm_mask: If True, don't validate WM masks (for --clam_only)
        is_intact_only: If True, show different status message
    
    Returns:
        List of (SlideInfo, tiff_path, wm_mask_path, status) tuples
    """
    results = []
    
    for slide_id in slide_ids:
        try:
            info = SlideInfo(slide_id.strip(), archive_root)
            tiff_path = info.get_tiff_path()
            
            # Check TIFF exists
            if not os.path.exists(tiff_path):
                results.append((info, tiff_path, None, f"TIFF not found: {tiff_path}"))
                continue
            
            # Skip WM mask validation if only running CLAM
            if skip_wm_mask:
                status_msg = "OK (WM mask will be found in run_dir)" if is_intact_only else "OK (WM mask not required)"
                results.append((info, tiff_path, None, status_msg))
                continue
            
            # Initialize wm_mask_path
            wm_mask_path = None
                        
            # Find WM mask
            if wm_mask_dir:
                # User provided a specific directory with WM masks
                tif_candidates = glob.glob(os.path.join(wm_mask_dir, "WM*.tif"))
                tif_candidates += glob.glob(os.path.join(wm_mask_dir, "wm*.tif"))
                if tif_candidates:
                    wm_mask_path = tif_candidates[0]  # choose first match
            else:
                # No wm_mask_dir provided - check CLAM output directory
                slide_name = f"slide_{info.full_id}"
                
                # Try both L0 and L1 directories
                possible_run_dirs = [
                    os.path.join(out_dir, slide_name, "L0_T256_S256"),
                    os.path.join(out_dir, slide_name, "L1_T256_S256"),
                ]
                
                for run_dir in possible_run_dirs:
                    if os.path.exists(run_dir):
                        # Look for WM*.tif in this directory
                        wm_masks = glob.glob(os.path.join(run_dir, "WM*.tif"))
                        wm_masks += glob.glob(os.path.join(run_dir, "wm*.tif"))
                        if wm_masks:
                            wm_mask_path = wm_masks[0]
                            break

            # Handle missing file
            if not wm_mask_path or not os.path.exists(wm_mask_path):
                results.append((info, tiff_path, wm_mask_path, f"WM mask not found: {wm_mask_path}"))
                continue

            # Success
            results.append((info, tiff_path, wm_mask_path, "OK"))
            
        except Exception as e:
            results.append((None, None, None, f"Parse error: {str(e)}"))
    
    return results

def run_clam_analysis(tiff_path, info, args):
    """
    Run CLAM superpixel analysis
    
    Returns:
        (success, run_dir, message)
    """
    slide_name = f"slide_{info.full_id}"
    run_dir = os.path.join(args.out_dir, slide_name, f"L0_T256_S256")
    
    # Check if already processed and skip if requested
    if args.resume and os.path.exists(os.path.join(run_dir, "superpixels_labels_thumb.npy")):
        print(f"[SKIP] {slide_name} - superpixel analysis already complete")
        return True, run_dir, "Already processed (resumed)"
    
    print(f"\n{'='*80}")
    print(f"[CLAM] Processing {slide_name}")
    print(f"{'='*80}")
    
    cmd = [
        "python", CLAM_SCRIPT,
        "--image", tiff_path,
        "--ckpt", args.ckpt,
        "--resnet50", args.resnet50,
        "--out", args.out_dir,
        "--tile", "256",
        "--stride", "256",
        "--sp_enable",
        "--sp_n", str(args.sp_n),
    ]
    
    if args.resume:
        cmd.extend(["--resume_tiles", "--resume_feats", "--resume_attn"])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"[CLAM] ✓ Success: {slide_name}")
        return True, run_dir, "Success"
    except subprocess.CalledProcessError as e:
        print(f"[CLAM] ✗ Failed: {slide_name}")
        return False, run_dir, f"CLAM failed: {e}"

def find_wm_mask_in_run_dir(run_dir, slide_id):
    """
    Find WM mask in the CLAM run directory
    Looks for files starting with 'WM' and ending with '.tif'
    
    Args:
        run_dir: CLAM output directory (e.g., .../L0_T256_S256/)
        slide_id: Slide identifier for logging
    
    Returns:
        path to WM mask or None
    """
    if not os.path.exists(run_dir):
        return None
    
    # List all files starting with WM and ending with .tif
    wm_masks = [f for f in os.listdir(run_dir) if f.startswith('WM') and f.endswith('.tif')]
    
    if len(wm_masks) == 0:
        print(f"[WARNING] No WM mask found in {run_dir} (looking for WM*.tif)")
        return None
    
    if len(wm_masks) > 1:
        print(f"[WARNING] Multiple WM masks found in {run_dir}: {wm_masks}")
        print(f"[WARNING] Using first one: {wm_masks[0]}")
    
    wm_mask_path = os.path.join(run_dir, wm_masks[0])
    print(f"[INFO] Found WM mask for {slide_id}: {wm_masks[0]}")
    return wm_mask_path

def run_intact_analysis(run_dir, tiff_path, wm_mask_path, info, args):
    """
    Run intact tissue analysis
    
    Returns:
        (success, output_dir, message)
    """
    output_dir = os.path.join(run_dir, "intact_analysis")
    
    # Check if already processed
    if args.resume and os.path.exists(os.path.join(output_dir, "mask_intact_tissue.npy")):
        print(f"[SKIP] {info.full_id} - intact analysis already complete")
        return True, output_dir, "Already processed (resumed)"
    
    # Verify run_dir exists and has required files
    required_files = [
        "superpixels_labels_thumb.npy",
        "superpixels_stats.json",
        "superpixels_mask_thumb.npy"
    ]
    
    missing_files = []
    for fname in required_files:
        if not os.path.exists(os.path.join(run_dir, fname)):
            missing_files.append(fname)
    
    if missing_files:
        msg = f"Missing CLAM outputs in {run_dir}: {', '.join(missing_files)}"
        print(f"[INTACT] ✗ {msg}")
        return False, output_dir, msg
    
    print(f"\n{'='*80}")
    print(f"[INTACT] Processing {info.full_id}")
    print(f"[INTACT] Run dir: {run_dir}")
    print(f"[INTACT] WM mask: {wm_mask_path}")
    print(f"{'='*80}")
    
    cmd = [
        "python", INTACT_SCRIPT,
        "--run_dir", run_dir,
        "--wm_mask", wm_mask_path,
        "--tiff_image", tiff_path,
        "--output_dir", output_dir,
    ]
    
    # Add either n_tiles or percentile
    if args.n_tiles is not None:
        cmd.extend(["--n_tiles", str(args.n_tiles)])
        print(f"[INTACT] Using --n_tiles {args.n_tiles}")
    else:
        cmd.extend(["--percentile", str(args.percentile)])
        print(f"[INTACT] Using --percentile {args.percentile}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[INTACT] ✓ Success: {info.full_id}")
        return True, output_dir, "Success"
    except subprocess.CalledProcessError as e:
        print(f"[INTACT] ✗ Failed: {info.full_id}")
        print(f"[INTACT] Error output:")
        print(e.stderr if e.stderr else e.stdout)
        return False, output_dir, f"Intact analysis failed: {e.stderr[:200] if e.stderr else str(e)}"

def generate_report(results, output_file):
    """Generate summary report"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_slides": len(results),
        "successful": sum(1 for r in results if r["status"] == "complete"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "results": results
    }
    
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Total slides:    {report['total_slides']}")
    print(f"Successful:      {report['successful']}")
    print(f"Failed:          {report['failed']}")
    print(f"Skipped:         {report['skipped']}")
    print(f"\nDetailed report: {output_file}")
    print(f"{'='*80}\n")
    
    return report

def main():
    parser = argparse.ArgumentParser(
        description="Batch process slides through CLAM and intact tissue analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # From file with percentile
  python batch_process_slides.py --input slides.txt --wm_mask_dir /path/to/masks
  
  # From file with exact number of tiles (5 superpixels with lowest attention)
  python batch_process_slides.py --input slides.txt --intact_only --n_tiles 5
  
  # From command line
  python batch_process_slides.py --slides "DI_123456_ABC-01-LFB+CV" "DI_789012_DEF-02-LFB+CV"
  
Input file format (one per line):
  slide_DI_123456_ABC-01-LFB+CV
  DI_789012_DEF-02-LFB+CV
  slide_DI_345678_GHI-03-LFB+CV
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", "-i", help="File containing slide IDs (one per line)")
    input_group.add_argument("--slides", "-s", nargs="+", help="Slide IDs as arguments")
    
    # File paths
    parser.add_argument("--wm_mask_dir", help="Directory containing WM masks (optional - will check CLAM run_dir first)")
    parser.add_argument("--archive_root", default=DEFAULT_ARCHIVE_ROOT, 
                       help=f"Archive root directory (default: {DEFAULT_ARCHIVE_ROOT})")
    
    # CLAM parameters
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to CLAM checkpoint")
    parser.add_argument("--resnet50", default=DEFAULT_RESNET50, help="Path to ResNet50 weights")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--sp_n", type=int, default=2000, help="Number of superpixels")
    
    # Intact analysis parameters - MODIFIED TO SUPPORT --n_tiles
    parser.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE,
                       help="Bottom X%% attention for intact tissue (ignored if --n_tiles specified)")
    parser.add_argument("--n_tiles", type=int, default=None,
                       help="Use exactly N superpixels with lowest attention (overrides --percentile)")
    
    # Processing options
    parser.add_argument("--resume", action="store_true", 
                       help="Skip already processed slides")
    parser.add_argument("--clam_only", action="store_true",
                       help="Only run CLAM analysis (skip intact analysis)")
    parser.add_argument("--intact_only", action="store_true",
                       help="Only run intact analysis (assume CLAM already done)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Validate files only, don't process")
    
    # Output
    parser.add_argument("--report", default="batch_report.json",
                       help="Output report file")
    
    args = parser.parse_args()
    
    # Load slide IDs
    if args.input:
        with open(args.input, "r") as f:
            slide_ids = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        slide_ids = args.slides
    
    print(f"[INFO] Processing {len(slide_ids)} slides")
    
    # Check if WM masks are required
    need_wm_masks = not args.clam_only
    
    if need_wm_masks and not args.wm_mask_dir and not args.intact_only:
        print("[INFO] No --wm_mask_dir specified.")
        print("[INFO] Will look for WM masks (WM*.tif) in CLAM output directories.")
    
    # Print selection method
    if not args.clam_only:
        if args.n_tiles is not None:
            print(f"[INFO] Will select exactly {args.n_tiles} superpixels with lowest attention per slide")
        else:
            print(f"[INFO] Will select bottom {args.percentile}% of superpixels per slide")
    
    # Find and validate files
    print(f"[INFO] Validating files...")
    # For intact_only, we don't validate WM masks here since they're in run_dir
    skip_wm_validation = args.clam_only or args.intact_only
    validated = find_and_validate_files(slide_ids, args.wm_mask_dir, args.archive_root, 
                                       args.out_dir,
                                       skip_wm_mask=skip_wm_validation,
                                       is_intact_only=args.intact_only)
    
    # Print validation results
    print(f"\n{'='*80}")
    print("FILE VALIDATION")
    print(f"{'='*80}")
    for i, (info, tiff_path, wm_mask_path, status) in enumerate(validated, 1):
        if info:
            print(f"\n{i}. {info.full_id}")
            print(f"   TIFF: {tiff_path}")
            print(f"   WM Mask: {wm_mask_path}")
            print(f"   Status: {status}")
        else:
            print(f"\n{i}. Error: {status}")
    
    # Count valid slides (status starts with "OK")
    valid_slides = [v for v in validated if v[3].startswith("OK")]
    print(f"\n{'='*80}")
    print(f"Valid slides: {len(valid_slides)} / {len(validated)}")
    print(f"{'='*80}\n")
    
    if args.dry_run:
        print("[INFO] Dry run complete. Exiting.")
        return
    
    if not valid_slides:
        print("[ERROR] No valid slides to process!")
        return
    
    # Additional check for intact_only - verify CLAM outputs and WM masks exist
    if args.intact_only:
        print(f"\n[INFO] Checking for existing CLAM results and WM masks...")
        slides_ready = []
        
        for info, tiff_path, wm_mask_path, _ in valid_slides:
            slide_name = f"slide_{info.full_id}"
            base_run_dir = os.path.join(args.out_dir, slide_name)
            possible_dirs = [
                os.path.join(base_run_dir, "L0_T256_S256"),
                os.path.join(base_run_dir, "L1_T256_S256"),
            ]
            
            run_dir_found = None
            for possible_dir in possible_dirs:
                if os.path.exists(possible_dir):
                    run_dir_found = possible_dir
                    break
            
            if not run_dir_found:
                print(f"  ✗ {info.full_id}: No CLAM results found")
                continue
            
            # Check for WM mask in run directory
            wm_mask_in_dir = find_wm_mask_in_run_dir(run_dir_found, info.full_id)
            
            if not wm_mask_in_dir:
                print(f"  ✗ {info.full_id}: CLAM results found but no WM mask (WM*.tif)")
                continue
            
            print(f"  ✓ {info.full_id}: Ready (CLAM + WM mask found)")
            slides_ready.append((info, tiff_path, wm_mask_path, "OK"))
        
        print(f"\n[INFO] Ready to process: {len(slides_ready)}/{len(valid_slides)} slides")
        
        if len(slides_ready) == 0:
            print("[ERROR] No slides ready for intact analysis!")
            print("[ERROR] Need: CLAM results + WM*.tif files in run directories")
            return
        
        # Replace valid_slides with only the ready ones
        valid_slides = slides_ready
    
    # Process each slide
    results = []
    
    for info, tiff_path, wm_mask_path, _ in valid_slides:
        result = {
            "slide_id": info.full_id,
            "sample_number": info.sample_number,
            "location": info.location,
            "slide_number": info.slide_number,
            "tiff_path": tiff_path,
            "wm_mask_path": wm_mask_path,
            "status": "unknown",
            "messages": []
        }
        
        try:
            # Run CLAM analysis
            if not args.intact_only:
                success, run_dir, msg = run_clam_analysis(tiff_path, info, args)
                result["clam_run_dir"] = run_dir
                result["clam_status"] = "success" if success else "failed"
                result["messages"].append(f"CLAM: {msg}")
                
                if not success and not args.clam_only:
                    result["status"] = "failed"
                    results.append(result)
                    continue
            else:
                # Assume CLAM already done
                slide_name = f"slide_{info.full_id}"
                
                # Try to find the run directory - could be L0 or L1
                base_run_dir = os.path.join(args.out_dir, slide_name)
                possible_dirs = [
                    os.path.join(base_run_dir, "L0_T256_S256"),
                    os.path.join(base_run_dir, "L1_T256_S256"),
                ]
                
                run_dir = None
                for possible_dir in possible_dirs:
                    if os.path.exists(possible_dir):
                        run_dir = possible_dir
                        break
                
                if not run_dir:
                    result["status"] = "failed"
                    result["messages"].append(f"CLAM run_dir not found. Searched: {possible_dirs}")
                    results.append(result)
                    continue
                
                print(f"[INFO] Found CLAM results at: {run_dir}")
            
            # Run intact analysis
            if not args.clam_only:
                # Find WM mask - check run_dir first, then wm_mask_dir
                wm_mask_to_use = None
                
                # First try: look in the CLAM run directory
                wm_mask_to_use = find_wm_mask_in_run_dir(run_dir, info.full_id)
                
                # Second try: use the one from validation (if provided via --wm_mask_dir)
                if not wm_mask_to_use and wm_mask_path and os.path.exists(wm_mask_path):
                    wm_mask_to_use = wm_mask_path
                    print(f"[INFO] Using WM mask from --wm_mask_dir: {wm_mask_path}")
                
                if not wm_mask_to_use:
                    result["status"] = "failed"
                    result["messages"].append("WM mask not found in run_dir or wm_mask_dir")
                    results.append(result)
                    continue
                
                success, output_dir, msg = run_intact_analysis(
                    run_dir, tiff_path, wm_mask_to_use, info, args
                )
                result["intact_output_dir"] = output_dir
                result["intact_status"] = "success" if success else "failed"
                result["messages"].append(f"Intact: {msg}")
                
                if not success:
                    result["status"] = "failed"
                else:
                    result["status"] = "complete"
            else:
                result["status"] = "complete"
            
        except Exception as e:
            result["status"] = "failed"
            result["messages"].append(f"Exception: {str(e)}")
        
        results.append(result)
    
    # Generate report
    report = generate_report(results, args.report)
    
    # Exit with error if any failed
    if report["failed"] > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()