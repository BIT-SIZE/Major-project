import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from ml.dataset.dataset_inspector import PlantSegDatasetInspector

def main():
    inspector = PlantSegDatasetInspector(root_dir=str(ROOT_DIR))
    report_data = inspector.run_full_inspection(sample_limit_for_pixels=300)
    visual_samples = inspector.generate_visual_samples(num_samples=6)
    inspector.save_reports(report_data, visual_samples)
    print("\n[SUCCESS] Dataset inspection completed successfully!")

if __name__ == "__main__":
    main()
