import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from ml.evaluation.compare_models import ModelEvaluatorComparator

def main():
    comparator = ModelEvaluatorComparator(root_dir=str(ROOT_DIR))
    comparator.run_comparison(max_test_samples=40, batch_size=4)

if __name__ == "__main__":
    main()
