import torch
import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
import random
from pathlib import Path
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Union
import logging


def get_short_names_ml_signals(use_operational_switching: bool = True, use_abnormal_event: bool = True,
                               use_emergency_event: bool = True) -> tuple:
    """
    This function returns a set of short names ML signals for (without i_bus).

    Args:
        use_operational_switching (bool): Include operational switching signals.
        use_abnormal_event (bool): Include abnormal event signals.
        use_emergency_event (bool): Include emergency event signals.

    Returns:
        tuple: (all_signals, operational_signals, abnormal_signals, emergency_signals)
    """
    ml_operational_switching = [
        'ML_1', 'ML_1_1', 'ML_1_1_1', 'ML_1_2'
    ]

    ml_abnormal_event = [
        'ML_2', 'ML_2_1', 'ML_2_1_1', 'ML_2_1_2', 'ML_2_1_3',
        'ML_2_2', 'ML_2_3', 'ML_2_3_1', 'ML_2_4', 'ML_2_4_1', 'ML_2_4_2'
    ]

    ml_emergency_event = [
        'ML_3', 'ML_3_1', 'ML_3_2', 'ML_3_3'
    ]

    ml_signals = []
    if use_operational_switching:
        ml_signals.extend(ml_operational_switching)
    if use_abnormal_event:
        ml_signals.extend(ml_abnormal_event)
    if use_emergency_event:
        ml_signals.extend(ml_emergency_event)

    return ml_signals, ml_operational_switching, ml_abnormal_event, ml_emergency_event


def get_short_names_ml_analog_signals() -> List[str]:
    """
    This function returns a set of short names ML analog signals.

    Returns:
        list: A set of ML analog signals.
    """
    ml_current = ['IA', 'IB', 'IC', 'IN']
    ml_voltage_BB = ['UA BB', 'UB BB', 'UC BB', 'UN BB', 'UAB BB', 'UBC BB', 'UCA BB']
    ml_voltage_CL = ['UA CL', 'UB CL', 'UC CL', 'UN CL', 'UAB CL', 'UBC CL', 'UCA CL']

    ml_signals = []
    ml_signals.extend(ml_current)
    ml_signals.extend(ml_voltage_BB)
    ml_signals.extend(ml_voltage_CL)

    return ml_signals


def get_ml_signals(i_bus: str, use_operational_switching=True, use_abnormal_event=True,
                   use_emergency_event=True) -> set:
    """
    This function returns a set of ML signals for a given bus.

    Args:
        i_bus (str): The bus number.
        use_operational_switching (bool): Include operational switching signals.
        use_abnormal_event (bool): Include abnormal event signals.
        use_emergency_event (bool): Include emergency event signals.

    Returns:
        set: A set of ML signals for the given bus.
    """
    ml_operational_switching = {
        f'MLsignal_{i_bus}_1', f'MLsignal_{i_bus}_1_1',
        f'MLsignal_{i_bus}_1_1_1', f'MLsignal_{i_bus}_1_2'
    }

    ml_abnormal_event = {
        f'MLsignal_{i_bus}_2', f'MLsignal_{i_bus}_2_1', f'MLsignal_{i_bus}_2_1_1',
        f'MLsignal_{i_bus}_2_1_2', f'MLsignal_{i_bus}_2_1_3', f'MLsignal_{i_bus}_2_2',
        f'MLsignal_{i_bus}_2_3', f'MLsignal_{i_bus}_2_3_1', f'MLsignal_{i_bus}_2_4',
        f'MLsignal_{i_bus}_2_4_1', f'MLsignal_{i_bus}_2_4_2'
    }

    ml_emergency_event = {
        f'MLsignal_{i_bus}_3', f'MLsignal_{i_bus}_3_1',
        f'MLsignal_{i_bus}_3_2', f'MLsignal_{i_bus}_3_3'
    }

    ml_signals = set()
    if use_operational_switching:
        ml_signals.update(ml_operational_switching)
    if use_abnormal_event:
        ml_signals.update(ml_abnormal_event)
    if use_emergency_event:
        ml_signals.update(ml_emergency_event)

    return ml_signals


def get_available_device() -> torch.device:
    """Get available device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")
    return device


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # For Apple Silicon
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    print(f"🎲 Random seed set to {seed}")


def create_experiment_dir(base_dir: Union[str, Path], experiment_name: Optional[str] = None) -> Path:
    """Create experiment directory with timestamp and subdirectories."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{experiment_name}_{timestamp}" if experiment_name else f"experiment_{timestamp}"
    exp_dir = Path(base_dir) / exp_name

    # Create subdirectories
    subdirs = ["checkpoints", "metrics", "logs", "configs", "plots"]
    for subdir in subdirs:
        os.makedirs(exp_dir / subdir, exist_ok=True)

    print(f" Created experiment directory: {exp_dir}")
    return exp_dir


def load_config(config_path: Union[str, Path]) -> Dict:
    """Load configuration from YAML file with validation."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def save_config(config: Dict, save_path: Union[str, Path]):
    """Save configuration to YAML file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=4)

    print(f" Configuration saved to {save_path}")


# ==================== ADVANCED SPLITTING UTILITIES ====================

def analyze_dataset_characteristics(df: pd.DataFrame, target: pd.Series) -> Dict:
    """Analyze dataset characteristics to recommend splitting strategy."""

    file_groups = df.index.get_level_values('file_name').str[:32]
    unique_files = file_groups.unique()

    characteristics = {
        'total_samples': len(df),
        'total_files': len(unique_files),
        'class_distribution': {
            'normal': (target == 0).sum(),
            'anomaly': (target == 1).sum(),
            'emergency': (target == 2).sum()
        },
        'emergency_ratio': (target == 2).sum() / len(target),
        'files_with_emergency': 0,
        'avg_samples_per_file': len(df) / len(unique_files),
        'temporal_span_estimation': estimate_temporal_span(unique_files)
    }

    # Analyze files with emergencies
    for file in unique_files:
        file_mask = file_groups == file
        if (target[file_mask] == 2).any():
            characteristics['files_with_emergency'] += 1

    characteristics['emergency_file_ratio'] = characteristics['files_with_emergency'] / len(unique_files)

    return characteristics


def estimate_temporal_span(file_names: List[str]) -> Optional[int]:
    """Estimate temporal span in days from file names (if they contain dates)."""
    try:
        dates = []
        for fname in file_names:
            # Try to extract date patterns (YYYYMMDD)
            import re
            date_match = re.search(r'(\d{8})', fname)
            if date_match:
                date_str = date_match.group(1)
                date = datetime.strptime(date_str, '%Y%m%d')
                dates.append(date)

        if len(dates) >= 2:
            return (max(dates) - min(dates)).days

    except Exception:
        pass

    return None


def recommend_splitting_strategy(dataset_characteristics: Dict) -> Dict:
    """Recommend optimal splitting strategy based on dataset characteristics."""

    recommendations = {
        'primary_strategy': 'original',
        'alternative_strategies': [],
        'reasoning': [],
        'warnings': []
    }

    emergency_ratio = dataset_characteristics['emergency_ratio']
    emergency_file_ratio = dataset_characteristics['emergency_file_ratio']
    temporal_span = dataset_characteristics.get('temporal_span_estimation')

    # Decision logic
    if emergency_ratio < 0.02:  # Very rare emergencies
        recommendations['primary_strategy'] = 'emergency_stratified'
        recommendations['reasoning'].append("Very low emergency ratio - need emergency stratification")
        recommendations['alternative_strategies'].append('balanced_minority')

    elif emergency_file_ratio < 0.1:  # Few files with emergencies
        recommendations['primary_strategy'] = 'balanced_minority'
        recommendations['reasoning'].append("Few files contain emergencies - ensure representation")
        recommendations['alternative_strategies'].extend(['emergency_stratified', 'clustering_based'])

    elif temporal_span and temporal_span > 30:  # Long temporal span
        recommendations['primary_strategy'] = 'temporal_aware'
        recommendations['reasoning'].append(f"Long temporal span ({temporal_span} days) - use temporal split")
        recommendations['alternative_strategies'].append('emergency_stratified')

    else:  # General case
        recommendations['primary_strategy'] = 'clustering_based'
        recommendations['reasoning'].append("Balanced dataset - use clustering for diversity")
        recommendations['alternative_strategies'].extend(['emergency_stratified', 'original'])

    # Warnings
    if emergency_ratio < 0.005:
        recommendations['warnings'].append("Extremely low emergency ratio - consider data augmentation")

    if dataset_characteristics['total_files'] < 20:
        recommendations['warnings'].append("Small number of files - results may be unstable")

    return recommendations


def validate_split_quality(train_mask: pd.Series, val_mask: pd.Series, test_mask: pd.Series,
                           target: pd.Series) -> Dict:
    """Validate the quality of data splitting."""

    validation_results = {
        'passed': True,
        'checks': {},
        'warnings': [],
        'errors': []
    }

    # Check 1: No empty splits
    splits = {'train': train_mask, 'val': val_mask, 'test': test_mask}
    for split_name, mask in splits.items():
        if mask.sum() == 0:
            validation_results['errors'].append(f"Empty {split_name} split")
            validation_results['passed'] = False

        validation_results['checks'][f'{split_name}_size'] = mask.sum()

    # Check 2: Emergency representation
    for split_name, mask in splits.items():
        emergency_count = (target[mask] == 2).sum()
        total_count = mask.sum()

        if total_count > 0:
            emergency_ratio = emergency_count / total_count
            validation_results['checks'][f'{split_name}_emergency_ratio'] = emergency_ratio

            if emergency_count == 0 and split_name != 'test':
                validation_results['warnings'].append(f"No emergency samples in {split_name} split")
            elif emergency_ratio < 0.001:
                validation_results['warnings'].append(
                    f"Very low emergency ratio in {split_name}: {emergency_ratio:.4f}")

    # Check 3: Class balance consistency
    train_emergency_ratio = validation_results['checks'].get('train_emergency_ratio', 0)
    val_emergency_ratio = validation_results['checks'].get('val_emergency_ratio', 0)
    test_emergency_ratio = validation_results['checks'].get('test_emergency_ratio', 0)

    max_ratio_diff = max(abs(train_emergency_ratio - val_emergency_ratio),
                         abs(train_emergency_ratio - test_emergency_ratio),
                         abs(val_emergency_ratio - test_emergency_ratio))

    if max_ratio_diff > 0.05:  # 5% difference threshold
        validation_results['warnings'].append(f"Large emergency ratio difference between splits: {max_ratio_diff:.3f}")

    validation_results['checks']['max_emergency_ratio_difference'] = max_ratio_diff

    return validation_results


def generate_split_report(dataset_characteristics: Dict, split_validation: Dict,
                          strategy_name: str, experiment_dir: Path):
    """Generate comprehensive split analysis report."""

    report = {
        'timestamp': datetime.now().isoformat(),
        'strategy_used': strategy_name,
        'dataset_characteristics': dataset_characteristics,
        'split_validation': split_validation,
        'recommendations': recommend_splitting_strategy(dataset_characteristics)
    }

    # Save detailed report
    report_path = experiment_dir / "split_analysis_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    # Generate summary
    summary = []
    summary.append("=" * 60)
    summary.append(" DATA SPLITTING ANALYSIS REPORT")
    summary.append("=" * 60)
    summary.append(f"Strategy used: {strategy_name}")
    summary.append(f"Total samples: {dataset_characteristics['total_samples']:,}")
    summary.append(f"Total files: {dataset_characteristics['total_files']}")
    summary.append(f"Emergency ratio: {dataset_characteristics['emergency_ratio']:.4f}")

    summary.append("\n SPLIT SIZES:")
    for check_name, value in split_validation['checks'].items():
        if '_size' in check_name:
            summary.append(f"   {check_name}: {value:,}")

    summary.append("\n EMERGENCY DISTRIBUTION:")
    for check_name, value in split_validation['checks'].items():
        if '_emergency_ratio' in check_name and value is not None:
            summary.append(f"   {check_name}: {value:.4f}")

    if split_validation['warnings']:
        summary.append("\n⚠  WARNINGS:")
        for warning in split_validation['warnings']:
            summary.append(f"   - {warning}")

    if split_validation['errors']:
        summary.append("\n ERRORS:")
        for error in split_validation['errors']:
            summary.append(f"   - {error}")

    summary.append("\n RECOMMENDATIONS:")
    rec = report['recommendations']
    summary.append(f"   Primary strategy: {rec['primary_strategy']}")
    for reason in rec['reasoning']:
        summary.append(f"   - {reason}")

    summary.append("=" * 60)

    summary_text = "\n".join(summary)

    # Save summary
    summary_path = experiment_dir / "split_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_text)

    print(summary_text)
    print(f"\n Split analysis saved to: {report_path}")

    return report


def setup_logging(experiment_dir: Path, log_level: str = "INFO") -> logging.Logger:
    """Setup logging for experiments."""

    log_dir = experiment_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger('fdd_experiment')
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear any existing handlers
    logger.handlers.clear()

    # File handler
    log_filename = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_dir / log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logging initialized. Log file: {log_filename}")

    return logger


def calculate_data_hash(df: pd.DataFrame) -> str:
    """Calculate hash of dataframe for caching purposes."""
    return hashlib.md5(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()


def compare_split_strategies(results_dir: Path) -> Dict:
    """Compare results across different splitting strategies."""

    strategy_results = {}

    # Scan for experiment results
    for experiment_path in results_dir.glob("*/*/metrics/evaluation_results.json"):
        try:
            with open(experiment_path, 'r') as f:
                results = json.load(f)

            # Extract strategy info from path
            path_parts = experiment_path.parts
            if len(path_parts) >= 4:
                model_strategy = path_parts[-4]  # e.g., "ResNet1D_emergency_stratified"

                if '_' in model_strategy:
                    model_name, strategy = model_strategy.split('_', 1)

                    if strategy not in strategy_results:
                        strategy_results[strategy] = []

                    metrics = results.get('metrics', {})
                    f1_scores = metrics.get('f1_scores', [0, 0, 0])

                    strategy_results[strategy].append({
                        'model': model_name,
                        'f1_emergency': f1_scores[2] if len(f1_scores) > 2 else 0,
                        'accuracy': metrics.get('accuracy', 0),
                        'experiment_path': str(experiment_path.parent.parent)
                    })

        except Exception as e:
            print(f"Error processing {experiment_path}: {e}")

    # Calculate statistics
    comparison = {}
    for strategy, results in strategy_results.items():
        if results:
            f1_scores = [r['f1_emergency'] for r in results]
            comparison[strategy] = {
                'count': len(results),
                'mean_f1_emergency': np.mean(f1_scores),
                'std_f1_emergency': np.std(f1_scores),
                'max_f1_emergency': np.max(f1_scores),
                'best_model': max(results, key=lambda x: x['f1_emergency'])['model']
            }

    return comparison


# Backward compatibility aliases
create_exp_dir = create_experiment_dir  # Для совместимости со старым кодом

if __name__ == "__main__":
    # Test functions
    print(" Testing enhanced utilities...")

    # Test device detection
    device = get_available_device()

    # Test seed setting
    set_seed(42)

    # Test directory creation
    test_dir = create_experiment_dir("test_experiments", "utils_test")
    print(f"Test directory created: {test_dir}")

    print("All utilities working correctly!")