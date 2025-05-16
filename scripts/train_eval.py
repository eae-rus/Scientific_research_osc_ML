import argparse
import sys
import importlib
import os
import yaml
from sklearn.preprocessing import StandardScaler
import json

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fdd.trainer import FDDTrainer
from models.fdd.evaluator import FDDEvaluator
from models.fdd.models import MLP, CNN
from models.fdd.advanced_models import LSTM_CNN_Hybrid, TransformerFDD, ResNet1D, DualBranchFDD
from common.utils import set_seed, create_experiment_dir, load_config
from datasets.fdd_dataset import FDDDataset


def create_model(config):
    """Создает модель на основе конфигурации"""
    model_name = config.get('model', {}).get('name', 'CNN')
    window_size = config.get('data', {}).get('window_size', 32)
    input_dim = config.get('data', {}).get('dim', 5)
    output_dim = config.get('data', {}).get('classes', 3)

    print(f"Создание модели: {model_name}")
    print(f"Параметры: window_size={window_size}, input_dim={input_dim}, output_dim={output_dim}")

    if model_name == "MLP":
        return MLP(window_size, input_dim, output_dim)
    elif model_name == "CNN":
        return CNN(window_size, input_dim, output_dim)
    elif model_name == "LSTM_CNN_Hybrid":
        return LSTM_CNN_Hybrid(window_size, input_dim, output_dim)
    elif model_name == "TransformerFDD":
        return TransformerFDD(window_size, input_dim, output_dim)
    elif model_name == "ResNet1D":
        return ResNet1D(window_size, input_dim, output_dim)
    elif model_name == "DualBranchFDD":
        return DualBranchFDD(window_size, input_dim, output_dim)
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    parser.add_argument('--experiment_name', type=str, default=None, help="Name for the experiment")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Model name
    model_class_name = config.get('model', {}).get('name', 'CNN')

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Create experiment directory (используем имя модели для папки)
    base_results_dir = f"models/{model_class_name}/experiments"
    experiment_dir = create_experiment_dir(base_results_dir, args.experiment_name)

    # Create model using new function
    try:
        model = create_model(config)
    except Exception as e:
        print(f"Error: Could not create model '{model_class_name}'")
        print(f"Error details: {e}")
        sys.exit(1)

    print(f"Starting training for {model_class_name}")

    # Create dataset
    dataset = FDDDataset('data/csv/dataset.csv')

    # Scale the data (optional - uncomment if needed)
    # scaler = StandardScaler()
    # dataset.df[dataset.train_mask] = scaler.fit_transform(dataset.df[dataset.train_mask])

    # Create trainer instance with the loaded config
    trainer = FDDTrainer(config=config,
                         model=model,
                         dataset=dataset,
                         experiment_dir=experiment_dir)

    try:
        # Initialize model, datasets, and other components
        trainer.setup()

        # Run the training process
        trainer.train()

        print(f"Training completed successfully! Results saved to {experiment_dir}")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Save the configuration to the experiment directory
    config_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"Starting evaluation for {model_class_name}")

    # Create evaluator instance
    evaluator = FDDEvaluator(config=config,
                             model=model,
                             dataset=dataset,
                             experiment_dir=experiment_dir)
    try:
        # Load the model, Setup data
        evaluator.setup()

        # Run evaluation
        results = evaluator.evaluate()

        # Save results to json
        results_path = os.path.join(experiment_dir, "metrics")
        os.makedirs(results_path, exist_ok=True)  # Убедимся что папка создана
        results_file = os.path.join(results_path, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        # Print results
        print("\nEvaluation Results:")
        for metric, value in results.get('metrics', {}).items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

        # Особо выделим результат для класса 2 (аварии)
        if 'f1_scores' in results.get('metrics', {}):
            f1_scores = results['metrics']['f1_scores']
            if len(f1_scores) >= 3:
                print(f"\n*** F1-score для аварий (класс 2): {f1_scores[2]:.4f} ***")

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()