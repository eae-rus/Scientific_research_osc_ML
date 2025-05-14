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
from models.fdd.models import MLP
from common.utils import set_seed, create_experiment_dir, load_config
from datasets.fdd_dataset import FDDDataset


def main():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file")
    parser.add_argument('--experiment_name', type=str, default=None, help="Name for the experiment")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Model name
    model_class_name = config.get('model', {}).get('name', 'No')

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Create experiment directory
    base_results_dir = f"models/{model_class_name}/experiments"
    experiment_dir = create_experiment_dir(base_results_dir, args.experiment_name)

    # Create model
    try:
        model_module = importlib.import_module(f"models.fdd.models")
        model_class = getattr(model_module, model_class_name)
    except (ImportError, AttributeError) as e:
        print(f"Error: Could not import trainer for model '{model_class_name}'")
        print(f"Make sure the path Models exists")
        print(f"and contains a class named {model_class_name}")
        print(f"Error details: {e}")
        sys.exit(1)

    win_size = config.get('data', {}).get('window_size', 32)
    in_dim = config.get('data', {}).get('dim', 5)
    out_dim = config.get('data', {}).get('classes', 3)
    model = model_class(win_size, in_dim, out_dim)

    print(f"Starting training for {model_class_name}")

    # Create dataset
    dataset = FDDDataset('data/csv/dataset.csv')

    # Scale the data
    #scaler = StandardScaler()
    #dataset.df[dataset.train_mask] = scaler.fit_transform(dataset.df[dataset.train_mask])

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

        # Save results to jsonc
        results_path = os.path.join(experiment_dir, "metrics")
        results_file = os.path.join(results_path,
                                    f"evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

        # Print results
        print("\nEvaluation Results:")
        for metric, value in results.get('metrics', {}).items():
            if isinstance(value, (int, float)):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
