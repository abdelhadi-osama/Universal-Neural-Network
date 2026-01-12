import argparse
import yaml
import sys
import os
import pickle
import numpy as np

# Ensure Python can find our modules
sys.path.append(os.getcwd())

# --- IMPORT YOUR CUSTOM MODULES ---
from utils.logger import LoggerSetup
from data_pipeline.data_loader import DataLoader
from data_pipeline.preprocessor import Preprocessor
from pipeline.pipeline import NeuralNetworkPipeline
from evaluation.evaluator import Evaluator

# ---------------------------------------------------------
# STEP 1: DATA CONTROLLER
# ---------------------------------------------------------
def run_data_pipeline(config, logger):
    logger.info(">> PIPELINE STEP 1: Data Pipeline Entry Point")

    # 1. Initialize Loader
    loader = DataLoader(config_path=None, data_dir=config['paths']['raw_data'])
    
    # --- CRITICAL FIX: CONFIG WRAPPING ---
    # The DataLoader expects: config['data_pipeline']['datasets'][name]
    # But we only have the specific dataset config.
    # We reconstruct the structure it expects to prevent the crash.
    loader_friendly_config = config.copy()
    
    # Flatten: Ensure 'files' and 'urls' are visible where DataLoader looks
    if 'data_config' in config:
        loader_friendly_config.update(config['data_config'])

    # Wrap: Create the 'data_pipeline' -> 'datasets' hierarchy
    loader.config = {
        'data_pipeline': {
            'datasets': {
                config['dataset_name']: loader_friendly_config
            }
        }
    }
    
    # 2. Load Data (Pass the NAME as a String)
    logger.info(f"   [Action] Requesting dataset: {config['dataset_name']}")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = loader.load_dataset(config['dataset_name'])
    
    # 3. Preprocessing (Normalization)
    prep = Preprocessor()
    prep.fit(x_train)
    
    flatten_needed = (config.get('type') == 'image_flattened')
    x_train_norm = prep.transform(x_train, flatten=flatten_needed)
    x_val_norm   = prep.transform(x_val,   flatten=flatten_needed)
    x_test_norm  = prep.transform(x_test,  flatten=flatten_needed)

    # 4. Label Encoding Strategy
# Look inside data_config because that is where it is defined in YAML
    encoding_rule = config.get('target_encoding') or config.get('data_config', {}).get('target_encoding')  
      
    # CASE A: Dictionary Mapping (Breast Cancer)
    if isinstance(encoding_rule, dict):
        logger.info(f"   [Strategy] Applying Binary Mapping: {encoding_rule}")
        def apply_map(y, mapping):
            y_enc = np.zeros(y.shape, dtype=int)
            for label_name, label_val in mapping.items():
                y_enc[y == label_name] = label_val
            return y_enc.reshape(-1, 1)

        y_train_enc = apply_map(y_train, encoding_rule)
        y_val_enc   = apply_map(y_val, encoding_rule)
        y_test_enc  = apply_map(y_test, encoding_rule)

    # CASE B: One-Hot (MNIST)
    elif encoding_rule == 'one_hot':
        logger.info("   [Strategy] Applying One-Hot Encoding via Preprocessor")
        y_train_enc = prep.fit_encode_labels(y_train)
        y_val_enc   = prep.encode_labels(y_val)
        y_test_enc  = prep.encode_labels(y_test)

    else:
        y_train_enc, y_val_enc, y_test_enc = y_train, y_val, y_test

    logger.info(f"   [State] Data Ready. Final Train Shape: {x_train_norm.shape}")
    return (x_train_norm, y_train_enc), (x_val_norm, y_val_enc), (x_test_norm, y_test_enc)

# ---------------------------------------------------------
# STEP 3: MODEL CONTROLLER (TRAINING)
# ---------------------------------------------------------
def run_model_pipeline(config, data_bundle, logger):
    logger.info(">> PIPELINE STEP 3: Model Initialization & Training")

    # 1. Unpack Data to determine Input Size
    (x_train, y_train), (x_val, y_val), _ = data_bundle
    input_dim = x_train.shape[1]
    
    # 2. Build Dynamic Architecture [Input, ...Hidden..., Output]
    defined_arch = config['model']['architecture']
    full_layer_sizes = [input_dim] + defined_arch
    
    # 3. Prepare Scheduler Settings
    sched_config = config['scheduler'].copy()
    sched_method = sched_config.pop('method')

    # 4. Initialize Pipeline
    pipeline = NeuralNetworkPipeline(
        layer_sizes=full_layer_sizes,
        activations=config['model']['activations'],
        loss_type=config['model']['loss_function'],
        optimizer_method=config['optimizer']['method'],
        lr=config['training']['learning_rate'],
        beta1=config['optimizer']['beta1'],
        beta2=config['optimizer']['beta2'],
        epsilon=config['optimizer']['epsilon'],
        regularization=config['model']['regularization'],
        l_lambda=config['model']['l_lambda'],
        dropout_rates=config['model']['dropout_rates'],
        lr_scheduler=sched_method,
        **sched_config
    )

    # 5. Train
    logger.info(f"   [Execution] Training for {config['training']['epochs']} Epochs...")
    pipeline.train(x_train, y_train, x_val=x_val, y_val=y_val, epochs=config['training']['epochs'], verbose=True)
    
    return pipeline

# ---------------------------------------------------------
# STEP 4: EVALUATION & SAVING
# ---------------------------------------------------------
# ---------------------------------------------------------
# STEP 4: EVALUATION & SAVING
# ---------------------------------------------------------
def run_evaluation_pipeline(pipeline, data_bundle, config, logger):
    logger.info(">> PIPELINE STEP 4: Final Evaluation & Saving")

    # 1. Unpack Test Data
    _, _, (x_test, y_test) = data_bundle
    
    # 2. Run Evaluator
    evaluator = Evaluator(pipeline)
    
    # A. Text Metrics
    metrics = evaluator.evaluate(x_test, y_test)
    logger.info(f"   [Report Card] Test Accuracy: {metrics['accuracy']*100:.2f}%")
    logger.info(f"   [Report Card] F1 Score:      {metrics['f1_score']:.4f}")

    # --- B. VISUAL EVALUATION ---
    logger.info("   [Visuals] Generating plots...")
    
    # Check if Binary (Cancer) or Multi-class (MNIST)
    # If shape is (N, 1), it's binary. If (N, 10), it's multi-class.
    is_binary = (y_test.shape[1] == 1) if y_test.ndim > 1 else True

    try:
        # 1. Loss & Accuracy Curves (ALWAYS SHOW)
        evaluator.plot_history()
        
        # 2. Confusion Matrix (ONLY FOR CANCER)
        if is_binary:
            logger.info("   [Visuals] Showing Confusion Matrix for Binary Classification...")
            evaluator.plot_confusion_matrix(x_test, y_test)
            
            # 3. ROC Curve (ONLY FOR CANCER)
            evaluator.plot_roc_curve(x_test, y_test)
        else:
            logger.info("   [Visuals] Skipping Confusion Matrix pop-up for MNIST (Multi-class).")

    except Exception as e:
        logger.warning(f"   [Visuals] Could not plot (missing GUI?): {e}")

    # 3. Save Model
    save_dir = config['paths']['saved_models']
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    filename = f"{config['dataset_name']}_model.pkl"
    save_path = os.path.join(save_dir, filename)
    
    with open(save_path, 'wb') as f:
        pickle.dump(pipeline, f)
        
    logger.info(f"   [Persistence] Model saved to: {save_path}")
# ---------------------------------------------------------
# MAIN ORCHESTRATOR
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yml')
    parser.add_argument('--dataset', type=str, default='breast_cancer')
    args = parser.parse_args()

    logger = LoggerSetup.setup_logger()
    logger.info(f"=== STARTING RUN: {args.dataset} ===")

    try:
        # STEP 0: LOAD CONFIG
        if not os.path.exists(args.config): raise FileNotFoundError(args.config)
        with open(args.config, 'r') as f: full_config = yaml.safe_load(f)
        
        # Inject Settings
        if args.dataset not in full_config['datasets']: raise ValueError("Invalid dataset name")
        
        config = full_config['datasets'][args.dataset]
        config['paths'] = full_config['paths']
        config['optimizer'] = full_config['optimizer']
        config['scheduler'] = full_config['scheduler']
        config['dataset_name'] = args.dataset # <--- Important for saving name

        # STEP 1: DATA
        data_bundle = run_data_pipeline(config, logger)
        
        # STEP 3: MODEL (Build & Train)
        pipeline = run_model_pipeline(config, data_bundle, logger)
        
        # STEP 4: EVALUATE & SAVE
        run_evaluation_pipeline(pipeline, data_bundle, config, logger)
        
        logger.info("=== RUN SUCCESSFUL ===")

    except Exception as e:
        logger.error(f"Pipeline Crashed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
