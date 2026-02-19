"""
MLflow Model Comparison Dashboard
Shows all 4 models' performances side-by-side for easy comparison
Useful for demonstrating project progress to professors
"""

import mlflow
from pathlib import Path
import json
from datetime import datetime


def get_best_run_for_experiment(experiment_name: str) -> dict:
    """Get the best performing run from an experiment"""
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return {"status": "No experiment found", "experiment": experiment_name}
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        
        if len(runs) == 0:
            return {"status": "No runs found", "experiment": experiment_name}
        
        # Get best run by accuracy
        if "metrics.test_accuracy" in runs.columns:
            best_idx = runs["metrics.test_accuracy"].idxmax()
            best_run = runs.loc[best_idx]
            
            return {
                "status": "success",
                "experiment": experiment_name,
                "run_id": best_run["run_id"],
                "run_name": best_run.get("tags.mlflow.runName", ""),
                "accuracy": best_run.get("metrics.test_accuracy", 0),
                "precision": best_run.get("metrics.test_precision", 0),
                "recall": best_run.get("metrics.test_recall", 0),
                "f1_score": best_run.get("metrics.test_f1", 0),
                "start_time": best_run.get("start_time", ""),
            }
        else:
            return {"status": "No metrics found", "experiment": experiment_name}
    
    except Exception as e:
        return {"status": f"Error: {str(e)}", "experiment": experiment_name}


def print_model_comparison():
    """Print comparison of all 4 models"""
    print("\n" + "=" * 100)
    print("🎯 LEAGUE OF LEGENDS ML MODELS - PERFORMANCE COMPARISON")
    print("=" * 100)
    
    # MLflow experiments for all 4 models
    experiments = [
        ("rank-tier-classification", "Model 1: Rank Tier Classifier", "9-class classification"),
        ("progression-regression", "Model 2: Rank Progression Predictor", "Win rate regression"),
        ("smurf-anomaly-detection", "Model 3: Smurf Anomaly Detector", "Anomaly detection"),
        ("match-outcome-prediction", "Model 4: Match Outcome Predictor", "Binary classification"),
    ]
    
    results = []
    
    for exp_name, display_name, task_desc in experiments:
        print(f"\n📊 {display_name}")
        print(f"   Task: {task_desc}")
        print("-" * 100)
        
        result = get_best_run_for_experiment(exp_name)
        
        if result["status"] == "success":
            print(f"   ✅ Best Run: {result['run_name']}")
            print(f"   📈 Accuracy:  {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
            print(f"   📊 Precision: {result['precision']:.4f}")
            print(f"   🎯 Recall:    {result['recall']:.4f}")
            print(f"   ⚡ F1-Score:   {result['f1_score']:.4f}")
            results.append({
                "model": display_name,
                "accuracy": result['accuracy'],
                "precision": result['precision'],
                "recall": result['recall'],
                "f1": result['f1_score']
            })
        else:
            print(f"   ⚠️  {result['status']}")
    
    # Summary table
    if results:
        print("\n" + "=" * 100)
        print("📋 SUMMARY TABLE")
        print("=" * 100)
        print(f"{'Model':<40} {'Accuracy':<15} {'Precision':<15} {'Recall':<15} {'F1-Score':<15}")
        print("-" * 100)
        
        for r in results:
            model = r['model'].replace("Model 1: ", "").replace("Model 2: ", "").replace("Model 3: ", "").replace("Model 4: ", "")[:35]
            print(f"{model:<40} {r['accuracy']:.4f} ({r['accuracy']*100:5.2f}%)  "
                  f"{r['precision']:.4f}         {r['recall']:.4f}         {r['f1']:.4f}")
        
        # Calculate average accuracy
        avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
        print("-" * 100)
        print(f"{'Average Performance':<40} {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
    
    print("\n" + "=" * 100)
    print("🎓 PROFESSOR DEMO TIPS")
    print("=" * 100)
    print("""
    💡 Key Points to Mention:
    
    1. Data Pipeline
       - Collected 4,340+ players from Riot API
       - Engineered 30+ gameplay features
       - Stratified train/test split (80/20)
    
    2. Model Diversity
       - Classification (rank tier prediction)
       - Regression (progression forecasting)
       - Anomaly detection (smurf identification)
       - Binary classification (match outcomes)
    
    3. MLflow Integration
       - Automated experiment tracking
       - Reproducible results with versioning
       - Easy hyperparameter comparison
       - Artifacts (visualizations, models)
    
    4. Real-World Applications
       - ✅ Smurf detection (Model 3)
       - ✅ Ranked matchmaking verification
       - ✅ Player progression analytics
       - ✅ Match outcome prediction for tournaments
    """)
    
    print("=" * 100 + "\n")


def export_comparison_json():
    """Export comparison results to JSON"""
    experiments = [
        "rank-tier-classification",
        "progression-regression", 
        "smurf-anomaly-detection",
        "match-outcome-prediction"
    ]
    
    export_data = {
        "generated_at": datetime.now().isoformat(),
        "models": []
    }
    
    for exp_name in experiments:
        result = get_best_run_for_experiment(exp_name)
        export_data["models"].append(result)
    
    # Save to file
    output_path = Path("mlflow_comparison_export.json")
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"✅ Comparison exported to: {output_path}")


if __name__ == "__main__":
    # Set MLflow tracking URI (local by default)
    mlflow.set_tracking_uri("http://localhost:5000")
    
    print("\n🔍 Fetching model performance from MLflow...")
    print("(Make sure 'mlflow ui --port 5000' is running)\n")
    
    # Display comparison
    print_model_comparison()
    
    # Export to JSON
    try:
        export_comparison_json()
    except Exception as e:
        print(f"⚠️  Could not export JSON: {e}")
