"""
Generate visualization comparing Model V1 vs V2 performance
"""

import matplotlib.pyplot as plt
import numpy as np

def plot_model_comparison():
    """Create bar chart comparing Model 1 and Model 2 performance"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Model 1: Rank Tier Classifier
    models = ['V1 (Original)', 'V2 (Enriched)']
    accuracies = [0.5311, 0.6521]
    colors = ['#3498db', '#2ecc71']
    
    bars1 = ax1.bar(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylim(0, 1.0)
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model 1: Rank Tier Classifier\nAccuracy Improvement', 
                  fontsize=14, fontweight='bold')
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% baseline')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add percentage labels on bars
    for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc*100:.2f}%',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # Add improvement annotation
    ax1.annotate('', xy=(1, 0.6521), xytext=(0, 0.5311),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax1.text(0.5, 0.59, '+12.10 pp\n(+22.8%)', 
            ha='center', va='bottom', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            fontweight='bold')
    
    # Model 2: Progression Regressor
    r2_scores = [0.3574, 0.3572]
    colors2 = ['#3498db', '#e74c3c']
    
    bars2 = ax2.bar(models, r2_scores, color=colors2, edgecolor='black', linewidth=1.5)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax2.set_title('Model 2: Progression Regressor\nR² Score Comparison', 
                  fontsize=14, fontweight='bold')
    ax2.axhline(y=0.3574, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add percentage labels on bars
    for i, (bar, r2) in enumerate(zip(bars2, r2_scores)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{r2:.4f}\n({r2*100:.2f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add "no change" annotation
    ax2.text(0.5, 0.32, 'Minimal change\n(-0.02 pp)', 
            ha='center', va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7),
            fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: model_performance_comparison.png")
    plt.show()


def plot_feature_counts():
    """Create bar chart comparing feature counts"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Model 1\n(Rank Tier)', 'Model 2\n(Progression)']
    v1_features = [31, 12]
    v2_features = [40, 17]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, v1_features, width, label='V1 (Original)', 
                   color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, v2_features, width, label='V2 (Enriched)', 
                   color='#2ecc71', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_title('Feature Count Comparison: V1 vs V2', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add improvement annotations
    ax.annotate('', xy=(0.175, 40), xytext=(0.175, 31),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(0.175, 35.5, '+9', ha='center', fontsize=10, 
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    ax.annotate('', xy=(1.175, 17), xytext=(1.175, 12),
                arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    ax.text(1.175, 14.5, '+5', ha='center', fontsize=10, 
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('feature_count_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_count_comparison.png")
    plt.show()


if __name__ == '__main__':
    print("\n" + "="*80)
    print("GENERATING MODEL COMPARISON VISUALIZATIONS")
    print("="*80 + "\n")
    
    plot_model_comparison()
    plot_feature_counts()
    
    print("\n" + "="*80)
    print("✅ VISUALIZATION COMPLETE")
    print("="*80)
    print("\nGenerated files:")
    print("  • model_performance_comparison.png")
    print("  • feature_count_comparison.png")
