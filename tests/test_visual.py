# evaluation/visualize_results.py
"""
Generate beautiful visualizations for evaluation results
Creates charts for your application/presentation
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


class ResultsVisualizer:
    """Create visualizations from evaluation results"""
    
    def __init__(self, results_path: str = "./evaluation/results.json"):
        with open(results_path, 'r',encoding='utf-8') as f:
            self.results = json.load(f)
        
        self.output_dir = Path(results_path).parent / "figures"
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_overall_metrics(self):
        """Bar chart of key metrics"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scores = self.results["overall_scores"]
        metrics = {
            "Routing\nAccuracy": scores["routing_accuracy"],
            "Language\nDetection": scores["language_detection"],
            "Query\nSuccess": scores["success_rate"]
        }
        
        colors = ['#2ecc71', '#3498db', '#9b59b6']
        bars = ax.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('System Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "overall_metrics.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.output_dir / 'overall_metrics.png'}")
        plt.close()
    
    def plot_confusion_matrix(self):
        """Confusion matrix for routing"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        matrix = self.results["routing_accuracy"]["confusion_matrix"]
        labels = ["SQL", "Document", "Hybrid"]
        
        # Convert to numpy array
        data = np.array([
            [matrix["sql"]["sql"], matrix["sql"]["document"], matrix["sql"]["hybrid"]],
            [matrix["document"]["sql"], matrix["document"]["document"], matrix["document"]["hybrid"]],
            [matrix["hybrid"]["sql"], matrix["hybrid"]["document"], matrix["hybrid"]["hybrid"]]
        ])
        
        # Create heatmap
        sns.heatmap(data, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'}, ax=ax)
        
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title('Routing Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.output_dir / 'confusion_matrix.png'}")
        plt.close()
    
    def plot_language_coverage(self):
        """Show multilingual support"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ml_results = self.results["multilingual_support"]
        languages = ml_results["language_list"]
        
        # Count correct detections per language
        lang_accuracy = {}
        for result in ml_results["detailed_results"]:
            lang = result["expected_language"]
            if lang not in lang_accuracy:
                lang_accuracy[lang] = {"correct": 0, "total": 0}
            lang_accuracy[lang]["total"] += 1
            if result["correct"]:
                lang_accuracy[lang]["correct"] += 1
        
        # Calculate percentages
        lang_names = list(lang_accuracy.keys())
        accuracies = [(lang_accuracy[l]["correct"] / lang_accuracy[l]["total"] * 100) 
                     for l in lang_names]
        
        # Plot
        colors = plt.cm.viridis(np.linspace(0, 1, len(lang_names)))
        bars = ax.barh(lang_names, accuracies, color=colors, alpha=0.8)
        
        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{acc:.0f}%',
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Detection Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Multilingual Support - {len(languages)} Languages', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xlim(0, 110)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "language_coverage.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.output_dir / 'language_coverage.png'}")
        plt.close()
    
    def plot_response_times(self):
        """Response time distribution"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        perf = self.results["performance_metrics"]
        times = [r["response_time"] for r in perf["detailed_results"] 
                if "response_time" in r]
        
        if times:
            ax.hist(times, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(times), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(times):.2f}s')
            
            ax.set_xlabel('Response Time (seconds)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
            ax.set_title('Response Time Distribution', fontsize=14, fontweight='bold', pad=15)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "response_times.png", dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir / 'response_times.png'}")
            plt.close()
    
    def create_summary_dashboard(self):
        """Create a single dashboard with all metrics"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Overall metrics (top left)
        ax1 = fig.add_subplot(gs[0, :2])
        scores = self.results["overall_scores"]
        metrics = {
            "Routing": scores["routing_accuracy"],
            "Language": scores["language_detection"],
            "Success": scores["success_rate"]
        }
        colors = ['#2ecc71', '#3498db', '#9b59b6']
        bars = ax1.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.8)
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Performance Metrics', fontweight='bold')
        ax1.set_ylim(0, 110)
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. Key stats (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        stats_text = f"""
        KEY STATISTICS
        
        Languages: {scores['languages_supported']}
        
        Avg Response: {scores['avg_response_time']:.2f}s
        
        Total Tests: {self.results['routing_accuracy']['total'] + 
                      self.results['multilingual_support']['total']}
        
        Success Rate: {scores['success_rate']:.1f}%
        """
        ax2.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 3. Confusion matrix (middle left)
        ax3 = fig.add_subplot(gs[1:, :2])
        matrix = self.results["routing_accuracy"]["confusion_matrix"]
        data = np.array([
            [matrix["sql"]["sql"], matrix["sql"]["document"], matrix["sql"]["hybrid"]],
            [matrix["document"]["sql"], matrix["document"]["document"], matrix["document"]["hybrid"]],
            [matrix["hybrid"]["sql"], matrix["hybrid"]["document"], matrix["hybrid"]["hybrid"]]
        ])
        sns.heatmap(data, annot=True, fmt='d', cmap='Blues',
                   xticklabels=["SQL", "Document", "Hybrid"],
                   yticklabels=["SQL", "Document", "Hybrid"],
                   ax=ax3, cbar_kws={'label': 'Count'})
        ax3.set_title('Routing Confusion Matrix', fontweight='bold')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # 4. Language support (middle right)
        ax4 = fig.add_subplot(gs[1:, 2])
        ml_results = self.results["multilingual_support"]
        lang_count = ml_results["languages_tested"]
        accuracy = ml_results["accuracy"]
        
        # Pie chart
        sizes = [accuracy, 100-accuracy]
        colors_pie = ['#2ecc71', '#ecf0f1']
        explode = (0.1, 0)
        ax4.pie(sizes, explode=explode, labels=['Correct', 'Incorrect'],
               colors=colors_pie, autopct='%1.1f%%', startangle=90,
               textprops={'fontsize': 10, 'fontweight': 'bold'})
        ax4.set_title(f'Language Detection\n{lang_count} Languages Tested', fontweight='bold')
        
        # Main title
        fig.suptitle('Multilingual Hybrid NLP System - Evaluation Dashboard', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig(self.output_dir / "dashboard.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.output_dir / 'dashboard.png'}")
        plt.close()
    
    def generate_all_plots(self):
        """Generate all visualizations"""
        print("\n📊 Generating Visualizations...\n")
        
        self.plot_overall_metrics()
        self.plot_confusion_matrix()
        self.plot_language_coverage()
        self.plot_response_times()
        self.create_summary_dashboard()
        
        print(f"\n✅ All visualizations saved to: {self.output_dir}/\n")
        print("Generated files:")
        print("  • overall_metrics.png - Key performance metrics")
        print("  • confusion_matrix.png - Routing accuracy breakdown")
        print("  • language_coverage.png - Multilingual support")
        print("  • response_times.png - Performance distribution")
        print("  • dashboard.png - Complete overview (use this in presentation!)")


def main():
    """Main function"""
    import sys
    
    results_path = "./evaluation/results.json"
    
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    
    if not Path(results_path).exists():
        print(f"❌ Results file not found: {results_path}")
        print("Run evaluation first: python run_evaluation.py")
        return
    
    visualizer = ResultsVisualizer(results_path)
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()