# tests/test_visual.py
"""
Generate visualizations for BM25+FAISS+Reranking evaluation results
Creates charts showing hybrid retrieval performance
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


class ResultsVisualizer:
    """Create visualizations for production RAG evaluation"""
    
    def __init__(self, results_path: str = "./evaluation/resullts.json"):
        with open(results_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)
        
        self.output_dir = Path(results_path).parent / "figurres"
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
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title('System Performance Metrics\nBM25+FAISS+Reranking Architecture', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "overall_metrics.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.output_dir / 'overall_metrics.png'}")
        plt.close()
    
    def plot_confusion_matrix(self):
        """Confusion matrix for routing (includes HYBRID)"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        matrix = self.results["routing_accuracy"]["confusion_matrix"]
        labels = ["SQL", "Document", "Hybrid"]
        
        data = np.array([
            [matrix["sql"]["sql"], matrix["sql"]["document"], matrix["sql"]["hybrid"]],
            [matrix["document"]["sql"], matrix["document"]["document"], matrix["document"]["hybrid"]],
            [matrix["hybrid"]["sql"], matrix["hybrid"]["document"], matrix["hybrid"]["hybrid"]]
        ])
        
        sns.heatmap(data, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'}, ax=ax)
        
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title('Routing Confusion Matrix\n(SQL/Document/Hybrid)', 
                    fontsize=14, fontweight='bold', pad=15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "confusion_matrix.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.output_dir / 'confusion_matrix.png'}")
        plt.close()
    
    def plot_language_coverage(self):
        """Show multilingual support"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ml_results = self.results["multilingual_support"]
        
        lang_accuracy = {}
        for result in ml_results["detailed_results"]:
            lang = result["expected_language"]
            if lang not in lang_accuracy:
                lang_accuracy[lang] = {"correct": 0, "total": 0}
            lang_accuracy[lang]["total"] += 1
            if result["correct"]:
                lang_accuracy[lang]["correct"] += 1
        
        lang_names = list(lang_accuracy.keys())
        accuracies = [(lang_accuracy[l]["correct"] / lang_accuracy[l]["total"] * 100) 
                     for l in lang_names]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(lang_names)))
        bars = ax.barh(lang_names, accuracies, color=colors, alpha=0.8)
        
        for bar, acc in zip(bars, accuracies):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{acc:.0f}%',
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Detection Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Multilingual Support - {len(lang_names)} Languages', 
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
            ax.set_title('Response Time Distribution\nBM25+FAISS+Reranking Pipeline', 
                        fontsize=14, fontweight='bold', pad=15)
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "response_times.png", dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir / 'response_times.png'}")
            plt.close()
    
    def plot_retrieval_architecture(self):
        """NEW: Visualize retrieval pipeline stages"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        retrieval = self.results.get("retrieval_performance", {})
        config = retrieval.get("config", {})
        
        # Create pipeline visualization
        stages = ['Stage 1\nHybrid Search', 'Stage 2\nReranking', 'Stage 3\nGeneration']
        descriptions = [
            f'BM25+FAISS\nretrieve@{config.get("initial_k", 10)}',
            f'Cross-Encoder\nrerank@{config.get("rerank_k", 3)}',
            f'LLM\nAnswer'
        ]
        y_pos = np.arange(len(stages))
        
        colors = ['#3498db', '#2ecc71', '#9b59b6']
        bars = ax.barh(y_pos, [10, 3, 1], color=colors, alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(stages, fontsize=12, fontweight='bold')
        ax.set_xlabel('Number of Documents', fontsize=12, fontweight='bold')
        ax.set_title('Three-Stage Retrieval Pipeline\nBM25+FAISS → Cross-Encoder → LLM', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Add descriptions
        for i, (bar, desc) in enumerate(zip(bars, descriptions)):
            width = bar.get_width()
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                   desc, ha='left', va='center', fontsize=10)
        
        ax.set_xlim(0, 12)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "retrieval_architecture.png", dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {self.output_dir / 'retrieval_architecture.png'}")
        plt.close()
    
    def create_summary_dashboard(self):
        """Complete dashboard with retrieval info"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)
        
        # 1. Overall metrics
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
        
        # 2. Key stats (including retrieval info)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        retrieval_method = scores.get('retrieval_method', 'N/A')
        retrieval_stages = scores.get('retrieval_stages', 'N/A')
        
        stats_text = f"""
        KEY STATISTICS
        
        Languages: {scores['languages_supported']}
        
        Avg Response: {scores['avg_response_time']:.2f}s
        
        Retrieval: {retrieval_stages}-stage
        
        Method: Hybrid
        BM25+FAISS+Rerank
        """
        ax2.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # 3. Confusion matrix
        ax3 = fig.add_subplot(gs[1:3, :2])
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
        
        # 4. Language support pie
        ax4 = fig.add_subplot(gs[1, 2])
        ml_results = self.results["multilingual_support"]
        accuracy = ml_results["accuracy"]
        sizes = [accuracy, 100-accuracy]
        colors_pie = ['#2ecc71', '#ecf0f1']
        ax4.pie(sizes, labels=['Correct', 'Incorrect'],
               colors=colors_pie, autopct='%1.1f%%', startangle=90,
               textprops={'fontweight': 'bold'})
        ax4.set_title(f'Language Detection\n{ml_results["languages_tested"]} Languages', 
                     fontweight='bold')
        
        # 5. NEW: Retrieval pipeline
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.axis('off')
        pipeline_text = """
        RETRIEVAL PIPELINE
        
        Stage 1: Hybrid Search
        • BM25 (lexical)
        • FAISS (semantic)
        • Ensemble fusion
        
        Stage 2: Reranking
        • Cross-Encoder
        • Top 3 from 10
        
        Stage 3: Generation
        • LLM with context
        """
        ax5.text(0.1, 0.5, pipeline_text, fontsize=9, verticalalignment='center',
                fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        # 6. Response times
        ax6 = fig.add_subplot(gs[3, :])
        perf = self.results["performance_metrics"]
        times = [r["response_time"] for r in perf["detailed_results"] if "response_time" in r]
        if times:
            ax6.hist(times, bins=15, color='#3498db', alpha=0.7, edgecolor='black')
            ax6.axvline(np.mean(times), color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {np.mean(times):.2f}s')
            ax6.set_xlabel('Response Time (seconds)')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Response Time Distribution', fontweight='bold')
            ax6.legend()
            ax6.grid(axis='y', alpha=0.3)
        
        fig.suptitle('Production RAG System - Evaluation Dashboard\nBM25 + FAISS + Cross-Encoder Reranking', 
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
        self.plot_retrieval_architecture()  # NEW
        self.create_summary_dashboard()
        
        print(f"\n✅ All visualizations saved to: {self.output_dir}/\n")
        print("Generated files:")
        print("  • overall_metrics.png - Performance metrics")
        print("  • confusion_matrix.png - Routing accuracy (SQL/Doc/Hybrid)")
        print("  • language_coverage.png - Multilingual support")
        print("  • response_times.png - Performance distribution")
        print("  • retrieval_architecture.png - 3-stage pipeline (NEW)")
        print("  • dashboard.png - Complete overview (presentation-ready)")


def main():
    """Main function"""
    import sys
    
    results_path = "./evaluation/results.json"
    
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    
    if not Path(results_path).exists():
        print(f"❌ Results file not found: {results_path}")
        print("Run evaluation first: python tests/test_full_system.py")
        return
    
    visualizer = ResultsVisualizer(results_path)
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main()