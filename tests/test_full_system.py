
import asyncio
from pathlib import Path
import sys, os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from database.db_manager import DatabaseManager
from database.sample_data import create_sample_database
from agents.orchestrator import OrchestratorAgent
from tools.language_detector import LanguageDetector
from tests.test_system import SystemEvaluator, quick_benchmark
from config import settings


async def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║       MULTILINGUAL NLP SYSTEM - EVALUATION RUNNER            ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    # Setup
    print("🔧 Initializing system...")
    
    # Create sample DB if needed
    if not Path(settings.DB_PATH).exists():
        print("📊 Creating sample database...")
        create_sample_database(settings.DB_PATH)
    
    # ✅ FIXED: Initialize without arguments (uses get_db_config())
    db_manager = DatabaseManager()
    orchestrator = OrchestratorAgent(db_manager)
    language_detector = LanguageDetector()
    
    print("✅ System initialized\n")
    
    # Choose evaluation mode
    print("Select evaluation mode:")
    print("  1. Quick Benchmark (2 mins) - Routing + Language Detection")
    print("  2. Full Evaluation (5 mins) - Everything + Performance Tests")
    print("  3. Just Routing Accuracy")
    print("  4. Just Multilingual Support")
    
    choice = input("\nEnter choice (1-4) [default: 1]: ").strip() or "1"
    
    evaluator = SystemEvaluator(orchestrator, language_detector)
    
    if choice == "1":
        # Quick benchmark
        print("\n🚀 Running Quick Benchmark...\n")
        results = await quick_benchmark(orchestrator, language_detector)
        
    elif choice == "2":
        # Full evaluation
        print("\n🚀 Running Full Evaluation...\n")
        results = await evaluator.run_full_evaluation()
        
        # Save results
        save = input("\nSave results? (y/n) [default: y]: ").strip().lower() or "y"
        if save == "y":
            Path("./evaluation").mkdir(exist_ok=True)
            evaluator.save_results("./evaluation/results.json")
            print("\n✅ Results saved to ./evaluation/results.json")
    
    elif choice == "3":
        # Just routing
        print("\n📍 Evaluating Routing Accuracy...\n")
        results = await evaluator.evaluate_routing_accuracy()
        print(f"\n📊 Routing Accuracy: {results['accuracy']:.1f}%")
    
    elif choice == "4":
        # Just multilingual
        print("\n🌍 Evaluating Multilingual Support...\n")
        results = await evaluator.evaluate_multilingual_support()
        print(f"\n🌍 Language Detection: {results['accuracy']:.1f}%")
    
    print("\n✅ Evaluation Complete!\n")


if __name__ == "__main__":
    asyncio.run(main())


    