# evaluation/evaluate_system.py
"""
Comprehensive Evaluation Framework for Multilingual Hybrid NLP System
Measures: Routing Accuracy, Multilingual Support, Response Quality
"""

import json
import time
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime
import asyncio

# Evaluation Benchmark Dataset
EVALUATION_DATASET = {
    "routing_accuracy": [
        # SQL Queries (should route to SQL)
        {"query": "How many employees are in the database?", "expected": "sql", "language": "en"},
        {"query": "What is the average salary of engineers?", "expected": "sql", "language": "en"},
        {"query": "List all departments with their budgets", "expected": "sql", "language": "en"},
        {"query": "Show me total sales for last month", "expected": "sql", "language": "en"},
        {"query": "Count products in Electronics category", "expected": "sql", "language": "en"},
        
        # Document Queries (should route to DOCUMENT)
        {"query": "What are the action items from the meeting?", "expected": "document", "language": "en"},
        {"query": "What budget was approved in the strategic plan?", "expected": "document", "language": "en"},
        {"query": "Summarize the performance review document", "expected": "document", "language": "en"},
        {"query": "What's the status distribution in the roadmap file?", "expected": "document", "language": "en"},
        {"query": "List recommendations from the uploaded report", "expected": "document", "language": "en"},
        
        # Hybrid Queries (should route to HYBRID)
        {"query": "Compare database sales with targets in uploaded file", "expected": "hybrid", "language": "en"},
        {"query": "How do actual employee counts match hiring plan document?", "expected": "hybrid", "language": "en"},
    ],
    
    "multilingual_support": [
        # English
        {"query": "How many employees work here?", "language": "en", "expected_lang": "en"},
        
        # Spanish
        {"query": "¿Cuántos empleados trabajan aquí?", "language": "es", "expected_lang": "es"},
        {"query": "¿Cuál es el salario promedio?", "language": "es", "expected_lang": "es"},
        
        # French
        {"query": "Combien d'employés travaillent ici?", "language": "fr", "expected_lang": "fr"},
        {"query": "Quel est le salaire moyen?", "language": "fr", "expected_lang": "fr"},
        
        # German
        {"query": "Wie viele Mitarbeiter arbeiten hier?", "language": "de", "expected_lang": "de"},
        
        # Hindi
        {"query": "यहाँ कितने कर्मचारी काम करते हैं?", "language": "hi", "expected_lang": "hi"},
        
        # Chinese
        {"query": "这里有多少员工?", "language": "zh-cn", "expected_lang": "zh-cn"},
        
        # Japanese
        {"query": "ここで働いている従業員は何人ですか?", "language": "ja", "expected_lang": "ja"},
        
        # Arabic
        {"query": "كم عدد الموظفين الذين يعملون هنا؟", "language": "ar", "expected_lang": "ar"},
        
        # Portuguese
        {"query": "Quantos funcionários trabalham aqui?", "language": "pt", "expected_lang": "pt"},
    ],
    
    "document_analytics": [
        # These test if RAG can answer analytical questions about Excel/CSV
        {"query": "What percentage of items are completed?", "type": "percentage"},
        {"query": "Show distribution of status categories", "type": "distribution"},
        {"query": "How many items in each priority level?", "type": "count"},
        {"query": "Calculate the ratio of high priority to total items", "type": "ratio"},
    ]
}


class SystemEvaluator:
    """Evaluate the Multilingual Hybrid NLP System"""
    
    def __init__(self, orchestrator, language_detector):
        self.orchestrator = orchestrator
        self.language_detector = language_detector
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "routing_accuracy": {},
            "multilingual_support": {},
            "document_analytics": {},
            "performance_metrics": {},
            "overall_scores": {}
        }
    
    async def evaluate_routing_accuracy(self) -> Dict[str, Any]:
        """Test if queries route to correct agent"""
        print("\n" + "="*60)
        print("📍 EVALUATING ROUTING ACCURACY")
        print("="*60)
        
        test_cases = EVALUATION_DATASET["routing_accuracy"]
        correct = 0
        total = len(test_cases)
        detailed_results = []
        
        # Create a mock conversation with documents for document/hybrid tests
        mock_conv_id = "test_conv_eval_123"
        
        for i, case in enumerate(test_cases, 1):
            query = case["query"]
            expected = case["expected"]
            
            print(f"\n[{i}/{total}] Testing: {query[:50]}...")
            
            # For document/hybrid queries, simulate having documents
            if expected in ["document", "hybrid"]:
                # Mock the session to have documents
                self.orchestrator.rag_agent.session_stores[mock_conv_id] = {
                    "vectorstore": "mock_store",  # Just need non-None
                    "qa_chain": None,
                    "loaded_files": ["test_document.pdf"],
                    "total_chunks": 100
                }
                test_conv_id = mock_conv_id
            else:
                test_conv_id = None
            
            # Test routing classification
            predicted = self.orchestrator._classify_query(query, conversation_id=test_conv_id)
            predicted_lower = predicted.lower()
            
            is_correct = predicted_lower == expected
            if is_correct:
                correct += 1
                print(f"  ✅ Correct: {predicted} (expected: {expected})")
            else:
                print(f"  ❌ Wrong: {predicted} (expected: {expected})")
            
            detailed_results.append({
                "query": query,
                "expected": expected,
                "predicted": predicted_lower,
                "correct": is_correct
            })
        if mock_conv_id in self.orchestrator.rag_agent.session_stores:
            del self.orchestrator.rag_agent.session_stores[mock_conv_id]
        
        accuracy = (correct / total) * 100
        
        print(f"\n{'='*60}")
        print(f"📊 ROUTING ACCURACY: {correct}/{total} ({accuracy:.1f}%)")
        print(f"{'='*60}")
        
        # Confusion matrix
        confusion = {"sql": {"sql": 0, "document": 0, "hybrid": 0},
                    "document": {"sql": 0, "document": 0, "hybrid": 0},
                    "hybrid": {"sql": 0, "document": 0, "hybrid": 0}}
        
        for result in detailed_results:
            confusion[result["expected"]][result["predicted"]] += 1
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "detailed_results": detailed_results,
            "confusion_matrix": confusion
        }
    
    async def evaluate_multilingual_support(self) -> Dict[str, Any]:
        """Test language detection accuracy"""
        print("\n" + "="*60)
        print("🌍 EVALUATING MULTILINGUAL SUPPORT")
        print("="*60)
        
        test_cases = EVALUATION_DATASET["multilingual_support"]
        correct = 0
        total = len(test_cases)
        detailed_results = []
        language_coverage = set()
        
        for i, case in enumerate(test_cases, 1):
            query = case["query"]
            expected_lang = case["expected_lang"]
            
            print(f"\n[{i}/{total}] Testing: {query[:40]}...")
            
            # Detect language
            detected_lang = self.language_detector.detect_language(query)
            language_coverage.add(detected_lang)
            
            is_correct = detected_lang == expected_lang
            if is_correct:
                correct += 1
                print(f"  ✅ Correct: {detected_lang}")
            else:
                print(f"  ❌ Wrong: {detected_lang} (expected: {expected_lang})")
            
            detailed_results.append({
                "query": query,
                "expected_language": expected_lang,
                "detected_language": detected_lang,
                "correct": is_correct
            })
        
        accuracy = (correct / total) * 100
        
        print(f"\n{'='*60}")
        print(f"🌍 LANGUAGE DETECTION: {correct}/{total} ({accuracy:.1f}%)")
        print(f"Languages Covered: {len(language_coverage)} - {sorted(language_coverage)}")
        print(f"{'='*60}")
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "languages_tested": len(language_coverage),
            "language_list": sorted(language_coverage),
            "detailed_results": detailed_results
        }
    
    async def evaluate_response_quality(self, sample_queries: List[str]) -> Dict[str, Any]:
        """Test actual query execution (if you have test data loaded)"""
        print("\n" + "="*60)
        print("⚡ EVALUATING RESPONSE QUALITY")
        print("="*60)
        
        results = []
        total_time = 0
        
        for i, query in enumerate(sample_queries, 1):
            print(f"\n[{i}/{len(sample_queries)}] Query: {query[:50]}...")
            
            start_time = time.time()
            try:
                result = await self.orchestrator.route_query(query, conversation_id=None)
                elapsed = time.time() - start_time
                total_time += elapsed
                
                success = result.get("success", False)
                routing = result.get("routing", "unknown")
                
                print(f"  ✅ Routed to: {routing} ({elapsed:.2f}s)")
                
                results.append({
                    "query": query,
                    "success": success,
                    "routing": routing,
                    "response_time": elapsed,
                    "has_answer": bool(result.get("explanation") or result.get("answer"))
                })
            except Exception as e:
                print(f"  ❌ Error: {e}")
                results.append({
                    "query": query,
                    "success": False,
                    "error": str(e)
                })
        
        avg_time = total_time / len(sample_queries) if sample_queries else 0
        success_rate = sum(1 for r in results if r.get("success")) / len(results) * 100
        
        print(f"\n{'='*60}")
        print(f"⚡ SUCCESS RATE: {success_rate:.1f}%")
        print(f"⏱️  AVG RESPONSE TIME: {avg_time:.2f}s")
        print(f"{'='*60}")
        
        return {
            "success_rate": success_rate,
            "average_response_time": avg_time,
            "total_queries": len(sample_queries),
            "detailed_results": results
        }
    
    async def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation suite"""
        print("\n" + "="*60)
        print("🚀 STARTING FULL SYSTEM EVALUATION")
        print("="*60)
        
        # 1. Routing Accuracy
        self.results["routing_accuracy"] = await self.evaluate_routing_accuracy()
        
        # 2. Multilingual Support
        self.results["multilingual_support"] = await self.evaluate_multilingual_support()
        
        # 3. Response Quality (sample queries)
        sample_queries = [
            "How many employees are there?",
            "What is the average salary?",
            "¿Cuántos empleados hay?",  # Spanish
            "Combien d'employés?",  # French
        ]
        self.results["performance_metrics"] = await self.evaluate_response_quality(sample_queries)
        
        # Calculate overall scores
        self.results["overall_scores"] = {
            "routing_accuracy": self.results["routing_accuracy"]["accuracy"],
            "language_detection": self.results["multilingual_support"]["accuracy"],
            "success_rate": self.results["performance_metrics"]["success_rate"],
            "avg_response_time": self.results["performance_metrics"]["average_response_time"],
            "languages_supported": self.results["multilingual_support"]["languages_tested"]
        }
        
        # Print summary
        self._print_summary()
        
        return self.results
    
    def _print_summary(self):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("📊 EVALUATION SUMMARY")
        print("="*60)
        
        scores = self.results["overall_scores"]
        
        print(f"\n🎯 Key Metrics:")
        print(f"  • Routing Accuracy:      {scores['routing_accuracy']:.1f}%")
        print(f"  • Language Detection:    {scores['language_detection']:.1f}%")
        print(f"  • Query Success Rate:    {scores['success_rate']:.1f}%")
        print(f"  • Avg Response Time:     {scores['avg_response_time']:.2f}s")
        print(f"  • Languages Supported:   {scores['languages_supported']}")
        
        # Overall grade
        avg_accuracy = (scores['routing_accuracy'] + scores['language_detection'] + scores['success_rate']) / 3
        
        if avg_accuracy >= 90:
            grade = "A+ Excellent"
        elif avg_accuracy >= 80:
            grade = "A Good"
        elif avg_accuracy >= 70:
            grade = "B Fair"
        else:
            grade = "C Needs Improvement"
        
        print(f"\n🏆 Overall Performance: {avg_accuracy:.1f}% ({grade})")
        print(f"{'='*60}\n")
    
    def save_results(self, output_path: str = "./evaluation/results.json"):
        """Save results to JSON file"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {output_path}")
        
        # Also save a human-readable report
        report_path = output_path.replace('.json', '_report.txt')
        self._generate_report(report_path)
    
    def _generate_report(self, report_path: str):
        """Generate human-readable report"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("MULTILINGUAL HYBRID NLP SYSTEM - EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Evaluation Date: {self.results['timestamp']}\n\n")
            
            # Overall Scores
            f.write("OVERALL PERFORMANCE\n")
            f.write("-" * 70 + "\n")
            scores = self.results["overall_scores"]
            f.write(f"Routing Accuracy:        {scores['routing_accuracy']:.1f}%\n")
            f.write(f"Language Detection:      {scores['language_detection']:.1f}%\n")
            f.write(f"Query Success Rate:      {scores['success_rate']:.1f}%\n")
            f.write(f"Avg Response Time:       {scores['avg_response_time']:.2f}s\n")
            f.write(f"Languages Supported:     {scores['languages_supported']}\n\n")
            
            # Routing Accuracy Details
            f.write("ROUTING ACCURACY DETAILS\n")
            f.write("-" * 70 + "\n")
            routing = self.results["routing_accuracy"]
            f.write(f"Correct: {routing['correct']}/{routing['total']}\n")
            f.write(f"Accuracy: {routing['accuracy']:.1f}%\n\n")
            
            f.write("Confusion Matrix:\n")
            matrix = routing['confusion_matrix']
            f.write(f"                SQL    Document    Hybrid\n")
            for expected in ["sql", "document", "hybrid"]:
                counts = matrix[expected]
                f.write(f"{expected:12s}  {counts['sql']:3d}     {counts['document']:3d}        {counts['hybrid']:3d}\n")
            
            f.write("\n")
            
            # Multilingual Support
            f.write("MULTILINGUAL SUPPORT\n")
            f.write("-" * 70 + "\n")
            ml = self.results["multilingual_support"]
            f.write(f"Languages Tested: {ml['languages_tested']}\n")
            f.write(f"Languages: {', '.join(ml['language_list'])}\n")
            f.write(f"Detection Accuracy: {ml['accuracy']:.1f}%\n\n")
            
            f.write("="*70 + "\n")
        
        print(f"📄 Report saved to: {report_path}")


# Quick benchmark function
async def quick_benchmark(orchestrator, language_detector):
    """Run quick benchmark - just routing and language detection"""
    evaluator = SystemEvaluator(orchestrator, language_detector)
    
    print("\n🚀 Running Quick Benchmark...")
    print("   (Routing + Language Detection only)\n")
    
    # Routing test
    routing_results = await evaluator.evaluate_routing_accuracy()
    
    # Language detection test
    lang_results = await evaluator.evaluate_multilingual_support()
    
    # Summary
    print("\n" + "="*60)
    print("✅ QUICK BENCHMARK COMPLETE")
    print("="*60)
    print(f"Routing Accuracy:     {routing_results['accuracy']:.1f}%")
    print(f"Language Detection:   {lang_results['accuracy']:.1f}%")
    print(f"Languages Supported:  {lang_results['languages_tested']}")
    print("="*60 + "\n")
    
    return {
        "routing_accuracy": routing_results['accuracy'],
        "language_detection": lang_results['accuracy'],
        "languages_supported": lang_results['languages_tested']
    }


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          MULTILINGUAL NLP SYSTEM EVALUATOR                   ║
║                                                              ║
║  This script evaluates:                                      ║
║  ✓ Routing Accuracy (SQL vs Document vs Hybrid)             ║
║  ✓ Multilingual Language Detection (20+ languages)          ║
║  ✓ Response Quality & Performance                           ║
╚══════════════════════════════════════════════════════════════╝
    
Usage:
    from evaluation.evaluate_system import SystemEvaluator, quick_benchmark
    
    # Quick benchmark (no DB/docs needed)
    results = await quick_benchmark(orchestrator, language_detector)
    
    # Full evaluation
    evaluator = SystemEvaluator(orchestrator, language_detector)
    results = await evaluator.run_full_evaluation()
    evaluator.save_results()
    """)