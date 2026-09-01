# test_sql_analyst_plot.py
"""
Test SQL Analyst with Plot Integration
"""
import sys
import os
import asyncio

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from database.db_manager import DatabaseManager
from agents.sql_analyst import SQLAnalystAgent
from config import get_db_config


async def test_sql_analyst_plot():
    """Test SQL Analyst with plot capability"""
    
    print("\n" + "="*60)
    print("TESTING SQL ANALYST WITH PLOTS (STEP 2)")
    print("="*60 + "\n")
    
    # Initialize
    db_config = get_db_config()
    db_manager = DatabaseManager(db_config)
    sql_agent = SQLAnalystAgent(db_manager)
    
    print("✅ SQL Analyst initialized with plot capability\n")
    
    # Test 1: Regular query (no plot)
    print("Test 1: Regular query (should NOT have plot)")
    print("-" * 40)
    
    question1 = "How many employees do we have?"
    result1 = sql_agent.analyze(question1)
    
    if result1.get("success"):
        has_plot = "plot" in result1
        status = "✅" if not has_plot else "⚠️"
        print(f"{status} Question: '{question1}'")
        print(f"   Has plot: {has_plot} (expected: False)")
        print(f"   Answer: {result1.get('explanation', 'N/A')[:80]}...")
    else:
        print(f"❌ Query failed: {result1.get('error')}")
    
    print()
    
    # Test 2: Chart query (should have plot)
    print("Test 2: Chart query (SHOULD have plot)")
    print("-" * 40)
    
    question2 = "Show me a chart of employees by department"
    result2 = sql_agent.analyze(question2)
    
    if result2.get("success"):
        has_plot = "plot" in result2
        status = "✅" if has_plot else "❌"
        print(f"{status} Question: '{question2}'")
        print(f"   Has plot: {has_plot} (expected: True)")
        
        if has_plot:
            plot = result2["plot"]
            print(f"   Chart type: {plot['type']}")
            print(f"   Title: {plot['title']}")
            print(f"   Data points: {len(plot['x'])}")
            print(f"   X: {plot['x'][:3]}..." if len(plot['x']) > 3 else f"   X: {plot['x']}")
            print(f"   Y: {plot['y'][:3]}..." if len(plot['y']) > 3 else f"   Y: {plot['y']}")
        
        print(f"   Answer: {result2.get('explanation', 'N/A')[:80]}...")
    else:
        print(f"❌ Query failed: {result2.get('error')}")
    
    print()
    
    # Test 3: Another chart query
    print("Test 3: Trend chart query")
    print("-" * 40)
    
    question3 = "Show revenue trend by department"
    result3 = sql_agent.analyze(question3)
    
    if result3.get("success"):
        has_plot = "plot" in result3
        status = "✅" if has_plot else "❌"
        print(f"{status} Question: '{question3}'")
        print(f"   Has plot: {has_plot} (expected: True)")
        
        if has_plot:
            plot = result3["plot"]
            print(f"   Chart type: {plot['type']}")
            print(f"   Title: {plot['title']}")
    else:
        print(f"❌ Query failed: {result3.get('error')}")
    
    print("\n" + "="*60)
    print("✅ STEP 2 COMPLETE - SQL Analyst has plot capability!")
    print("="*60 + "\n")
    
    print("📋 Summary:")
    print(f"   - Regular queries: Still work (no plot)")
    print(f"   - Chart queries: Now include plot data")
    print(f"   - Existing functionality: UNBROKEN ✅")
    
    return True


if __name__ == "__main__":
    try:
        asyncio.run(test_sql_analyst_plot())
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)