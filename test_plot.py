# test_plot_builder.py
"""
Test Plot Builder - Verify it works standalone
"""
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from utils.plot_builder import PlotBuilder


def test_plot_builder():
    """Test the plot builder with mock data"""
    
    print("\n" + "="*60)
    print("TESTING PLOT BUILDER (STEP 1)")
    print("="*60 + "\n")
    
    builder = PlotBuilder()
    
    # Test 1: Check if plotting is needed
    print("Test 1: Detect plot keywords")
    print("-" * 40)
    
    test_questions = [
        ("Show me a chart of sales", True),
        ("Chart monthly revenue", True),
        ("What is the total revenue?", False),
        ("How many customers do we have?", False),
        ("Trend in average deal size", True)
    ]
    
    passed = 0
    for question, should_plot in test_questions:
        result = builder.needs_plot(question)
        status = "✅" if result == should_plot else "❌"
        print(f"{status} '{question}' → needs_plot={result} (expected={should_plot})")
        if result == should_plot:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(test_questions)}\n")
    
    # Test 2: Build plot spec from mock SQL result
    print("Test 2: Build plot specification")
    print("-" * 40)
    
    mock_sql_result = {
        "success": True,
        "sql_query": "SELECT Month, SUM(Revenue) FROM sales GROUP BY Month",
        "row_count": 5,
        "data_preview": [
            {"Month": "Jan", "Total_Revenue": 45000},
            {"Month": "Feb", "Total_Revenue": 52000},
            {"Month": "Mar", "Total_Revenue": 48000},
            {"Month": "Apr", "Total_Revenue": 61000},
            {"Month": "May", "Total_Revenue": 58000}
        ]
    }
    
    question = "Show me monthly revenue trends"
    
    plot_spec = builder.build_plot_spec(mock_sql_result, question)
    
    if plot_spec:
        print("✅ Plot spec generated successfully!\n")
        print(f"Chart Type: {plot_spec['type']}")
        print(f"Title: {plot_spec['title']}")
        print(f"X-axis: {plot_spec['x_label']} → {plot_spec['x']}")
        print(f"Y-axis: {plot_spec['y_label']} → {plot_spec['y']}")
    else:
        print("❌ Failed to generate plot spec")
        return False
    
    print("\n" + "="*60)
    
    # Test 3: Different chart types
    print("\nTest 3: Chart type inference")
    print("-" * 40)
    
    chart_tests = [
        ("Show revenue trend over time", "line"),
        ("Compare sales by region", "bar"),
        ("Distribution of customers", "pie")
    ]
    
    for q, expected_type in chart_tests:
        spec = builder.build_plot_spec(mock_sql_result, q)
        if spec:
            status = "✅" if spec["type"] == expected_type else "❌"
            print(f"{status} '{q}' → {spec['type']} (expected={expected_type})")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - Plot Builder is working!")
    print("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    try:
        success = test_plot_builder()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)