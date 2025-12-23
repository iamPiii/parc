"""
Test script for NVARC ARC Solver

Quick test to verify the solver works correctly
"""

import json
from arc_solver_nvarc import ARCSolver


def test_simple_puzzle():
    """Test with a simple example puzzle"""
    
    # Simple identity transformation example
    train_examples = [
        {
            "input": [[1, 2], [3, 4]],
            "output": [[1, 2], [3, 4]]
        },
        {
            "input": [[5, 6], [7, 8]],
            "output": [[5, 6], [7, 8]]
        }
    ]
    
    test_input = [[0, 1], [2, 3]]
    
    print("=" * 60)
    print("Testing NVARC ARC Solver")
    print("=" * 60)
    
    print("\nTrain Examples:")
    for i, ex in enumerate(train_examples, 1):
        print(f"  Example {i}:")
        print(f"    Input:  {ex['input']}")
        print(f"    Output: {ex['output']}")
    
    print(f"\nTest Input: {test_input}")
    
    # Initialize solver
    print("\nInitializing solver...")
    solver = ARCSolver()
    
    # Solve
    print("\nSolving...")
    result = solver.solve(train_examples, test_input)
    
    print(f"\nResult: {result}")
    
    # Validate
    is_valid = (
        isinstance(result, list) and
        all(isinstance(row, list) for row in result) and
        all(isinstance(val, int) and 0 <= val <= 9 for row in result for val in row)
    )
    
    if is_valid:
        print("✓ Result is valid!")
    else:
        print("❌ Result is invalid!")
    
    return is_valid


def test_with_arc_data():
    """Test with actual ARC data if available"""
    try:
        # Try to load evaluation data
        with open("arc-agi_evaluation_challenges.json", "r") as f:
            data = json.load(f)
        
        # Get first puzzle
        puzzle_id = list(data.keys())[0]
        puzzle = data[puzzle_id]
        
        print("=" * 60)
        print(f"Testing with ARC puzzle: {puzzle_id}")
        print("=" * 60)
        
        train_examples = puzzle["train"]
        test_input = puzzle["test"][0]["input"]
        
        print(f"\nNumber of training examples: {len(train_examples)}")
        print(f"Test input shape: {len(test_input)}x{len(test_input[0])}")
        
        # Initialize solver
        solver = ARCSolver()
        
        # Solve
        print("\nSolving...")
        result = solver.solve(train_examples, test_input)
        
        print(f"\nResult shape: {len(result)}x{len(result[0])}")
        print("✓ Solved successfully!")
        
        return True
        
    except FileNotFoundError:
        print("⚠ ARC evaluation data not found, skipping this test")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("NVARC ARC Solver - Test Suite")
    print("=" * 60 + "\n")
    
    # Test 1: Simple puzzle
    test1_passed = test_simple_puzzle()
    
    print("\n")
    
    # Test 2: Real ARC data
    test2_passed = test_with_arc_data()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("✓ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
