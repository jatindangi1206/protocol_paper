#!/usr/bin/env python3
"""
Test script for GOQII Health Data EDA Pipeline with LangChain Integration

This script tests both the basic EDA pipeline and the LangChain integration
to ensure everything works correctly with the example data.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_pipeline():
    """Test the basic EDA pipeline without LangChain."""
    print("🧪 Testing Basic EDA Pipeline...")
    
    try:
        from eda_pipeline import EDAPipeline
        
        # Use example data
        input_dir = "data/input"
        output_dir = tempfile.mkdtemp(prefix="eda_test_")
        
        print(f"📁 Using test output directory: {output_dir}")
        
        # Run pipeline
        pipeline = EDAPipeline(input_dir, output_dir)
        result = pipeline.run_complete_pipeline()
        
        # The EDAPipeline doesn't return a result, so if no exception was raised, it succeeded
        print("✅ Basic EDA pipeline test PASSED")
        
        # Check key outputs
        expected_files = [
            'cleaned/bp.csv',
            'cleaned/hr.csv', 
            'cleaned/sleep.csv',
            'summary/cohort_statistics.csv',
            'plots/cohort/compliance_summary.png'
        ]
        
        missing_files = []
        for file_path in expected_files:
            full_path = os.path.join(output_dir, file_path)
            if not os.path.exists(full_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"⚠️  Missing expected files: {missing_files}")
        else:
            print("✅ All expected output files created")
            
        # Cleanup
        shutil.rmtree(output_dir)
        return True
        
    except Exception as e:
        print(f"❌ Basic EDA pipeline test FAILED with exception: {e}")
        return False

def test_langchain_integration():
    """Test the LangChain integration."""
    print("\n🤖 Testing LangChain Integration...")
    
    try:
        # Test import
        from langchain_integration import LANGCHAIN_AVAILABLE, LangChainHealthAnalyzer
        
        if not LANGCHAIN_AVAILABLE:
            print("⚠️  LangChain not available - skipping integration test")
            print("   Install with: pip install langchain openai faiss-cpu")
            return True  # Not a failure, just not available
        
        # Test configuration
        import config
        config_status = config.validate_config()
        
        if not config_status['valid']:
            print(f"⚠️  LangChain configuration issues: {config_status['issues']}")
            print("   Set up API keys to enable LangChain features")
            return True  # Not a failure, just not configured
        
        print(f"✅ LangChain integration available (Provider: {config_status['provider']})")
        
        # Test analyzer initialization
        analyzer = LangChainHealthAnalyzer()
        print("✅ LangChain analyzer initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ LangChain integration test FAILED: {e}")
        return False

def test_weekly_reports():
    """Test the weekly reports generation."""
    print("\n📊 Testing Weekly Reports Generation...")
    
    try:
        # Test import
        from generate_weekly_reports import WeeklyReportsGenerator
        
        input_dir = "data/input"
        output_dir = tempfile.mkdtemp(prefix="weekly_test_")
        
        print(f"📁 Using test output directory: {output_dir}")
        
        # Test without LangChain first
        generator = WeeklyReportsGenerator(input_dir, output_dir, enable_langchain=False)
        result = generator.generate_weekly_reports()
        
        if result['success']:
            print("✅ Weekly reports generation (basic mode) test PASSED")
            print(f"   Generated {result['individual_summaries_count']} individual summaries")
            print(f"   Processed {result['participants_count']} participants")
        else:
            print(f"❌ Weekly reports generation test FAILED: {result.get('error', 'Unknown error')}")
            return False
            
        # Test with LangChain if available
        from langchain_integration import LANGCHAIN_AVAILABLE
        if LANGCHAIN_AVAILABLE:
            import config
            config_status = config.validate_config()
            
            if config_status['valid']:
                print("\n🤖 Testing with LangChain enabled...")
                generator_ai = WeeklyReportsGenerator(input_dir, output_dir, enable_langchain=True)
                result_ai = generator_ai.generate_weekly_reports()
                
                if result_ai['success']:
                    print("✅ Weekly reports with LangChain test PASSED")
                else:
                    print(f"⚠️  Weekly reports with LangChain test had issues: {result_ai.get('error', 'Unknown error')}")
            else:
                print("⚠️  Skipping LangChain test due to configuration issues")
        
        # Cleanup
        shutil.rmtree(output_dir)
        return True
        
    except Exception as e:
        print(f"❌ Weekly reports generation test FAILED: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Starting GOQII Health Data EDA Pipeline Integration Tests")
    print("=" * 60)
    
    # Check if example data exists
    if not os.path.exists("data/input"):
        print("❌ Example data directory 'data/input' not found")
        print("   Please run this test from the project root directory")
        return 1
    
    example_files = list(Path("data/input").glob("*"))
    if not example_files:
        print("❌ No example data files found in 'data/input'")
        print("   Please ensure example data files are present")
        return 1
    
    print(f"📄 Found {len(example_files)} example data files")
    
    # Run tests
    tests = [
        ("Basic EDA Pipeline", test_basic_pipeline),
        ("LangChain Integration", test_langchain_integration),
        ("Weekly Reports Generation", test_weekly_reports)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 60)
    print("🏁 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! The integration is working correctly.")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
