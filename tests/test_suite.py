# test_suite.py - comprehensive test suite for evaluation
import time
import json
import os
from typing import List, Dict
from src.enhanced_main import AgenticPipeline

class TestSuite:
    def __init__(self):
        self.pipeline = AgenticPipeline()
        self.results = []
        
    def run_tests(self) -> List[Dict]:
        """Run all test queries and collect metrics"""
        
        test_queries = [
            # Knowledge Base Tests
            {
                "query": "What is cloud pricing?",
                "expected_tool": "KB",
                "category": "KB_Relevant",
                "description": "Direct match in knowledge base"
            },
            {
                "query": "How do rate limiting patterns work?",
                "expected_tool": "KB", 
                "category": "KB_Relevant",
                "description": "API rate limiting in KB"
            },
            {
                "query": "What is machine learning?",
                "expected_tool": "KB",
                "category": "KB_Irrelevant", 
                "description": "Not in KB - should detect irrelevance"
            },
            
            # CSV Tool Tests
            {
                "query": "What is the price of coffee?",
                "expected_tool": "TOOL_CSV",
                "category": "CSV_Tool",
                "description": "Product pricing query"
            },
            {
                "query": "Find SKU PEN456",
                "expected_tool": "TOOL_CSV",
                "category": "CSV_Tool", 
                "description": "Direct SKU lookup"
            },
            {
                "query": "Show me available products",
                "expected_tool": "TOOL_CSV",
                "category": "CSV_Tool",
                "description": "Product availability"
            },
            
            # Web Search Tests
            {
                "query": "What is the latest news about AI?",
                "expected_tool": "TOOL_WEB",
                "category": "Web_Tool",
                "description": "Current events query"
            },
            {
                "query": "How to implement REST APIs in Python?",
                "expected_tool": "TOOL_WEB", 
                "category": "Web_Tool",
                "description": "Technical how-to query"
            },
            
            # API Tool Tests
            {
                "query": "Get user information for user ID 1",
                "expected_tool": "TOOL_API",
                "category": "API_Tool",
                "description": "User data request"
            },
            {
                "query": "Show me posts from user 2",
                "expected_tool": "TOOL_API",
                "category": "API_Tool", 
                "description": "Posts API call"
            },
            {
                "query": "What's the weather in New York?",
                "expected_tool": "TOOL_API",
                "category": "API_Tool",
                "description": "Weather API simulation"
            },
            
            # Complex Multi-step Tests
            {
                "query": "Find the price of coffee and then search for coffee brewing tips",
                "expected_tool": "TOOL_CSV",  # Should prioritize CSV first
                "category": "Complex",
                "description": "Multi-step query requiring multiple tools"
            }
        ]
        
        print(f"🧪 Running {len(test_queries)} test queries...")
        print("=" * 60)
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n📝 Test {i}/{len(test_queries)}: {test_case['description']}")
            print(f"Query: {test_case['query']}")
            print(f"Expected: {test_case['expected_tool']}")
            
            start_time = time.time()
            try:
                final_answer, trace = self.pipeline.handle_query(test_case['query'])
                end_time = time.time()
                
                # Extract metrics from trace
                total_latency = sum(step['details'].get('latency_s', 0) for step in trace)
                tool_used = None
                tool_latency = 0
                
                for step in trace:
                    if step['step'] == 'tool_call':
                        tool_used = step['details'].get('tool', 'unknown')
                        tool_latency = step['details'].get('latency_s', 0)
                        break
                
                result = {
                    "test_id": i,
                    "query": test_case['query'],
                    "expected_tool": test_case['expected_tool'],
                    "actual_tool": tool_used,
                    "category": test_case['category'],
                    "description": test_case['description'],
                    "total_latency": total_latency,
                    "tool_latency": tool_latency,
                    "success": tool_used == test_case['expected_tool'],
                    "answer_length": len(final_answer),
                    "trace_steps": len(trace),
                    "timestamp": time.time()
                }
                
                self.results.append(result)
                
                # Print result
                status = "✅ PASS" if result['success'] else "❌ FAIL"
                print(f"Result: {status}")
                print(f"Tool used: {tool_used}")
                print(f"Total latency: {total_latency:.3f}s")
                print(f"Answer preview: {final_answer[:100]}...")
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
                self.results.append({
                    "test_id": i,
                    "query": test_case['query'],
                    "expected_tool": test_case['expected_tool'],
                    "actual_tool": "ERROR",
                    "category": test_case['category'],
                    "description": test_case['description'],
                    "total_latency": 0,
                    "tool_latency": 0,
                    "success": False,
                    "error": str(e),
                    "timestamp": time.time()
                })
        
        return self.results
    
    def generate_report(self) -> Dict:
        """Generate comprehensive evaluation report"""
        if not self.results:
            return {"error": "No test results available"}
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.get('success', False))
        failed_tests = total_tests - passed_tests
        
        # Calculate metrics by category
        categories = {}
        for result in self.results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = {'total': 0, 'passed': 0, 'latencies': []}
            
            categories[cat]['total'] += 1
            if result.get('success', False):
                categories[cat]['passed'] += 1
            categories[cat]['latencies'].append(result.get('total_latency', 0))
        
        # Calculate average latencies
        avg_latencies = {}
        for cat, data in categories.items():
            if data['latencies']:
                avg_latencies[cat] = sum(data['latencies']) / len(data['latencies'])
        
        # Tool usage statistics
        tool_usage = {}
        for result in self.results:
            tool = result.get('actual_tool', 'unknown')
            tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            },
            "category_performance": {
                cat: {
                    "total": data['total'],
                    "passed": data['passed'],
                    "success_rate": (data['passed'] / data['total']) * 100 if data['total'] > 0 else 0,
                    "avg_latency": avg_latencies.get(cat, 0)
                }
                for cat, data in categories.items()
            },
            "tool_usage": tool_usage,
            "latency_stats": {
                "overall_avg": sum(r.get('total_latency', 0) for r in self.results) / total_tests,
                "tool_avg": sum(r.get('tool_latency', 0) for r in self.results) / total_tests,
                "max_latency": max(r.get('total_latency', 0) for r in self.results),
                "min_latency": min(r.get('total_latency', 0) for r in self.results)
            },
            "detailed_results": self.results
        }
        
        return report
    
    def save_report(self, filename: str = "evaluation_report.json"):
        """Save evaluation report to file"""
        report = self.generate_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📊 Report saved to {filename}")

if __name__ == "__main__":
    print("🚀 Starting Mini Agentic Pipeline Evaluation")
    print("=" * 60)
    
    test_suite = TestSuite()
    results = test_suite.run_tests()
    
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    
    report = test_suite.generate_report()
    
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"Average Latency: {report['latency_stats']['overall_avg']:.3f}s")
    
    print("\n📈 Category Performance:")
    for cat, perf in report['category_performance'].items():
        print(f"  {cat}: {perf['passed']}/{perf['total']} ({perf['success_rate']:.1f}%) - {perf['avg_latency']:.3f}s avg")
    
    print("\n🔧 Tool Usage:")
    for tool, count in report['tool_usage'].items():
        print(f"  {tool}: {count} times")
    
    # Save detailed report
    test_suite.save_report()
    
    print(f"\n✅ Evaluation complete! Check evaluation_report.json for detailed results.")
