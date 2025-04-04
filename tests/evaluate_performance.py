# tests/evaluate_performance.py
import requests
import time
import pandas as pd
import json
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

API_URL = "http://localhost:8000"

def test_api_response_time(endpoint: str, payload: Dict, n_requests: int = 10) -> List[float]:
    """Test API response time for a specific endpoint"""
    response_times = []
    
    for _ in range(n_requests):
        start_time = time.time()
        response = requests.post(f"{API_URL}/{endpoint}", json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            response_times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Sleep to avoid overwhelming the API
        time.sleep(0.2)
    
    return response_times

def evaluate_qa_accuracy(test_questions: List[Dict]) -> Dict:
    """Evaluate the accuracy of QA responses"""
    results = []
    
    for test_case in test_questions:
        question = test_case["question"]
        expected_answer = test_case["expected_answer"]
        
        # Send question to API
        response = requests.post(f"{API_URL}/ask", json={"question": question})
        
        if response.status_code == 200:
            data = response.json()
            actual_answer = data["answer"]
            
            # Simple exact match for now - could use more sophisticated metrics
            exact_match = expected_answer.lower() in actual_answer.lower()
            
            results.append({
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": actual_answer,
                "exact_match": exact_match,
                "processing_time_ms": data["metadata"]["processing_time_ms"]
            })
    
    # Calculate accuracy
    accuracy = sum(1 for r in results if r["exact_match"]) / len(results) if results else 0
    
    return {
        "accuracy": accuracy,
        "results": results
    }

def plot_response_times(analytics_times: List[float], qa_times: List[float]):
    """Plot API response times"""
    plt.figure(figsize=(10, 6))
    
    # Create DataFrame for plotting
    data = pd.DataFrame({
        "Endpoint": ["Analytics"] * len(analytics_times) + ["QA"] * len(qa_times),
        "Response Time (ms)": analytics_times + qa_times
    })
    
    # Create box plot
    sns.boxplot(x="Endpoint", y="Response Time (ms)", data=data)
    plt.title("API Response Times")
    plt.tight_layout()
    plt.savefig("response_times.png")
    
    # Print statistics
    analytics_stats = {
        "min": min(analytics_times),
        "max": max(analytics_times),
        "mean": sum(analytics_times) / len(analytics_times),
        "median": sorted(analytics_times)[len(analytics_times) // 2]
    }
    
    qa_stats = {
        "min": min(qa_times),
        "max": max(qa_times),
        "mean": sum(qa_times) / len(qa_times),
        "median": sorted(qa_times)[len(qa_times) // 2]
    }
    
    print("Analytics Endpoint Stats (ms):")
    for key, value in analytics_stats.items():
        print(f"  {key}: {value:.2f}")
    
    print("\nQA Endpoint Stats (ms):")
    for key, value in qa_stats.items():
        print(f"  {key}: {value:.2f}")

def main():
    print("Starting Performance Evaluation...")
    
    # Test questions with expected answers
    test_questions = [
        {
            "question": "Show me total revenue for July 2017.",
            "expected_answer": "The total revenue for July 2017"
        },
        {
            "question": "Which countries had the highest booking cancellations?",
            "expected_answer": "highest booking cancellations"
        },
        {
            "question": "What is the average price of a hotel booking?",
            "expected_answer": "average price"
        },
        {
            "question": "What is the cancellation rate?",
            "expected_answer": "cancellation rate"
        },
        {
            "question": "What is the average lead time for bookings?",
            "expected_answer": "lead time"
        }