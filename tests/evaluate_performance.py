# tests/evaluate_performance.py  (needs the server running: python main.py)
import statistics
import sys
import time

import requests

API_URL = "http://localhost:8000"

CASES = [
    ("Show me total revenue for July 2017.", "3,132,959.07"),
    ("What is the average price of a hotel booking?", "101.83"),
    ("Which locations had the highest booking cancellations?", "Portugal"),
    ("What is the cancellation rate for resort hotels?", "27.76%"),
    ("What is the average lead time for Portugal?", "115.57"),
    ("Which month had the highest revenue in 2016?", "August 2016"),
]


def main():
    passed, times = 0, []
    for question, expected in CASES:
        t = time.time()
        r = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=60)
        ms = (time.time() - t) * 1000
        times.append(ms)
        ok = r.status_code == 200 and expected in r.json()["answer"]
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {ms:7.0f} ms  {question}")
    print(f"\nAccuracy: {passed}/{len(CASES)}  |  mean {statistics.mean(times):.0f} ms")

    t = time.time()
    r = requests.post(f"{API_URL}/analytics", json={}, timeout=120)
    print(f"/analytics: status {r.status_code}, {(time.time() - t) * 1000:.0f} ms (first call, uncached)")
    sys.exit(0 if passed == len(CASES) else 1)


if __name__ == "__main__":
    main()
