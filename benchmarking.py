
import json
import time
from statistics import mean

from document_loader import vector_store, mmr_retriever, retriever



with open("benchmark.json", encoding="utf-8") as file:
    test_cases = json.load(file)

similarity_retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},
)

retrievers = {
    "Similarity": similarity_retriever,
    "MMR": mmr_retriever,
    "MultiQuery + MMR": retriever,
}


def page_number(doc):
    page = doc.metadata.get("page")
    return page + 1 if isinstance(page, int) else None


def evaluate(name, current_retriever):
    recalls = []
    hits = []
    latencies = []

    for case in test_cases:
        relevant = set(case["relevant_pages"])

        start = time.perf_counter()
        docs = current_retriever.invoke(case["question"])[:2]
        latencies.append(time.perf_counter() - start)

        found = {
            page_number(doc)
            for doc in docs
            if page_number(doc) is not None
        }

        if not relevant:
            continue

        recalls.append(len(found & relevant) / len(relevant))
        hits.append(int(bool(found & relevant)))

    if not recalls:
        return None

    return {
        "Method": name,
        "Recall@2": mean(recalls),
        "Hit@2": mean(hits),
        "Average latency (seconds)": mean(latencies),
    }


results = []

for name, current_retriever in retrievers.items():
    print(f"Evaluating {name}...")
    result = evaluate(name, current_retriever)
    if result:
        results.append(result)

print("\n===== BENCHMARK RESULTS =====")

for result in results:
    print(f"\n{result['Method']}")
    print(f"Recall@2: {result['Recall@2']:.2%}")
    print(f"Hit@2: {result['Hit@2']:.2%}")
    print(
        f"Average latency: "
        f"{result['Average latency (seconds)']:.3f} seconds"
    )

with open("benchmark_results.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

print("\nResults saved to benchmark_results.json")
