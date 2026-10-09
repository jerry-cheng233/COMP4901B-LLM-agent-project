"""
Run evaluation on the NQ test set.

This script runs both the no-search baseline and the search-augmented agent
on the Natural Questions test set and saves the results.
"""

import os
import json
import argparse
from tqdm import tqdm
from agent import SearchAgent, DirectQA, BrowsingAgent


def load_dataset(filepath: str):
    """Load the NQ test dataset from JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def run_no_search(
    dataset: list,
    api_key: str,
    output_file: str,
    model: str = "deepseek-chat"
):
    """Run the no-search baseline on the dataset."""
    qa = DirectQA(api_key=api_key, model=model)
    
    results = []
    for item in tqdm(dataset, desc="Running no-search baseline"):
        question = item["question"]
        
        try:
            answer = qa.answer(question)
        except Exception as e:
            print(f"Error on question '{question}': {e}")
            answer = ""
        
        result = {
            "id": item["id"],
            "question": question,
            "answers": item["answers"],
            "llm_response": answer
        }
        results.append(result)
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(results)} predictions to {output_file}")
    return results


def run_with_search(
    dataset: list,
    api_key: str,
    serper_key: str,
    predictions_file: str,
    trajectories_file: str,
    model: str = "deepseek-chat",
    max_search_steps: int = 3,
    num_search_results: int = 5
):
    """Run the search-augmented agent on the dataset."""
    agent = SearchAgent(
        api_key=api_key,
        serper_key=serper_key,
        model=model,
        max_search_steps=max_search_steps,
        num_search_results=num_search_results
    )
    
    predictions = []
    trajectories = []
    
    for item in tqdm(dataset, desc="Running search agent"):
        question = item["question"]
        
        try:
            result = agent.run(question)
            answer = result["final_answer"]
        except Exception as e:
            print(f"Error on question '{question}': {e}")
            answer = ""
            result = {
                "question": question,
                "steps": [],
                "final_answer": answer,
                "total_search_steps": 0
            }
        
        # Save prediction
        pred = {
            "id": item["id"],
            "question": question,
            "answers": item["answers"],
            "llm_response": answer
        }
        predictions.append(pred)
        
        # Save trajectory
        traj = {
            "id": item["id"],
            "question": question,
            "ground_truths": item["answers"],
            "trajectory": result
        }
        trajectories.append(traj)
    
    # Save predictions
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"Saved {len(predictions)} predictions to {predictions_file}")
    
    # Save trajectories
    with open(trajectories_file, 'w', encoding='utf-8') as f:
        for t in trajectories:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
    print(f"Saved {len(trajectories)} trajectories to {trajectories_file}")
    
    return predictions, trajectories


def run_with_browsing(
    dataset: list,
    api_key: str,
    serper_key: str,
    predictions_file: str,
    trajectories_file: str,
    model: str = "deepseek-chat",
    max_steps: int = 4,
    num_search_results: int = 5
):
    """Run the browsing-enabled agent on the dataset."""
    agent = BrowsingAgent(
        api_key=api_key,
        serper_key=serper_key,
        model=model,
        max_steps=max_steps,
        num_search_results=num_search_results
    )

    predictions = []
    trajectories = []

    for item in tqdm(dataset, desc="Running browsing agent"):
        question = item["question"]

        try:
            result = agent.run(question)
            answer = result["final_answer"]
        except Exception as e:
            print(f"Error on question '{question}': {e}")
            answer = ""
            result = {
                "question": question,
                "steps": [],
                "final_answer": answer,
                "total_search_steps": 0,
                "total_browse_steps": 0
            }

        # Save prediction
        pred = {
            "id": item["id"],
            "question": question,
            "answers": item["answers"],
            "llm_response": answer
        }
        predictions.append(pred)

        # Save trajectory
        traj = {
            "id": item["id"],
            "question": question,
            "ground_truths": item["answers"],
            "trajectory": result
        }
        trajectories.append(traj)

    # Save predictions
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"Saved {len(predictions)} predictions to {predictions_file}")

    # Save trajectories
    with open(trajectories_file, 'w', encoding='utf-8') as f:
        for t in trajectories:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
    print(f"Saved {len(trajectories)} trajectories to {trajectories_file}")

    return predictions, trajectories


def main():
    parser = argparse.ArgumentParser(description="Run NQ evaluation")
    parser.add_argument("--mode", choices=["no_search", "search", "browse", "both", "all"],
                       default="both", help="Evaluation mode")
    parser.add_argument("--data", default="data/nq_test_100.jsonl",
                       help="Path to test data")
    parser.add_argument("--deepseek_key", default=None,
                       help="DeepSeek API key (or set DEEPSEEK_API_KEY env var)")
    parser.add_argument("--serper_key", default=None,
                       help="Serper API key (or set SERPER_API_KEY env var)")
    parser.add_argument("--model", default="deepseek-chat",
                       help="Model name")
    parser.add_argument("--max_search_steps", type=int, default=3,
                       help="Max search steps for agent")
    parser.add_argument("--num_search_results", type=int, default=5,
                       help="Number of search results per query")
    
    args = parser.parse_args()
    
    # Get API keys from args or environment
    deepseek_key = args.deepseek_key or os.environ.get("DEEPSEEK_API_KEY")
    serper_key = args.serper_key or os.environ.get("SERPER_API_KEY")

    if not deepseek_key:
        print("Error: DEEPSEEK_API_KEY not set. Please set it as environment variable or pass --deepseek_key")
        return

    # Load dataset
    print(f"Loading dataset from {args.data}...")
    dataset = load_dataset(args.data)
    print(f"Loaded {len(dataset)} questions")

    if args.mode in ["no_search", "both"]:
        print("\n=== Running No-Search Baseline ===")
        run_no_search(
            dataset=dataset,
            api_key=deepseek_key,
            output_file="results/predictions_nosearch.jsonl",
            model=args.model
        )

    if args.mode in ["search", "both", "all"]:
        if not serper_key:
            print("Error: SERPER_API_KEY not set. Please set it as environment variable or pass --serper_key")
            return
        print("\n=== Running Search Agent ===")
        run_with_search(
            dataset=dataset,
            api_key=deepseek_key,
            serper_key=serper_key,
            predictions_file="results/predictions_search.jsonl",
            trajectories_file="results/agent_trajectories.jsonl",
            model=args.model,
            max_search_steps=args.max_search_steps,
            num_search_results=args.num_search_results
        )

    if args.mode in ["browse", "all"]:
        if not serper_key:
            print("Error: SERPER_API_KEY not set. Please set it as environment variable or pass --serper_key")
            return
        print("\n=== Running Browsing Agent ===")
        run_with_browsing(
            dataset=dataset,
            api_key=deepseek_key,
            serper_key=serper_key,
            predictions_file="results/predictions_browse.jsonl",
            trajectories_file="results/browse_trajectories.jsonl",
            model=args.model,
            max_steps=args.max_search_steps + 1,  # Allow one extra step for browsing
            num_search_results=args.num_search_results
        )


if __name__ == "__main__":
    main()

