import asyncio
import argparse
import json
from pathlib import Path
import pandas as pd
import ir_datasets
import aiolimiter
from openai import AsyncOpenAI, APITimeoutError, APIError, RateLimitError
from tqdm.asyncio import tqdm
from dotenv import dotenv_values
import tiktoken

# Tier 5: 10,000 RPM, 30M TPM
RATE_LIMIT_RPM = 9500

# GPT-4o pricing per 1M tokens
GPT4O_INPUT_COST = 2.50
GPT4O_OUTPUT_COST = 10.00

MAX_RETRIES = 3


def get_messages_few_shot(query, document, examples):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. "
        },
        {
            "role": "user",
            "content": f"Query: \"{query}\"\nPassage: \"{document}\"\nScore the relevance of the passage to the query using the scale from 0 to 3. Provide only the integer score as your response, with no additional text."
        }
    ]
    for ex_query, ex_document, ex_relevance in examples:
        messages.append({"role": "user", "content": f"Query: {ex_query}\nPassage: {ex_document[:4096]}\nRelevance: "})
        messages.append({"role": "assistant", "content": f"{ex_relevance}"})

    messages.append({
        "role": "user",
        "content": f"Query: {query}\nPassage: {document}\nRelevance: "
    })
    return messages


def get_messages_umbrela(query, document, examples):
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant. "
        },
        {
            "role": "user",
            "content": f"Query: \"{query}\"\nPassage: \"{document}\"\nScore the relevance of the passage to the query using the scale from 0 to 3. Provide only the integer score as your response, with no additional text."
        }
    ]


def get_messages_human(query, document, examples):
    return [
        {
            "role": "system",
            "content": "Please assess the Relevance of the provided Passage to the following Query. Please only output \"1\" (relevant) or \"0\" (irrelevant). "
        },
        {
            "role": "user",
            "content": f"Query: {query}\nPassage: {document}\nRelevance:"
        }
    ]


def count_tokens(messages, model="gpt-4o"):
    """Count tokens in messages using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 3
    for message in messages:
        num_tokens += 3
        for key, value in message.items():
            num_tokens += len(encoding.encode(str(value)))
    return num_tokens


def load_data(data_dir):
    qrels = (
        pd.DataFrame(ir_datasets.load("msmarco-passage/trec-dl-2020/judged").qrels_iter())
        .astype({"query_id": str, "doc_id": str})
        .drop("iteration", axis="columns")
        .assign(relevance=lambda df: (df["relevance"] > 0.5).astype(int))
        .merge(
            pd.read_parquet(f"{data_dir}/processed/trecdl-shallow-qrels-256.parquet")
            .astype({"query_id": str, "doc_id": str})
            .assign(relevance=lambda df: (df["relevance"] > 0.5).astype(int))
            .rename({"relevance": "relevance_adapter_256"}, axis="columns"),
            how="left",
            on=["query_id", "doc_id"],
        )
        .merge(pd.read_parquet(f"{data_dir}/external/train-trecdl.parquet"))
        .assign(train=lambda df: df["relevance_adapter_256"].isna())
        .drop(["relevance_adapter_256"], axis=1)
    )

    example_qrels = qrels.query("train == True").drop("train", axis="columns")
    predict_qrels = qrels.query("train == False").drop("train", axis="columns")
    return example_qrels, predict_qrels


def load_errors(error_file, predict_qrels):
    """Load errors and return filtered predict_qrels and prompt configs"""
    errors = []
    with open(error_file) as f:
        for line in f:
            errors.append(json.loads(line))

    # Create set of (query_id, doc_id, prompt) tuples to retry
    error_configs = []
    for error in errors:
        query_id = error["query_id"]
        doc_id = error["doc_id"]
        prompt_str = error["prompt"]

        # Parse prompt string to extract prompt type and num_examples
        if "_" in prompt_str and prompt_str.startswith("few_shot"):
            prompt = "few_shot"
            num_examples = 8 #int(prompt_str.split("_")[-1])
        else:
            prompt = prompt_str
            num_examples = 8  # default for non-few_shot

        error_configs.append((query_id, doc_id, prompt, num_examples))

    print(f"Loaded {len(error_configs)} failed requests to retry")
    return error_configs


async def process_request(row, get_messages, prompt, num_examples,
                          example_qrels, client, rate_limiter, semaphore, lock,
                          output_file, error_file, dry_run=False):
    query = row["query"].replace('\n', ' ').strip()
    document = row["doc"].replace('\n', ' ').strip()
    examples = (
        example_qrels
        .groupby("relevance")
        .head(int(num_examples / 2))
        .loc[:, ["query", "doc", "relevance"]]
        .sample(frac=1.0)
        .values.tolist()
    )
    messages = get_messages(
        query, document,
        [(q.replace('\n', ' ').strip(), d.replace('\n', ' ').strip(), r)
         for q, d, r in examples]
    )

    async with semaphore:
        if dry_run:
            input_tokens = count_tokens(messages)
            output_tokens = 3
            return {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }

        # Retry logic
        for attempt in range(MAX_RETRIES):
            try:
                async with rate_limiter:
                    completion = await client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        timeout=60.0
                    )

                    result = {
                        "query_id": row["query_id"],
                        "doc_id": row["doc_id"],
                        "prompt": prompt + f"_{num_examples}" if prompt == "few_shot" else prompt,
                        "relevance": completion.choices[0].message.content
                    }

                    async with lock:
                        with open(output_file, "a") as f:
                            f.write(json.dumps(result) + "\n")

                    return result

            except (APITimeoutError, APIError, RateLimitError) as e:
                if attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                else:
                    error_record = {
                        "query_id": row["query_id"],
                        "doc_id": row["doc_id"],
                        "prompt": prompt + f"_{num_examples}" if prompt == "few_shot" else prompt,
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                    async with lock:
                        with open(error_file, "a") as f:
                            f.write(json.dumps(error_record) + "\n")
                    return None
            except Exception as e:
                error_record = {
                    "query_id": row["query_id"],
                    "doc_id": row["doc_id"],
                    "prompt": prompt + f"_{num_examples}" if prompt == "few_shot" else prompt,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                async with lock:
                    with open(error_file, "a") as f:
                        f.write(json.dumps(error_record) + "\n")
                return None


async def main(data_dir, output_file, max_concurrent, dry_run, retry_errors_file):
    config = dotenv_values()
    client = AsyncOpenAI(api_key=config["OPENAI_API_KEY"]) if not dry_run else None

    print("Loading data...")
    example_qrels, predict_qrels = load_data(data_dir)

    error_file = output_file.replace(".jsonl", "_errors.jsonl")

    if not dry_run and not retry_errors_file:
        Path(output_file).unlink(missing_ok=True)
        Path(error_file).unlink(missing_ok=True)

    rate_limiter = aiolimiter.AsyncLimiter(max_rate=RATE_LIMIT_RPM, time_period=60) if not dry_run else None
    semaphore = asyncio.Semaphore(max_concurrent)
    lock = asyncio.Lock()

    # Map prompt names to functions
    prompt_map = {
        "few_shot": get_messages_few_shot,
        "umbrela": get_messages_umbrela,
        "human": get_messages_human
    }

    tasks = []

    if retry_errors_file:
        # Retry mode: only process failed requests
        error_configs = load_errors(retry_errors_file, predict_qrels)

        for query_id, doc_id, prompt, num_examples in error_configs:
            # Find the row in predict_qrels
            row = predict_qrels[
                (predict_qrels["query_id"] == query_id) &
                (predict_qrels["doc_id"] == doc_id)
                ]

            if len(row) == 0:
                print(f"Warning: Could not find query_id={query_id}, doc_id={doc_id} in data")
                continue

            row = row.iloc[0]
            get_messages = prompt_map[prompt]

            task = process_request(
                row, get_messages, prompt, num_examples,
                example_qrels, client, rate_limiter, semaphore, lock,
                output_file, error_file, dry_run
            )
            tasks.append(task)
    else:
        # Normal mode: process all requests
        print(f"Loaded {len(predict_qrels)} queries to process")

        for _, row in predict_qrels.iterrows():
            for get_messages, prompt in [
                (get_messages_few_shot, "few_shot_umbrela"),
                #(get_messages_umbrela, "umbrela"),
                #(get_messages_human, "human")
            ]:
                for num_examples in [8]:
                    task = process_request(
                        row, get_messages, prompt, num_examples,
                        example_qrels, client, rate_limiter, semaphore, lock,
                        output_file, error_file, dry_run
                    )
                    tasks.append(task)

                    if prompt != "few_shot":
                        break

    print(f"Processing {len(tasks)} requests...")
    results = []
    for coro in tqdm.as_completed(tasks, total=len(tasks)):
        result = await coro
        if result is not None:
            results.append(result)

    if dry_run:
        total_input_tokens = sum(r["input_tokens"] for r in results)
        total_output_tokens = sum(r["output_tokens"] for r in results)

        input_cost = (total_input_tokens / 1_000_000) * GPT4O_INPUT_COST
        output_cost = (total_output_tokens / 1_000_000) * GPT4O_OUTPUT_COST
        total_cost = input_cost + output_cost

        print("\n" + "=" * 60)
        print("DRY RUN COST ESTIMATE")
        print("=" * 60)
        print(f"Total requests: {len(results):,}")
        print(f"Input tokens:   {total_input_tokens:,} (${input_cost:.2f})")
        print(f"Output tokens:  {total_output_tokens:,} (${output_cost:.2f})")
        print(f"Total cost:     ${total_cost:.2f}")
        print(
            f"Avg tokens/req: {total_input_tokens / len(results):.0f} input, {total_output_tokens / len(results):.0f} output")
        print("=" * 60)
    else:
        failed_count = len(tasks) - len(results)
        print(f"\nComplete! Results written to {output_file}")
        print(f"Successful: {len(results):,} | Failed: {failed_count:,}")
        if failed_count > 0:
            print(f"Failed requests logged to {error_file}")
            print(f"\nTo retry failures, run:")
            print(f"  python script.py --retry-errors {error_file} --output {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT-4o relevance inference with async requests")
    parser.add_argument("--data-dir", type=str, default="../data", help="Path to data directory")
    parser.add_argument("--output", type=str, default="predictions_few_shot_umbrela.jsonl", help="Output file path")
    parser.add_argument("--max-concurrent", type=int, default=500, help="Max concurrent requests")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost without making API calls")
    parser.add_argument("--retry-errors", type=str, help="Path to errors file to retry failed requests")

    args = parser.parse_args()
    asyncio.run(main(args.data_dir, args.output, args.max_concurrent, args.dry_run, args.retry_errors))