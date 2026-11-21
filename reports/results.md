# Results on the evaluation set

500 questions from the WikiSQL test split (tables never seen in training). Execution accuracy: the query returns the same result as the true query. Runs: the query executes without error.

| Run | Model | Execution accuracy | Exact match | Runs | No aggregation | Aggregation | 2+ conditions | s/question |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | Qwen2.5-0.5B-Instruct-4bit | **23.2%** | 14.2% | 46.2% | 24.0% | 20.9% | 18.5% | 0.525 |
| qwen2.5-1.5b | Qwen2.5-1.5B-Instruct-4bit | **46.4%** | 34.6% | 63.0% | 50.9% | 33.3% | 41.9% | 1.602 |
| qwen2.5-0.5b-lora | Qwen2.5-0.5B-Instruct-4bit + LoRA | **78.2%** | 70.4% | 99.0% | 82.8% | 65.1% | 77.4% | 0.374 |

## Where the wrong answers come from (share of all questions)

| Run | unknown column | syntax error | wrong SELECT | wrong WHERE | both wrong | other |
|---|---|---|---|---|---|---|
| qwen2.5-0.5b | 25.4% | 23.4% | 3.4% | 11.2% | 5.2% | 8.2% |
| qwen2.5-1.5b | 14.4% | 19.2% | 3.6% | 6.6% | 4.8% | 5.0% |
| qwen2.5-0.5b-lora | 0.8% | 0.0% | 5.6% | 11.6% | 3.6% | 0.2% |
