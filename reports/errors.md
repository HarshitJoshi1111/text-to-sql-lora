# Why the wrong answers are wrong

Every wrong answer that runs is compared with the true query part by part (`python -m src.analyze`). Shares are of all evaluation questions; one answer can have several problems.

| Run | Wrong | Does not run | Other query shape | wrong aggregation | wrong SELECT column | condition on a wrong column | condition missing | extra condition | wrong operator | value: other words of the question | value: changed or invented |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen2.5-1.5b-lora-2k | 12.6% | 0.2% | 0.0% | 5.8% | 2.6% | 2.8% | 2.0% | 1.0% | 0.0% | 1.4% | 1.0% |
| qwen2.5-0.5b-lora-2k-r4 | 19.2% | 1.0% | 0.4% | 6.4% | 2.2% | 4.6% | 3.0% | 2.0% | 0.8% | 2.8% | 2.0% |
| qwen2.5-0.5b-lora | 21.8% | 1.0% | 0.0% | 6.6% | 3.0% | 6.2% | 3.6% | 3.2% | 0.6% | 3.0% | 2.6% |
| qwen2.5-0.5b-lora-2k | 23.2% | 2.0% | 0.0% | 6.2% | 3.2% | 6.4% | 3.2% | 3.6% | 0.0% | 3.4% | 2.4% |
| qwen2.5-0.5b-lora-1k | 26.8% | 3.2% | 0.0% | 6.4% | 4.0% | 6.2% | 4.2% | 3.6% | 0.4% | 5.2% | 2.8% |
| qwen2.5-0.5b-lora-2k-r16 | 27.2% | 4.0% | 0.0% | 6.8% | 4.0% | 6.6% | 4.2% | 3.4% | 0.2% | 3.2% | 4.0% |
| qwen2.5-1.5b | 53.4% | 37.0% | 5.2% | 3.2% | 1.8% | 2.2% | 0.6% | 2.6% | 0.4% | 1.4% | 2.8% |
| qwen2.5-0.5b | 76.8% | 53.8% | 8.2% | 2.8% | 2.2% | 2.0% | 0.8% | 4.4% | 0.4% | 3.4% | 4.8% |

Value problems are split by whether the generated value is still a piece of the question (the model picked the wrong words) or not (it changed a name, a spelling or a number). Other query shape: runs, but not in the WikiSQL form (ORDER BY, LIKE, OR, ...).

## Would the table cells have helped?

The prompt shows the column names and 3 example rows, not the whole table. For '=' conditions:

| Run | Condition on a wrong column, value only in the true column | Generated value not in the table |
|---|---|---|
| qwen2.5-1.5b-lora-2k | 13 of 14 answers | 12 of 62 wrong answers that run |
| qwen2.5-0.5b-lora-2k-r4 | 22 of 23 answers | 24 of 89 wrong answers that run |
| qwen2.5-0.5b-lora | 28 of 31 answers | 31 of 104 wrong answers that run |
| qwen2.5-0.5b-lora-2k | 31 of 32 answers | 33 of 106 wrong answers that run |
| qwen2.5-0.5b-lora-1k | 29 of 31 answers | 37 of 118 wrong answers that run |
| qwen2.5-0.5b-lora-2k-r16 | 32 of 33 answers | 44 of 116 wrong answers that run |
| qwen2.5-1.5b | 10 of 11 answers | 13 of 56 wrong answers that run |
| qwen2.5-0.5b | 10 of 10 answers | 45 of 74 wrong answers that run |

<!-- notes -->
## Notes on the fine-tuned model (qwen2.5-0.5b-lora, 109 wrong answers)

**Most WHERE mistakes come from not seeing the table.** The prompt holds the column names and 3 example rows. To put a name like "misa eguchi eri hozumi" on the right column, the model has to guess whether it is an opponent or a partner. In 31 answers a condition has the right value on a wrong column; in 28 of them the value occurs in the table only under the true column. In 31 of the 104 wrong answers that run, a generated value occurs nowhere in the table: small copying errors ("ghh" for "hgh", "brady brown" for "Brady Browne") or extra words ("usl pdl league" for "USL PDL"). Both could be caught by looking up the question's words in the table cells, which is the plan for the next step.

**Part of the aggregation errors are errors in the dataset.** WikiSQL's aggregation labels are known to be noisy. Checked by hand, the 33 answers with a different aggregation from the true query split into:

- 17 where the generated aggregation fits the question and the true one does not. In 8 of them it is the only difference, so a right answer is scored as wrong (1.6 % of the questions). Examples: "What is the total attendance for Candlestick Park?" (true query `COUNT`, generated `SUM`), "In which year did he finish 11th?" (true `COUNT`, generated none), "How many years have a conventional plan of 7%?" (true `SUM("Year")`, generated `COUNT`).
- 9 where the model is wrong, mostly `COUNT` for a "how many" question whose answer is a number in a cell: "How many hours were flown in each of the years where ...".
- 7 where neither fits, e.g. "What was the crowd number when the home team was Carlton?" (true `MIN`, generated `COUNT`).

The reported execution accuracy is therefore a slight underestimate. The evaluation set is left as it is, so that the numbers stay comparable with other WikiSQL results.
