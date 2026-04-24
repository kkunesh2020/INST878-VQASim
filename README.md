# BLV VQA Multi-Agent Pipeline

Python project for analyzing blind and low-vision (BLV) VQA interactions using a sequential multi-agent pipeline powered by the OpenAI API.

## Pipeline

For each interaction, the system runs four agents in order:

1. Context Generator (`ContextAgent`)
2. Task Generator (`TaskAgent`)
3. Follow-up Question Generator (`QuestionAgent`)
4. Ground-Truth Response Target Analyzer (`ResponseTargetAgent`)

The pipeline processes one interaction at a time and saves:

- Structured JSON output for downstream analysis
- Human-readable text output for qualitative review

## Requirements

- Python 3.11+
- OpenAI API key
- Standard library `argparse` for the CLI

Install dependencies:

```bash
pip install -r requirements.txt
```

## Environment Setup

1. Copy `.env.example` to `.env`
2. Add your API key:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o
```

## Run the Pipeline

Example command:

```bash
python main.py --participant P3 --source diary --interaction 1
```

With optional demographics and few-shot context:

```bash
python main.py --participant P3 --source diary --interaction 1 --include-demographics --include-fewshot
```

Run all valid interactions for one participant in batch mode:

```bash
python batch_run.py --participant P3 --source diary
```

Preview which interaction IDs will run without calling the API:

```bash
python batch_run.py --participant P3 --source diary --dry-run
```

Arguments:

- `--participant`: Participant ID (e.g., `P1`, `P2`, `P3`)
- `--source`: `diary` or `inlab`
- `--interaction`: 1-based interaction index
- `--include-demographics`: optionally inject participant demographics prompt context
- `--include-fewshot`: optionally inject 6 random participant examples from the selected source

Batch mode (`batch_run.py`) arguments:

- `--participant`: Participant ID (e.g., `P1`, `P2`, `P3`)
- `--source`: `diary` or `inlab`
- `--include-demographics`: include participant demographics prompt context
- `--include-fewshot`: include 6 random participant examples from the selected source
- `--dry-run`: list selected interaction IDs without running the pipeline
- `--stop-on-error`: stop batch execution on first failed interaction

Batch filtering rule:

- Runs only interactions with a participant response.
- Skips interactions whose question category is `No question` (case-insensitive, including values like `NO question`).

## Project Structure

```text
project_root/
|-- main.py
|-- config.py
|-- requirements.txt
|-- README.md
|-- .env.example
|
|-- agents/
|   |-- context_agent.py
|   |-- task_agent.py
|   |-- question_agent.py
|   `-- response_target_agent.py
|
|-- prompts/
|   |-- context_prompt.txt
|   |-- task_prompt.txt
|   |-- question_prompt.txt
|   |-- response_target_prompt.txt
|   |-- shared_prompt.txt
|   `-- optional/
|       |-- demographics_prompt.txt
|       `-- fewshot_prompt.txt
|
|-- utils/
|   |-- data_loader.py
|   |-- image_utils.py
|   |-- formatter.py
|   `-- openai_client.py
|
|-- outputs2/
|   |-- json/
|   `-- readable/
|
`-- participant_data/
    |-- P1/
    |-- P2/
    |-- P3/
    |-- diary_data/
    |   |-- P3.json
    |   `-- P3_images/
    `-- inlab_data/
        |-- P3_inlab.json
        `-- P3_inlab_images/
```

## Notes on Data Flexibility

`utils/data_loader.py` is designed to tolerate varying JSON schemas. It attempts to detect common key names for:

- Image paths
- AI-generated description
- Ground-truth participant question

If image paths are relative, the loader tries to resolve them relative to expected source folders.

## Output Files

Each run writes:

- `outputs2/json/{participant}_{source}_{interaction}.json`
- `outputs2/readable/{participant}_{source}_{interaction}.txt`
- Batch runs also write `outputs2/json/{participant}_{source}_batch_comparison.json`

If optional prompts are enabled, suffixes are appended before the extension:

- `_demo` when `--include-demographics` is used
- `_fewshot` when `--include-fewshot` is used
- `_demo_fewshot` when both are used

JSON output includes:

- Input data
- Context agent output
- Task agent output
- Question agent output
- Response-target agent output (for participant ground-truth question)
- Final ranked questions
- Ground-truth question
- Per-question comparison scores for generated question vs. ground truth:
    - `target_match_score` in `{0, 0.5, 1}`
    - `task_type_match_score` in `{0, 0.5, 1}`

Batch comparison output includes one entry per interaction run in the batch, with the interaction ID, ground-truth question/target/type, and all generated responses with their targets, types, and match scores.

Readable output includes sections for input, contexts, tasks, generated questions, generated question details (including task type and response target), ground truth, and comparison scores.

## Prompt Files

Prompt templates are loaded from disk at runtime. Replace placeholder content in `prompts/*.txt` with your real prompts.

## Output Parsing

The prompt templates are designed to return JSON lists. The runtime normalizes those lists into a consistent dictionary shape for downstream use:

- Context agent: `{"contexts": [...]}`
- Task agent: `{"tasks": [...]}`
- Question agent: `{"ranked_questions": [...], "candidates": [...]}`

If a model adds surrounding prose or markdown fences, the parser strips and recovers the embedded JSON automatically.
