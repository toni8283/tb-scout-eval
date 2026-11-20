# TB Scout Eval

A startup says it was valued at ₹7 crore. Someone asks how much it raised in its seed round. An assistant can quote the right number from the wrong record and still sound convincing.

TB Scout Eval checks for that mistake. It gives a model two short company records and one question, then checks whether the model answers with the right evidence, says the information is missing, or flags a conflict. The questions are inspired by funding and company research in TB Scout. All eight companies and their records are fictional.

## Try it in a minute

Open a terminal in this folder. The saved runs use Python 3.10+ and the standard library. Scoring works offline.

```sh
python evaluate.py --summary
python evaluate.py --case Q03
python -m unittest -v
```

The second command shows what the model saw, the expected answer, what each run returned, and which checks failed. Use `python evaluate.py` for the full per-run report. On Windows, you can use `py` if that is how you launch Python.

## What happened when the prompt got more specific?

One run used Qwen2.5-1.5B-Instruct with a basic structured-answer prompt. A second run used the same model and added rules for matching the exact metric, date and population, and citing both records when they disagree. Each condition made 24 fresh calls. The dataset and inference settings were identical across the two runs.

| Run | Answerable | Missing evidence | Conflicting evidence | Strict passes | Unsupported answers |
|---|---:|---:|---:|---:|---:|
| Qwen · basic prompt | 2/8 | 2/8 | 2/8 | **6/24** | 3 |
| Qwen · evidence rules | 0/8 | 2/8 | 2/8 | **4/24** | 6 |
| Keyword overlap reference | 8/8 | 1/8 | 0/8 | 9/24 | 15 |
| Always report insufficient evidence | 0/8 | 8/8 | 0/8 | 8/24 | 0 |

An **unsupported answer** is an `answered` decision when the records are missing or conflicting. The keyword reference policy uses task-specific extraction rules and can only choose one source. The always-abstain policy shows how a system can avoid unsupported claims while answering none of the answerable questions.

The evidence rules fixed two cases and regressed on four. In **Q22**, the records report seed rounds of ₹7 crore and ₹4 crore for Nila Robotics. Both prompts flagged the conflict. The evidence prompt also cited both records, which made that answer pass. In **Q06**, the evidence prompt correctly returned 18 paying clinics but cited the source about 28 free trials as well. The basic prompt cited only the paying-clinic record and passed.

In **Q03**, Nila Robotics' records give its post-money valuation but omit how much it raised. The basic prompt returned ₹7 crore as the round size. The evidence prompt switched to `conflicting_evidence`. The expected decision was `insufficient_evidence` because there was no claim about the round size in either source.

## How the benchmark works

- **Matched questions.** There are eight company questions, each with three versions. The wording stays fixed within each group, and the two records change. This tests whether the model changes its answer when the evidence changes.
- **Deterministic grader.** Every case has an expected decision, value, and set of supporting source IDs. All three checks must pass. The grader accepts numeric equivalents such as `4` and `4.0`, ignores citation order, and rejects extra or repeated citations. A missing answer requires a null value and no citations. A conflict requires a null value and both citations.
- **Saved experiment.** The prompt, settings, raw model responses and parsed answers are available for [the basic run](runs/qwen-basic.json) and [the evidence run](runs/qwen-evidence.json). The [case file](data/cases.json) contains the model inputs and case IDs. The [answer key](data/key.json) is used only for scoring.

The dataset is deliberately small and synthetic. The two prompts were compared on one quantized model with one run per condition. Treat the numbers as observations about these cases. They do not estimate performance on real company filings or establish that stricter prompts generally hurt.

## Run the comparison with a local model

With Ollama running and a model installed, pass its tag to the runner. Each run needs a new output directory.

```sh
python run_model.py qwen2.5:1.5b --output experiment-01
python evaluate.py --summary experiment-01/basic.json experiment-01/evidence.json
```

The checked-in comparison used **llama.cpp b8157 (2943210c1)** on CPU and the Qwen2.5-1.5B-Instruct Q4_K_M GGUF distributed under the Ollama tag `qwen2.5:1.5b`. Its SHA-256 is `183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4`. The saved files also record the dataset hash. [llama.cpp release](https://github.com/ggml-org/llama.cpp/releases/tag/b8157) · [Exact GGUF blob](https://registry.ollama.ai/v2/library/qwen2.5/blobs/sha256:183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4)

To use the same backend, save the weights as `qwen2.5-1.5b.gguf` and give the server the alias below.

```sh
llama-server -m qwen2.5-1.5b.gguf --alias qwen2.5-1.5b-instruct-q4_k_m --host 127.0.0.1 --port 8080 -c 1024 -np 1 -t 6
```

Then, from a second terminal in this project directory:

```sh
python run_model.py qwen2.5-1.5b-instruct-q4_k_m --backend llama.cpp --output experiment-02
python evaluate.py --summary experiment-02/basic.json experiment-02/evidence.json
```

Both prompts request JSON with temperature 0, seed 42 and a 128-token response cap. A fresh inference run can vary with the runtime, model weights or hardware. Scoring the already saved responses yields the same results. The GitHub workflow runs the tests and checks that regenerating the reference policies produces the checked-in files.
