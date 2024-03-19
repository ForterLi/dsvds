
---
pretty_name: "WhisperKit ASR Evaluation Results"
viewer: false
library_name: whisperkit
tags:
- whisper
- whisperkit
- coreml
- asr
- quantized
---
# WhisperKit Evaluation Results



## Dataset: `librispeech`
## Explanation: Short-form Audio (<30s/clip) - 5 hours of English audiobook clips

|                                                                                                                                                                            |   WER (↓) |   QoI (↑) |   File Size (MB) |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------:|----------:|-----------------:|
| [WhisperOpenAIAPI/openai_whisper-large-v2](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperOpenAIAPI/openai_whisper-large-v2/librispeech)               |      2.35 |     100   |             3100 |
| [WhisperKit/openai_whisper-large-v3](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-large-v3/librispeech)                           |      2.04 |      95.2 |             3100 |
| [WhisperKit/openai_whisper-large-v3_turbo](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-large-v3_turbo/librispeech)               |      2.03 |      95.4 |             3100 |
| [WhisperKit/openai_whisper-large-v3_turbo_1018MB](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-large-v3_turbo_1018MB/librispeech) |      1.99 |      94.8 |             1018 |
| [WhisperKit/openai_whisper-large-v2](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-large-v2/librispeech)                           |      2.77 |      96.6 |             3100 |
| [WhisperKit/openai_whisper-large-v2_1050MB](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-large-v2_1050MB/librispeech)             |      2.81 |      95   |             1050 |
| [WhisperKit/openai_whisper-large-v2_turbo](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-large-v2_turbo/librispeech)               |      2.76 |      96.6 |             3100 |
| [WhisperKit/openai_whisper-large-v2_turbo_1022MB](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-large-v2_turbo_1022MB/librispeech) |      2.66 |      94.9 |             1022 |
| [WhisperKit/openai_whisper-small.en](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-small.en/librispeech)                           |      3.12 |      85.8 |              483 |
| [WhisperKit/openai_whisper-small](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-small/librispeech)                                 |      3.45 |      83   |              483 |
| [WhisperKit/openai_whisper-base.en](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-base.en/librispeech)                             |      3.98 |      75.3 |              145 |
| [WhisperKit/openai_whisper-base](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-base/librispeech)                                   |      4.97 |      67.2 |              145 |
| [WhisperKit/openai_whisper-tiny.en](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-tiny.en/librispeech)                             |      5.61 |      63.9 |               66 |
| [WhisperKit/openai_whisper-tiny](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-tiny/librispeech)                                   |      7.47 |      52.5 |               66 |
| [whisper.cpp/openai_whisper-large-v3](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/whisper.cpp/openai_whisper-large-v3/librispeech)                         |      1.97 |      95.4 |             3100 |

## Dataset: `earnings22`
## Explanation: Long-Form Audio (>1hr/clip) - 120 hours of earnings call recordings in English with various accents

|                                                                                                                                                             |   WER (↓) |   QoI (↑) |   File Size (MB) |
|:------------------------------------------------------------------------------------------------------------------------------------------------------------|----------:|----------:|-----------------:|
| [WhisperOpenAIAPI/openai_whisper-large-v2](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperOpenAIAPI/openai_whisper-large-v2/earnings22) |     16.27 |     100   |             3100 |
| [WhisperKit/openai_whisper-large-v3](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-large-v3/earnings22)             |     15.17 |      58.5 |             3100 |
| [WhisperKit/openai_whisper-base.en](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-base.en/earnings22)               |     23.49 |       6.5 |              145 |
| [WhisperKit/openai_whisper-tiny.en](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-tiny.en/earnings22)               |     28.64 |       5.7 |               66 |
| [whisper.cpp/openai_whisper-large-v3](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/whisper.cpp/openai_whisper-large-v3/earnings22)           |     33.58 |       6.5 |             3100 |


We believe that rigorously measuring the quality of inference is necessary for developers and
enterprises to make informed decisions when opting to use optimized or compressed variants of
any machine learning model in production. To contextualize `WhisperKit`, we take the following Whisper
implementations and benchmark them using a consistent evaluation harness:

Server-side:
- `WhisperOpenAIAPI`: [OpenAI's Whisper API](https://platform.openai.com/docs/guides/speech-to-text)
($0.36 per hour of audio as of 02/29/24, 25MB file size limit per request)

On-device:
- `WhisperKit`: Argmax's implementation [[Eval Harness]](https://github.com/argmaxinc/whisperkittools/blob/main/whisperkit/pipelines.py#L100) [[Repo]](https://github.com/argmaxinc/WhisperKit)
- `whisper.cpp`: A C++ implementation form ggerganov [[Eval Harness]](https://github.com/argmaxinc/whisperkittools/blob/main/whisperkit/pipelines.py#L212) [[Repo]](https://github.com/ggerganov/whisper.cpp)
- `WhisperMLX`: A Python implementation from Apple MLX [[Eval Harness]](https://github.com/argmaxinc/whisperkittools/blob/main/whisperkit/pipelines.py#L338) [[Repo]](https://github.com/ml-explore/mlx-examples/blob/main/whisper/whisper/transcribe.py)
(All on-device implementations are available for free under MIT license as of 03/19/2024)

`WhisperOpenAIAPI` sets the reference and we assume that it is using the equivalent of [openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2)
in float16 precision along with additional undisclosed optimizations from OpenAI. In all measurements, we care primarily about per-example no-regressions (quantified as `qoi` below)
which is a stricter metric compared to dataset average [Word Error RATE (WER)](https://en.wikipedia.org/wiki/Word_error_rate). A 100% `qoi` preserves perfect backwards-compatibility on the test distribution and avoids "perceived regressions", the phenomenon
where per-example known behavior changes after a code/model update and causes divergence in downstream code or breaks the user experience itself (even if dataset averages might stay flat
across updates). Pseudocode for `qoi`:

```python
qoi = []
for example in dataset:
    no_regression = wer(optimized_model(example)) <= wer(reference_model(example))
    qoi.append(no_regression)
qoi = (sum(qoi) / len(qoi)) * 100.
```

Note that the ordering of models with respect to `WER` does not necessarily match the ordering with respect to `QoI`. This is because the reference model gets assigned
a QoI of 100% by definition. Any per-example regression by other implementations get penalized while per-example improvements are not rewarded. `QoI` (higher is better) matters
where the production behavior is established by the reference results and the goal is to not regress when switching to an optimized or compressed model. On the other hand,
`WER` (lower is better) matters when there is no established production behavior and one is picking the best quality versus model size trade off point.

We anticipate developers that use Whisper (or similar models) in production to have their own Quality Assurance test sets and [whisperkittools](https://github.com/argmaxinc/whisperkittools) offers
the tooling necessary to run the same measurements on such custom test sets, please see the [Model Evaluation on Custom Dataset]((https://github.com/argmaxinc/whisperkittools)) for details.

### Datasets
- [librispeech](https://huggingface.co/datasets/argmaxinc/librispeech): ~5 hours of short English audio clips, tests short-form transcription quality
- [earnings22](https://huggingface.co/datasets/argmaxinc/earnings22): ~120 hours of English audio clips from earnings calls with various accents, tests long-form transcription quality

### Reproducing Results
Results in this page are generated by our cluster of Apple Silicon Macs. We use them as self-hosted runners on
Github Actions as our CI infrastructure. Due to [security concerns](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions#hardening-for-self-hosted-runners),
we are unable to open up the cluster to the public. However, any Apple Silicon Mac (even with 8GB RAM) can be used to
run identical [evaluation jobs](#evaluation) locally. For reference, our M2 Ultra devices complete a `librispeech` + `openai/whisper-large-v3`
evaluation in under 1 hour regardless of the Whisper implementation. Older Apple Silicon Macs should take less than 1 day to complete the same evaluation.



### Glossary

- `_turbo`: Indicates the presence of additional optimizations (not compression) to unlock streaming transcription
as described in our [Blog Post](https://www.takeargmax.com/blog/whisperkit).

- `_*MB`: Indicates the presence of model compression. Instead of cluttering the filename with details like
`_AudioEncoder-5.8bits_TextDecoder-6.1bits_QLoRA-rank=16`, we choose to summarize the compression spec as the
resulting total file size since this is what matters to developers in production.

