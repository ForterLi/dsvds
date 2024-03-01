
---
pretty_name: "WhisperKit ASR Evaluation Results"
tags:
- whisper
- whisperkit
- coreml
- asr
- quantized
---
# WhisperKit Evaluation Results



## Dataset: `librispeech`

### Quality Evaluation 

|                                                                                                                                                                            |   WER |   QoI (%) |   File Size (MB) |
|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------:|----------:|-----------------:|
| [WhisperOpenAIAPI/openai_whisper-large-v2](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperOpenAIAPI/openai_whisper-large-v2/librispeech)               |  2.85 |     100   |             3100 |
| [WhisperKit/openai_whisper-large-v2](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-large-v2/librispeech)                           |  3.28 |      96.6 |             3100 |
| [WhisperKit/openai_whisper-large-v2_1050MB](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-large-v2_1050MB/librispeech)             |  3.32 |      95   |             1050 |
| [WhisperKit/openai_whisper-large-v2_turbo](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-large-v2_turbo/librispeech)               |  3.24 |      96.6 |             3100 |
| [WhisperKit/openai_whisper-large-v2_turbo_1022MB](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-large-v2_turbo_1022MB/librispeech) |  3.33 |      94.9 |             1022 |
| [WhisperKit/openai_whisper-small](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-small/librispeech)                                 |  3.98 |      82.9 |              483 |
| [WhisperKit/openai_whisper-base](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-base/librispeech)                                   |  6.11 |      67.1 |              145 |
| [WhisperKit/openai_whisper-tiny](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-tiny/librispeech)                                   |  8.94 |      52.4 |               66 |
| [WhisperKit/openai_whisper-large-v3](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-large-v3/librispeech)                           |  2.48 |      95.2 |             3100 |
| [WhisperKit/openai_whisper-large-v3_turbo](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-large-v3_turbo/librispeech)               |  2.44 |      95.4 |             3100 |
| [WhisperKit/openai_whisper-large-v3_turbo_1018MB](https://hf.co/datasets/argmaxinc/whisperkit-evals/tree/main/WhisperKit/openai_whisper-large-v3_turbo_1018MB/librispeech) |  2.49 |      94.8 |             1018 |


### Quality-of-Inference (QoI) Certification
We believe that rigorously measuring the quality of inference is necessary for developers and
enterprises to make informed decisions when opting to use optimized or compressed variants of
any machine learning model in production. For WhisperKit, we take the following implementations
and benchmark them using consistent evaluation harnesses:

- `WhisperOpenAIAPI`: [OpenAI's Whisper API](https://platform.openai.com/docs/guides/speech-to-text)($0.36/hour as of 02/29/24, 25MB max file size)
- `WhisperKit`: Argmax's Core ML implementation [[Eval Harness]](https://github.com/argmaxinc/whisperkittools/blob/main/whisperkit/pipelines.py#L100) [[Repo]](https://github.com/argmaxinc/WhisperKit)
- `whisper.cpp`: A C++ implementation form ggerganov [[Eval Harness]](https://github.com/argmaxinc/whisperkittools/blob/main/whisperkit/pipelines.py#L212) [[Repo]](https://github.com/ggerganov/whisper.cpp)
- `WhisperMLX`: A Python implementation from Apple MLX [[Eval Harness]](https://github.com/argmaxinc/whisperkittools/blob/main/whisperkit/pipelines.py#L338) [[Repo]](https://github.com/ml-explore/mlx-examples/blob/main/whisper/whisper/transcribe.py)

`WhisperOpenAIAPI` is the reference and we assume that it is using the equivalent of
[openai/whisper-large-v2](https://huggingface.co/openai/whisper-large-v2) in float16 precision.
In all measurements, we care primarily about per-example no-regressions (quantified as `qoi` below)
which is a stricter metric compared to dataset average WER. A 100% `qoi` preserves perfect
backwards-compatibility on the test distribution and avoids "perceived regressions", the phenomenon
where per-example known behavior changes after a code/model update and causes divergence in
downstream code or breaks the user experience itself (even if dataset averages might stay flat
across updates). Pseudocode for `qoi`:

```python
qoi = []
for example in dataset:
    no_regression = wer(optimized_model(example)) <= wer(reference_model(example))
    qoi.append(no_regression)
qoi = (sum(qoi) / len(qoi)) * 100.
```

We use `librispeech/test.clean` (~5 hours of short English audio clips) and `earnings22` (~120 hours of long English audio clips with various accents).
We anticipate developers that use Whisper (or similar models) in production to have their own Quality Assurance test sets and whisperkittools offers
the tooling necessary to run the same measurements on such custom test sets, please see the [Model Evaluation on Custom Dataset](#evaluate-on-custom-dataset) for details.

### Reproducing Results
Results in this page are generated by our cluster of Apple Silicon Macs. We use them as self-hosted runners on
Github Actions as our CI infrastructure. Due to [security concerns](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions#hardening-for-self-hosted-runners),
we are unable to open up the cluster to the public. However, any Apple Silicon Mac (even with 8GB RAM) can be used to
run identical [evaluation jobs](#evaluation) locally. For reference, our M2 Ultra devices complete a `librispeech` + `openai/whisper-large-v3`
evaluation in under 1 hour regardless of the Whisper implementation. Older Apple Silicon Macs should take less than 1 day to complete the same evaluation.



Glossary:

- `_turbo`: Indicates the presence of additional optimizations (not compression) to unlock streaming transcription
as described in our [Blog Post](https://www.takeargmax.com/blog/whisperkit).

- `_*MB`: Indicates the presence of model compression. Instead of cluttering the filename with details like
`_AudioEncoder-5.8bits_TextDecoder-6.1bits_QLoRA-rank=16`, we choose to summarize the compression spec as the
resulting total file size since this is what matters to developers in production.

