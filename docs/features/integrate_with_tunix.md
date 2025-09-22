## Motivation

Integrate with tunix to make contribution to post-training.

## Solution

Sglang-jax will implement a SglangJaxRollout in tunix to generate completions.

### SglangJaxRollout

```python3
class BaseRollout(abc.ABC):
  """Base RolloutWorker."""

  @abc.abstractmethod
  def generate(
      self,
      prompts: list[str],
      rollout_config: RolloutConfig,
      **kwargs,
  ) -> RolloutOutput:
    """Generates samples from the model."""

  @abc.abstractmethod
  def get_per_token_logps(
      self,
      prompt_tokens: jax.Array,
      completion_tokens: jax.Array,
  ) -> jax.Array:
    """Returns per-token log probabilities from the model."""

  @abc.abstractmethod
  def update_params(
      self,
      params: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    """Updates the rollout model parameters."""

  @abc.abstractmethod
  def pad_id(self) -> int:
    """Returns the pad id."""

  @abc.abstractmethod
  def eos_id(self) -> int:
    """Returns the eos id."""

  @abc.abstractmethod
  def model(self) -> Any:
    """Returns the rollout model."""
```

### Work in Sglang-jax

1. Use jit to wrap `return_logprob` logic to improve performance and test accuracy.
2. Support `generate()` interface for batch prompts in a request, like `LLM.generate()`.
3. Ensure the outputs with `samplingParams` are exptected by tunix.

#### Wrap `return_logprob` logic

TODO: add code modification.

#### `generate()` API

Note: ensure the output contains all information tunix need.

```python
def generate(
    self,
    prompt: Optional[Union[List[str], str]] = None,
    sampling_params: Optional[Union[List[Dict], Dict]] = None,
    # The token ids for text; one can either specify text or input_ids.
    input_ids: Optional[Union[List[List[int]], List[int]]] = None,
    return_logprob: Optional[Union[List[bool], bool]] = False,
    logprob_start_len: Optional[Union[List[int], int]] = None,
    top_logprobs_num: Optional[Union[List[int], int]] = None,
    token_ids_logprob: Optional[Union[List[List[int]], List[int]]] = None,
    stream: bool = False,
) -> Union[Dict, Iterator[Dict]]:
    pass
```

- use example

```python
from sgl_jax.srt.entrypoints.engine import Engine
if __name__ == '__main__':
    engine=Engine(model_path= 'Qwen/Qwen-7B-Chat', trust_remote_code=True, dist_init_addr='0.0.0.0:10011', nnodes=1 , tp_size=4, device='tpu' ,random_seed=3, node_rank=0, mem_fraction_static=0.4, chunked_prefill_size=8192, download_dir='/tmp', dtype='bfloat16', precompile_bs_paddings = [64], max_running_requests = 64, skip_server_warmup=True, attention_backend='fa',precompile_token_paddings=[8192], page_size=64 ,log_requests=True, log_requests_level=3)
    output=engine.generate(prompt=['您好', "hello"], sampling_params={"n",2, "temperature": 0.7}, return_logprob=True)
    print(len(list(output)), output)
```

- add_default_sampling_params

- fix add for sample

  ```python
  return_logprob: bool,
  top_logprobs_nums: List[int],
  token_ids_logprobs: List[List[int]],
  ```

TODO: add definition.

#### `vllm Sample` vs `sglang_jax Sample`

TODO: add field comparison and align the meaning of every field.

The exmaple of logprobs of output_ids:
```bash
# input_ids = [0,1,2], vocab_size = 5, output_ids = [1,4], sampling algorithm = greedy

# prefill logits, the first generated token is 1
[
  [0.1, 0.2, 0.3, 0.5, -0.1],
  [0.1, 0.2, 0.3, 0.5, -0.1],
  [0.1, *0.5*, 0.3, 0.1, -0.1],
]

# decode logits, token is 4
[
  [0.1, *0.4*, 0.1, 0.2, 0.6],
]

# expected output logprobs:
[0.5, 0.4]
```


## Discussion

D1. Is multi-sampling is required in the future?

D2. Is beam search is required in the future?

D3: For `generate`, is it enough to only have the `logprob` corresponding to each output_id? Are prompt_ids logprobs necessary?

## Test

1. Test whether `return_logprob` result is expected
2. Test `generate()` interface
3. Test e2e result compared with VllmRollout.
