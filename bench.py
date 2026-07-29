from evalscope import TaskConfig, run_task

task_cfg = TaskConfig(
    model="Qwen/Qwen2.5-VL-32B-Instruct",
    api_url="http://127.0.0.1:30000/v1",
    api_key="EMPTY",
    eval_type="openai_api",
    dataset_hub="huggingface",
    datasets=[
        "mmmu",
        "mmmu_pro",
        "docvqa",
        "chartqa",
    ],
    dataset_args={
        "mmmu": {
            "dataset_id": "MMMU/MMMU",
            "eval_split": "validation",
        },
        "mmmu_pro": {
            "dataset_id": "MMMU/MMMU_Pro",
            "eval_split": "test",
            "extra_params": {
                "dataset_format": "vision",
            },
        },
        "docvqa": {
            "dataset_id": "lmms-lab/DocVQA",
            "eval_split": "test",
        },
        "chartqa": {
            "dataset_id": "lmms-lab/ChartQA",
            "eval_split": "test",
            "subset_list": [
                "human_test",
                "augmented_test",
            ],
        },
    },
    eval_batch_size=32,
    generation_config={
        "max_tokens": 4096,
        "temperature": 0.0,
        "n": 1,
    },
)

run_task(task_cfg=task_cfg)
