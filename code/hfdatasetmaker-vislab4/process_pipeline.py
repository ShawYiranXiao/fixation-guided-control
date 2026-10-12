# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Shaw Yiran Xiao and Weihao Qiu
def process_dataset(dataset, processors, batch_size=5000, load_cache=True):
    #  Process the entire dataset with map
    for processor in processors:
        print(f"processor: {processor['processor_function'].__name__}")
        dataset = dataset.map(
            lambda batch: processor["processor_function"](batch),
            batched=True,
            batch_size=batch_size,
            num_proc=processor["num_proc"],
            load_from_cache_file=load_cache,
        )
    return dataset
