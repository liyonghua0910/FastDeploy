"""
# Copyright (c) 2025  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import argparse
import json
import os
import random
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas
import requests

################################################################################################
# Utils
################################################################################################
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"


def log(s="", level="INFO", **kwargs):
    prefix = time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()) + f" [{level}]"
    s = prefix + " " + s
    if level == "ERROR":
        s = RED + BOLD + s + RESET
    elif level == "WARN":
        s = YELLOW + s + RESET
    elif level == "DEBUG":
        s = GREEN + s + RESET
    print(s, **kwargs)


def error(*args, **kwargs):
    kwargs.update({"level": "ERROR"})
    log(*args, **kwargs)


def warn(*args, **kwargs):
    kwargs.update({"level": "WARN"})
    log(*args, **kwargs)


def debug(*args, **kwargs):
    if not os.environ.get("FD_DEBUG") == "1":
        return
    kwargs.update({"level": "DEBUG"})
    log(*args, **kwargs)


################################################################################################
# Chat Template Parser
################################################################################################


class QwenChatTemplateParser:
    """template parser for qwen series model"""

    def __init__(self, disable_thinking=False, tokenizer=None):
        """initialize the tokens for templated final prompt"""
        self.disable_thinking = disable_thinking
        self.eot_token = "<|im_end|>\n"
        self.eos = "<|im_end|>\n"
        self.system_token = "<|im_start|>system\n"
        self.user_token = "<|im_start|>user\n"
        self.assistant_token = "<|im_start|>assistant\n"
        if disable_thinking:
            self.assistant_token += "<think>\n\n</think>\n\n"
        self.generation_prompt = self.assistant_token

        self.tool_start_token = "\n<tool_call>\n"
        self.tool_end_token = "\n</tool_call>"

        self.tool_response_start_token = "<tool_response>\n"
        self.tool_response_end_token = "\n</tool_response>"

    def parse(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt=False,
        is_first_msg=False,
        **kwargs,
    ) -> str:
        """concat the tempalate for qwen series model"""
        result = ""

        # if the first message is not a system message, add the system message
        if is_first_msg and messages[0]["role"] != "system":
            result += (
                self.system_token
                + "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
                + self.eot_token
            )

        for message in messages:
            if message["role"] == "system":
                result += self.parse_system(message)
            elif message["role"] == "user":
                result += self.parse_user(message)
            elif message["role"] == "assistant":
                result += self.parse_assistant(message)
            elif message["role"] == "tool":
                result += self.parse_tool(message)
            else:
                raise NotImplementedError(f"Unsupported message role: {message['role']}")

        if add_generation_prompt:
            result += self.generation_prompt
        return result

    def parse_system(self, message):
        """for system"""
        return self.system_token + message["content"] + self.eot_token

    def parse_user(self, message):
        """for user"""
        return self.user_token + message["content"] + self.eot_token

    def parse_assistant(self, message):
        """for assistant"""
        result = self.assistant_token + message["content"] + self.eot_token
        return result

    def parse_tool(self, message):
        """for tool"""
        return (
            self.user_token
            + self.tool_response_start_token
            + message["content"]
            + self.tool_response_end_token
            + self.eot_token
        )


################################################################################################
# Datasets and Tasks
################################################################################################


class AgenticRLDataset:
    """
    Dataset for Agentic RL in JSONL format.

    Each line in the file is a complete multi-turn chat session containing
    `system`, `user`, and `assistant` messages. This class loads multiple
    JSONL shards and exposes them as an iterable/indexable collection.
    """

    def __init__(self, dataset_path, max_samples=None, shuffle=False) -> None:
        self.dataset = []
        for i in range(1, 6):
            with open(f"{dataset_path}/{i}.jsonl", "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.dataset.extend([json.loads(line) for line in lines])
        if max_samples is not None and len(self.dataset) > max_samples:
            self.dataset = self.dataset[:max_samples]
        if shuffle:
            random.shuffle(self.dataset)

    def __iter__(self):
        return iter(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class AgenticRLTask:
    """
    Represents a single multi-turn dialogue thread for Agentic RL.

    Starting from one multi-round sample in the dataset, the task
    incrementally builds the round context (1 → 2 → 3, ...), sends a
    request to the server for each round, and stores the model outputs
    in `self.result`. The context grows round by round, for example:

        Round 1: [system prompt, user prompt] -> [assistant response]
        Round 2: [system prompt, user prompt, assistant response, user prompt] -> [assistant response]
        ...

    Inputs come from the ground-truth conversation context in the sample;
    outputs are produced by the model. We strictly cap response length to
    facilitate aligned and reproducible performance testing.
    """

    def __init__(
        self,
        task_id=None,
        model=None,
        tokenizer=None,
        template_parser=None,
        server_ip=None,
        server_port=None,
        data=None,
    ) -> None:
        self.task_id = task_id
        self.model = model
        self.tokenizer = tokenizer
        self.template_parser = template_parser
        self.server_ip = server_ip
        self.server_port = server_port
        self.data = data
        self.result = {}

    def build_url_and_payload(self, ip, port, messages=None, prompt=None, min_tokens=None, max_tokens=None):
        assert (messages is None) ^ (prompt is None), "messages and prompt cannot be both specified or neither"
        payload = {
            "model": self.model,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        }
        if messages is not None:
            url = f"http://{ip}:{port}/v1/chat/completions"
            payload["messages"] = messages
        else:
            url = f"http://{ip}:{port}/v1/completions"
            payload["prompt"] = prompt
        if min_tokens is not None:
            payload["min_tokens"] = 100  # min_tokens
        if max_tokens is not None:
            payload["max_tokens"] = 100  # max_tokens
        return url, payload

    def multi_round_messages_generator(self):
        messages = self.data["chat_completions"]
        round_id = 0
        round_msgs = []
        round_out_lens = []
        for msg_id, msg in enumerate(messages):
            round_msgs.append(msg)
            if msg["role"] == "user":
                round_id += 1
                if msg_id + 1 < len(messages) and messages[msg_id + 1]["role"] == "assistant":
                    round_exp_out = messages[msg_id + 1]["content"]
                    round_exp_out_len = len(self.tokenizer(round_exp_out).input_ids)
                    round_out_lens.append(round_exp_out_len)
                else:
                    round_exp_out_len = int(sum(round_out_lens) / len(round_out_lens))
                yield (round_id, round_msgs, round_exp_out_len)
                break

    def execute(self):

        task_start_time = time.perf_counter()
        log(f"Task {self.task_id} starts, iterating over all rounds.")

        for round_id, round_msgs, round_exp_out_len in self.multi_round_messages_generator():
            if self.template_parser is not None:
                prompt = self.template_parser.parse(round_msgs, add_generation_prompt=True)
                url, payload = self.build_url_and_payload(
                    self.server_ip,
                    self.server_port,
                    prompt=prompt,
                    min_tokens=round_exp_out_len,
                    max_tokens=round_exp_out_len,
                )
            else:
                url, payload = self.build_url_and_payload(
                    self.server_ip,
                    self.server_port,
                    messages=round_msgs,
                    min_tokens=round_exp_out_len,
                    max_tokens=round_exp_out_len,
                )

            try:
                log(f"Task {self.task_id} round {round_id} starts, url: {url} payload: {payload}")
                round_start_time = round_last_token_time = time.perf_counter()
                round_ttft = 0
                round_tpots = []
                round_output = ""
                round_input_tokens = 0
                round_output_tokens = 0
                round_cached_tokens = 0

                with requests.post(url=url, json=payload, timeout=1 * 24 * 60 * 60, stream=True) as response:
                    response.raise_for_status()
                    chunk_id = 0
                    for line in response.iter_lines(chunk_size=1, decode_unicode=True):
                        if not line:
                            continue
                        elif line.startswith("data: "):
                            data_str = line[len("data: ") :].strip()
                        else:
                            data_str = line.strip()

                        # "data: {...}" or "data: [DONE]"
                        if data_str == "[DONE]":
                            break
                        else:
                            chunk_id += 1
                            round_this_token_time = time.perf_counter()
                            obj = json.loads(data_str)
                            choices = obj.get("choices", [])
                            if len(choices) > 0:
                                if chunk_id == 1:
                                    round_ttft = round_this_token_time - round_last_token_time
                                else:
                                    round_tpots.append(round_this_token_time - round_last_token_time)
                                round_last_token_time = round_this_token_time
                                if "chat" in url:
                                    round_output += choices[0].get("delta").get("content")
                                else:
                                    round_output += choices[0].get("text")
                            else:  # last usage chunk
                                round_cached_tokens = (
                                    obj.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
                                )
                                round_input_tokens = obj.get("usage", {}).get("prompt_tokens", 0)
                                round_output_tokens = obj.get("usage", {}).get("completion_tokens", 0)

                self.result[f"task_{self.task_id}_round_{round_id}"] = {
                    "task_id": self.task_id,
                    "round_id": round_id,
                    "success": True,
                    "input_tokens": round_input_tokens,
                    "output_tokens": round_output_tokens,
                    "cached_tokens": round_cached_tokens,
                    "time_to_first_token": round_ttft,
                    "time_per_output_token": (
                        sum(round_tpots) / len(round_tpots) if len(round_tpots) > 0 else float("nan")
                    ),
                    "end_to_end_latency": round_last_token_time - round_start_time,
                }
                log(
                    f"Task {self.task_id} round {round_id} finished, "
                    f"output: {repr(round_output) if len(round_output) < 1024 else repr(round_output[:1024//2] + '.........' + repr(round_output[-1024//2:]))}"
                )
                log(f"Task {self.task_id} round {round_id} result: {self.result}")

            except Exception as e:
                self.result[f"task_{self.task_id}_round_{round_id}"] = {
                    "task_id": self.task_id,
                    "round_id": round_id,
                    "success": False,
                    "error": traceback.format_exc(e),
                }
                error(f"Task {self.task_id} round {round_id} failed, result: {self.result}")
                break

        task_finish_time = time.perf_counter()
        log(f"Task {self.task_id} finished, cost time: {task_finish_time-task_start_time:.4f} s.")
        return self.result

    def __call__(self):
        return self.execute()


def get_chat_template_parser(path):
    if path == "qwen":
        return QwenChatTemplateParser()
    else:
        return None


def get_tasks(args):
    chat_template_parser = get_chat_template_parser(args.chat_template_parser)
    dataset = AgenticRLDataset(args.dataset, max_samples=args.num_tasks, shuffle=True)
    tasks = []
    for i in range(args.num_tasks):
        from paddleformers.transformers import AutoTokenizer

        task = AgenticRLTask(
            model=args.model,
            tokenizer=AutoTokenizer.from_pretrained(args.tokenizer),
            template_parser=chat_template_parser,
            server_ip=args.server_ip,
            server_port=args.server_port,
            data=dataset[i],
            task_id=i + 1,
        )
        tasks.append(task)
    return tasks


################################################################################################
# Result Postprocessing and Display
################################################################################################


def summarize(results: dict):
    summary = results.pop("summary")
    details = pandas.DataFrame(results.values())

    output = []
    output.append("-" * 80)
    output.extend(
        [
            ("Benchmark Duration", summary["duration"]),
            ("Number of workers", summary["num_workers"]),
            ("Number of requests", summary["num_requests"]),
            ("Number of tasks", summary["num_tasks"]),
            ("Request success ratio", details["success"].sum() / summary["num_requests"]),
        ]
    )
    output.append("-" * 80)
    output.extend(
        [
            ("Request per second (QPS)", summary["num_requests"] / summary["duration"]),
            ("Output token per second (OTPS)", details["output_tokens"].sum() / summary["duration"]),
            (
                "Token per second (TPS)",
                (details["input_tokens"].sum() + details["output_tokens"].sum()) / summary["duration"],
            ),
        ]
    )
    output.append("-" * 80)
    output.extend(
        [
            ("Input tokens (sum)", details["input_tokens"].sum()),
            ("Input tokens (mean)", details["input_tokens"].mean()),
            ("Output tokens (sum)", details["output_tokens"].sum()),
            ("Output tokens (mean)", details["output_tokens"].mean()),
            ("Cached tokens (sum)", details["cached_tokens"].sum()),
            ("Cached tokens (mean)", details["cached_tokens"].mean()),
            ("Cache hit ratio (mean)", (details["cached_tokens"] / details["input_tokens"]).mean()),
            ("Cache hit ratio (p50)", (details["cached_tokens"] / details["input_tokens"]).median()),
            ("Cache hit ratio (p90)", (details["cached_tokens"] / details["input_tokens"]).quantile(0.9)),
            ("Cache hit ratio (p95)", (details["cached_tokens"] / details["input_tokens"]).quantile(0.95)),
            ("Cache hit ratio (p99)", (details["cached_tokens"] / details["input_tokens"]).quantile(0.99)),
        ]
    )
    output.append("-" * 80)
    output.extend(
        [
            ("Time-to-first-token (mean)", details["time_to_first_token"].mean()),
            ("Time-to-first-token (p50)", details["time_to_first_token"].median()),
            ("Time-to-first-token (p90)", details["time_to_first_token"].quantile(0.9)),
            ("Time-to-first-token (p95)", details["time_to_first_token"].quantile(0.95)),
            ("Time-to-first-token (p99)", details["time_to_first_token"].quantile(0.99)),
            ("Time-per-output-token (mean)", details["time_per_output_token"].mean()),
            ("Time-per-output-token (p50)", details["time_per_output_token"].median()),
            ("Time-per-output-token (p90)", details["time_per_output_token"].quantile(0.9)),
            ("Time-per-output-token (p95)", details["time_per_output_token"].quantile(0.95)),
            ("Time-per-output-token (p99)", details["time_per_output_token"].quantile(0.99)),
            ("End-to-end latency (mean)", details["end_to_end_latency"].mean()),
            ("End-to-end latency (p50)", details["end_to_end_latency"].median()),
            ("End-to-end latency (p90)", details["end_to_end_latency"].quantile(0.9)),
            ("End-to-end latency (p95)", details["end_to_end_latency"].quantile(0.95)),
            ("End-to-end latency (p99)", details["end_to_end_latency"].quantile(0.99)),
        ]
    )
    output.append("-" * 80)

    output_str = ""
    w = max([len(o[0]) if isinstance(o, tuple) else 0 for o in output]) + 1
    for o in output:
        if isinstance(o, tuple):
            k, v = o
            if isinstance(v, float):
                v = f"{v:.4f}"
            output_str += f"{k:{w}s}: {v}"
        elif isinstance(o, str):
            output_str += o
        output_str += "\n"
    return output_str


def main():
    args = parse_args()
    tasks = get_tasks(args)
    start_time = time.perf_counter()
    results = {}
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for task in tasks:
            futures.append(executor.submit(task))
        for completed_task in as_completed(futures):
            results.update(completed_task.result())
    finish_time = time.perf_counter()
    log(f"All tasks finished, cost time: {finish_time - start_time}")
    results.update(
        {
            "summary": {
                "duration": finish_time - start_time,
                "num_requests": len(results),
                "num_tasks": args.num_tasks,
                "num_workers": args.num_workers,
            }
        }
    )
    result_path = os.path.join(args.result_dir, f"results.{args.id}.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
        log(f"Results are saved to {result_path}")
    log(f"Benchmark Summary ({args.id}): \n" + summarize(results))
    return


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=str, default=time.strftime("%Y%m%d_%H%M%S", time.localtime()))
    parser.add_argument("--model", type=str)
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--chat_template_parser", "--chat-template-parser", type=str, choices=["qwen"], default=None)
    parser.add_argument("--num_workers", "--num-workers", type=int, default=64)
    parser.add_argument("--num_tasks", "--num-tasks", type=int, default=64)
    parser.add_argument("--server_ip", "--server-ip", type=str, default="0.0.0.0")
    parser.add_argument("--server_port", "--server-port", type=str, default="8580")
    parser.add_argument(
        "--server_type", "--server-type", type=str, choices=["fd", "vllm"], default="fd"
    )  # TODO: support vllm
    parser.add_argument("--result_dir", "--result-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2025)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.result_dir is None:
        args.result_dir = f"./{args.id}"
    os.makedirs(args.result_dir, exist_ok=True)

    log("###### Benchmark Args ######")
    w = max(map(len, list(vars(args).keys())))
    for k, v in vars(args).items():
        log(f"{k:{w}} = {v}")
    log("############################")

    return args


if __name__ == "__main__":
    main()
