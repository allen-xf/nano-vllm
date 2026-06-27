import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, collect_metrics: bool = False, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = [] #用于多卡
        self.events = []  #用于多卡
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size): # 这里是子线程
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config, collect_metrics=collect_metrics)
        self.has_spec = config.draft_model is not None
        if self.has_spec:
            self.model_runner.set_block_manager(self.scheduler.block_manager)
        # functions to register and unregister cleanup functions.
        # These registered functions are called when the interpreter exits normally
        # Functions registered with atexit will not be called if the program terminates abnormally due to:
        # A fatal internal error in the Python interpreter.
        atexit.register(self.exit)

    def exit(self):
        if not hasattr(self, 'model_runner'):
            return
        atexit.unregister(self.exit)
        self.model_runner.call("exit")
        del self.model_runner
        import gc
        gc.collect()
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)  # 从词表中得到id
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self, verbose=False):
        prefill_seqs, decode_seqs = self.scheduler.schedule()
        self.scheduler.record_step(prefill_seqs, decode_seqs)
        if verbose and (prefill_seqs or decode_seqs):
            print(f"  Step: prefill={len(prefill_seqs)}, decode={len(decode_seqs)}")

        if self.has_spec:
            result = self.model_runner.call("run_speculative_step", prefill_seqs, decode_seqs)
            self.scheduler.postprocess_speculative_step(result)
            all_seqs = prefill_seqs + decode_seqs
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in all_seqs if seq.is_finished]
            num_prefill_tokens = sum(seq.scheduled_chunk_size for seq in prefill_seqs) if prefill_seqs else 0
            num_decode_tokens = sum(len(tokens) for tokens in result["decode_accepted_tokens"])
            if verbose and decode_seqs:
                accepted = [len(tokens) for tokens in result["decode_accepted_tokens"]]
                avg_accepted = num_decode_tokens / len(decode_seqs)
                print(f"  Spec: accepted={accepted}, avg={avg_accepted:.1f}")
            return outputs, num_prefill_tokens, num_decode_tokens
        else:
            token_ids = self.model_runner.call("run", prefill_seqs, decode_seqs)
            self.scheduler.postprocess(prefill_seqs, decode_seqs, token_ids)
            all_seqs = prefill_seqs + decode_seqs
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in all_seqs if seq.is_finished]
            num_prefill_tokens = sum(seq.scheduled_chunk_size for seq in prefill_seqs) if prefill_seqs else 0
            num_decode_tokens = len(decode_seqs)
            return outputs, num_prefill_tokens, num_decode_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        verbose: bool = False,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_prefill_tokens, num_decode_tokens = self.step(verbose=verbose)
            if use_tqdm:
                elapsed = perf_counter() - t
                if num_prefill_tokens > 0:
                    prefill_throughput = num_prefill_tokens / elapsed
                if num_decode_tokens > 0:
                    decode_throughput = num_decode_tokens / elapsed
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
