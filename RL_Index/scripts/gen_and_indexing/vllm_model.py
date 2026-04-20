from typing import Optional, Dict, Any
import vllm


class VllmModel:
    """
    Wrapper for vLLM LLM model with optional system prompt.
    """
    
    DEFAULT_SAMPLING_PARAMS = {
        'max_tokens': 500,
        'top_p': 0.8,
        'top_k': 20,
        'temperature': 0.7,
        'repetition_penalty': 1.05,
        'stop_token_ids': [151645, 151643],
    }

    def __init__(self, pretrain_path: str, system_prompt: str = "") -> None:
        self.model_name = pretrain_path
        self.system_prompt = system_prompt

        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.6,
            trust_remote_code=True,
            dtype="bfloat16",
            enforce_eager=False,
            max_model_len=16000,
            enable_lora=False
        )

        self.default_kwargs = self.DEFAULT_SAMPLING_PARAMS.copy()
        self.tokenizer = self.llm.get_tokenizer()
        self.tokenizer.padding_side = 'left'

    def predict(self, prompt: str, instruction: Optional[str] = None, lora_request: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate model output given a prompt and optional instruction.

        Args:
            prompt: Original doc text.
            instruction: Optional system-level instruction; defaults to self.system_prompt.
            lora_request: Optional LoRA adaptation request.

        Returns:
            Generated text.
        """
        instruction = instruction or self.system_prompt
        
        # truncate prompt if too long
        max_prompt_length = 16000 - self.default_kwargs['max_tokens'] - 10
        tokenized_prompt = self.tokenizer.tokenize(prompt)
        if len(tokenized_prompt) > max_prompt_length:
            tokenized_prompt = tokenized_prompt[:max_prompt_length]
            prompt = self.tokenizer.convert_tokens_to_string(tokenized_prompt)

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}
        ]

        sampling_params = vllm.SamplingParams(**self.default_kwargs)
        response = self.llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            use_tqdm=False,
            lora_request=lora_request
        )[0]

        output_text = response.outputs[0].text
        return output_text
    
    def predict_batch(self, prompt_list: list, instruction: Optional[str] = None, lora_request: Optional[Dict[str, Any]] = None) -> list:
        """
        Generate model outputs for a batch of prompts.

        Args:
            prompt_list: List of original doc texts.
            instruction: Optional system-level instruction; defaults to self.system_prompt.
            lora_request: Optional LoRA adaptation request.
            
        Returns:
            List of generated texts.
        """
        instruction = instruction or self.system_prompt
        
        message_list = []
        
        for i in range(len(prompt_list)):
            # truncate prompt if too long
            max_prompt_length = 16000 - self.default_kwargs['max_tokens'] - 10
            tokenized_prompt = self.tokenizer.tokenize(prompt_list[i])
            if len(tokenized_prompt) > max_prompt_length:
                tokenized_prompt = tokenized_prompt[:max_prompt_length]
                prompt_list[i] = self.tokenizer.convert_tokens_to_string(tokenized_prompt)
            messages = [
                {"role": "system", "content": instruction},
                {"role": "user", "content": prompt_list[i]}  # vLLM requires batch input, will handle below
            ]
            message_list.append(messages)

        prompt_texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False, # This prevents tokenization of the messages, keeping them as raw text.
                add_generation_prompt=True # This adds a prompt for the model to generate a response.
            )
            for messages in message_list
        ]
        sampling_params = vllm.SamplingParams(**self.default_kwargs)
        responses = self.llm.generate(
            prompt_texts,  # batch of one
            sampling_params=sampling_params,
            use_tqdm=True,
            lora_request=lora_request
        )

        texts = []
        for r in responses:
            texts.append(r.outputs[0].text if r.outputs else "")

        return texts

