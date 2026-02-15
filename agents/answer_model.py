# Qwen3-4B in action.
import time
import os

os.environ["HF_HUB_CACHE"] = "/root/.cache/huggingface"


import torch
from typing import Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from .reward import a_agent_reward_fn
from trl import GRPOTrainer, GRPOConfig
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq
import json

def load_mcq_file(path):
    with open(path, "r") as f:
        data = json.load(f)

    dataset = [{"prompt": json.dumps(item)} for item in data]
    return dataset

def load_mcq_sft(path):
    with open(path, "r") as f:
        data = json.load(f)

    dataset = []
    for item in data:
        text = json.dumps(item) + "\nAnswer: " + item["answer"]
        dataset.append({"text": text})

    return dataset


torch.random.manual_seed(0)


class AAgent(object):
    def __init__(self, **kwargs):
        model_name = "Qwen/Qwen2.5-14B-Instruct"

        # load the tokenizer and the model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )

    def generate_response(
        self, message: str | List[str], system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."
        if isinstance(message, str):
            message = [message]
        # Prepare all messages for batch processing
        all_messages = []
        for msg in message:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg},
            ]
            all_messages.append(messages)

        # convert all messages to text format
        texts = []
        for messages in all_messages:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            texts.append(text)

        # tokenize all texts together with padding
        model_inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.model.device)

        tgps_show_var = kwargs.get("tgps_show", False)
        # conduct batch text completion
        if tgps_show_var:
            start_time = time.time()
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=kwargs.get("max_new_tokens", 1024),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if tgps_show_var:
            generation_time = time.time() - start_time

        # decode the batch
        batch_outs = []
        if tgps_show_var:
            token_len = 0
        for i, (input_ids, generated_sequence) in enumerate(
            zip(model_inputs.input_ids, generated_ids)
        ):
            # extract only the newly generated tokens
            output_ids = generated_sequence[len(input_ids) :].tolist()

            # compute total tokens generated
            if tgps_show_var:
                token_len += len(output_ids)

            # remove thinking content using regex
            # result = re.sub(r'<think>[\s\S]*?</think>', '', full_result, flags=re.DOTALL).strip()
            index = (
                len(output_ids) - output_ids[::-1].index(151668)
                if 151668 in output_ids
                else 0
            )

            # decode the full result
            content = self.tokenizer.decode(
                output_ids[index:], skip_special_tokens=True
            ).strip("\n")
            batch_outs.append(content)
        if tgps_show_var:
            return (
                batch_outs[0] if len(batch_outs) == 1 else batch_outs,
                token_len,
                generation_time,
            )
        return batch_outs[0] if len(batch_outs) == 1 else batch_outs, None, None


if __name__ == "__main__":
    # Single message (backward compatible)
    ans_agent = AAgent()
    
    # ------- GRPO ----------
    # config = GRPOConfig(
    #     output_dir="./aagent-grpo",
    #     learning_rate=5e-6,
    # )

    # config = GRPOConfig(
    #     temperature=0.8,
    #     learning_rate=5e-6,
    #     weight_decay=0.001,
    #     warmup_ratio=0.1,
    #     lr_scheduler_type="linear",
    #     optim="adamw_8bit",
    #     logging_steps=1,
        
    #     per_device_train_batch_size=4,  
    #     gradient_accumulation_steps=1,
    #     num_generations=4,             
        
    #     max_steps=100,
    #     save_steps=100,
    #     report_to="none",
    #     output_dir="./aagent-grpo",
    # )


    # trainer = GRPOTrainer(
    #     model=ans_agent.model,
    #     processing_class=ans_agent.tokenizer,
    #     args=config,
    #     reward_funcs=[a_agent_reward_fn],
    # )

    mcq_files = [
        "data_part_1.json",
        "data_part_2.json",
        "data_part_3.json",
        "data_part_4.json",
        "data_part_5.json",
    ]

    #yet to define mcq_dataset
    # trainer.train(prompts=mcq_dataset)
    # for i, file in enumerate(mcq_files, 1):
    #     print(f"\n=== Training on {file} ({i}/5) ===")
    
    #     prompts = load_mcq_file(file)
    
    #     trainer = GRPOTrainer(
    #         model=ans_agent.model,
    #         processing_class=ans_agent.tokenizer,
    #         args=config,
    #         reward_funcs=[a_agent_reward_fn],
    #         train_dataset=prompts,   # ⭐ HERE
    #     )
    
    #     trainer.train()
    
    #     trainer.save_model(f"./aagent-train/aagent-grpo-stage{i}")

    # SFT
    if ans_agent.tokenizer.pad_token is None:
        ans_agent.tokenizer.pad_token = ans_agent.tokenizer.eos_token
        ans_agent.tokenizer.pad_token_id = ans_agent.tokenizer.eos_token_id
    
    # Setup trainer with ROCm-friendly settings and proper data handling
    for i, file in enumerate(mcq_files, 1):
        print(f"\n=== Training on {file} ({i}/5) ===")
    
        prompts = load_mcq_sft(file)
        trainer = SFTTrainer(
            model=ans_agent.model,
            tokenizer=ans_agent.tokenizer,
            train_dataset=prompts,
            dataset_text_field="text",
            max_seq_length=1024,
            data_collator=DataCollatorForSeq2Seq(tokenizer=ans_agent.tokenizer, padding=True),
            packing=False,
            args=SFTConfig(
                per_device_train_batch_size=32,  # 🚀 MI300X can handle this with 192GB HBM3!
                gradient_accumulation_steps=2,   # Effective batch size = 32*2 = 64
                warmup_steps=5,
                num_train_epochs=1,
                learning_rate=1e-4,
                logging_steps=1,
                optim="adamw_8bit",  # Pure torch optimizer
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="logical_reasoning_rocm_outputs",
                report_to="none",
                bf16=True,
                dataloader_pin_memory=False,
                remove_unused_columns=True,  # Remove unused columns to avoid tensor issues
                gradient_checkpointing=True,
                dataloader_num_workers=0,  # Single worker for ROCm stability
            ),
        )
        
        # Train only on responses
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
        
        FastLanguageModel.for_training(ans_agent.model)
        trainer_stats = trainer.train()
        
        
        # trainer_stats = trainer.train()
    
    # ------- DEFAULT CODE BELOW FOR TESTING -----------
    #Default test code
    """
    response, tl, gt = ans_agent.generate_response(
        "Solve: 2x + 5 = 15",
        system_prompt="You are a math tutor.",
        tgps_show=True,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
    )
    print(f"Single response: {response}")
    print(
        f"Token length: {tl}, Generation time: {gt:.2f} seconds, Tokens per second: {tl/gt:.2f}"
    )
    print("-----------------------------------------------------------")

    # Batch processing (new capability)
    messages = [
        "What is the capital of France?",
        "Explain the theory of relativity.",
        "What are the main differences between Python and Java?",
        "What is the significance of the Turing Test in AI?",
        "What is the capital of Japan?",
    ]
    responses, tl, gt = ans_agent.generate_response(
        messages,
        max_new_tokens=512,
        temperature=0.1,
        top_p=0.9,
        do_sample=True,
        tgps_show=True,
    )
    print("Responses:")
    for i, resp in enumerate(responses):
        print(f"Message {i+1}: {resp}")
    print(
        f"Token length: {tl}, Generation time: {gt:.2f} seconds, Tokens per second: {tl/gt:.2f}"
    )
    print("-----------------------------------------------------------")

    # Custom parameters
    response = ans_agent.generate_response(
        "Write a story", temperature=0.8, max_new_tokens=512
    )
    print(f"Custom response: {response}")
"""