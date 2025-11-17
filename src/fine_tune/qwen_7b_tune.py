"""
Fine-tune Qwen (e.g., Qwen/Qwen2.5-7B-Instruct) on
`junidude14/korean_roleplay_dataset_for_chat_game_2` using LoRA / QLoRA.

This script:
- Loads the HF dataset with fields: instruction, input, output
- Converts to a chat conversation using the model's chat template
- Trains with TRL's SFTTrainer and PEFT (LoRA/QLoRA)

Quickstart (GPU recommended):

  python -m src.fine_tune.qwen_7b_tune \
	--model_name Qwen/Qwen2.5-7B-Instruct \
	--output_dir ./outputs/qwen2.5-7b-roleplay-sft \
	--max_seq_length 2048 \
	--per_device_train_batch_size 1 \
	--gradient_accumulation_steps 16 \
	--num_train_epochs 2 \
	--lr 2e-4 \
	--use_qlora

Dependencies:
- transformers, datasets, accelerate, peft, trl, bitsandbytes
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

from datasets import load_dataset
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Fine-tune Qwen with LoRA on Korean roleplay dataset")
	parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model name or path")
	parser.add_argument(
		"--dataset_name",
		type=str,
		default="junidude14/korean_roleplay_dataset_for_chat_game_2",
		help="HF dataset path",
	)
	parser.add_argument("--output_dir", type=str, default="./outputs/qwen-roleplay-sft", help="Output dir")
	parser.add_argument("--max_seq_length", type=int, default=2048)
	parser.add_argument("--num_train_epochs", type=int, default=1)
	parser.add_argument("--per_device_train_batch_size", type=int, default=1)
	parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
	parser.add_argument("--lr", type=float, default=2e-4)
	parser.add_argument("--warmup_ratio", type=float, default=0.03)
	parser.add_argument("--weight_decay", type=float, default=0.0)
	parser.add_argument("--logging_steps", type=int, default=10)
	parser.add_argument("--save_steps", type=int, default=1000)
	parser.add_argument("--save_total_limit", type=int, default=3)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--bf16", action="store_true", help="Use bfloat16 if available")
	parser.add_argument("--fp16", action="store_true", help="Use float16 if bf16 not used")
	parser.add_argument("--use_lora", action="store_true", help="Enable LoRA finetuning (default if use_qlora is off)")
	parser.add_argument("--use_qlora", action="store_true", help="Enable QLoRA (4-bit)")
	parser.add_argument("--lora_r", type=int, default=16)
	parser.add_argument("--lora_alpha", type=int, default=32)
	parser.add_argument("--lora_dropout", type=float, default=0.05)
	parser.add_argument("--gradient_checkpointing", action="store_true")
	parser.add_argument("--packing", action="store_true", help="Pack multiple samples per sequence (SFTTrainer)")
	parser.add_argument("--push_to_hub", action="store_true")
	parser.add_argument("--hub_model_id", type=str, default=None)
	parser.add_argument("--hub_private_repo", action="store_true")
	return parser.parse_args()


def convert_to_messages(instruction: str, history: str, target: str) -> List[Dict[str, str]]:
	"""
	Convert dataset triple (instruction, input, output) into a chat messages list
	compatible with tokenizer.apply_chat_template for Qwen.

	Dataset format:
	- instruction: Korean system instruction string
	- input: conversation history where lines start with "USR:" or "NPC:" and are separated by \n
	We'll map:
	  - instruction -> system
	  - lines with USR: -> user
	  - lines with NPC: -> assistant
	  - final target -> assistant (the one we want the model to generate next)

	For SFT, we include the full history and append the target assistant turn.
	"""
	messages: List[Dict[str, str]] = []
	if instruction and instruction.strip():
		messages.append({"role": "system", "content": instruction.strip()})

	def add_turn(role: str, content: str):
		content = content.strip()
		if not content:
			return
		messages.append({"role": role, "content": content})

	if history:
		for raw in history.split("\n"):
			line = raw.strip()
			if not line:
				continue
			# Normalize prefixes, accept variants like "USR :", "NPC :"
			if line.lower().startswith("usr:"):
				add_turn("user", line.split(":", 1)[1].strip())
			elif line.lower().startswith("npc:"):
				add_turn("assistant", line.split(":", 1)[1].strip())
			else:
				# Default unknown lines as user messages to be safe
				add_turn("user", line)

	if target and target.strip():
		add_turn("assistant", target.strip())

	return messages


def build_formatting_func(tokenizer: AutoTokenizer):
	def format_sample(batch) -> List[str]:
		texts: List[str] = []
		for inst, inp, out in zip(batch["instruction"], batch["input"], batch["output"]):
			messages = convert_to_messages(inst, inp, out)
			# Convert to a single training string using the model's chat template
			text = tokenizer.apply_chat_template(
				messages,
				tokenize=False,
				add_generation_prompt=False,
			)
			texts.append(text)
		return texts

	return format_sample


def main():
	args = parse_args()

	os.makedirs(args.output_dir, exist_ok=True)

	# Load tokenizer first (for chat template)
	tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
	# Some Qwen tokenizers may not have a pad token; set to eos for SFT
	if tokenizer.pad_token_id is None:
		tokenizer.pad_token = tokenizer.eos_token

	# BitsAndBytes config for QLoRA
	bnb_config = None
	torch_dtype = None
	if args.use_qlora:
		bnb_config = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_use_double_quant=True,
			bnb_4bit_quant_type="nf4",
			bnb_4bit_compute_dtype="bfloat16" if args.bf16 else "float16",
		)

	# Load model
	model = AutoModelForCausalLM.from_pretrained(
		args.model_name,
		torch_dtype=(None if args.bf16 or args.fp16 else None),
		quantization_config=bnb_config,
		device_map="auto",
	)

	if args.gradient_checkpointing:
		model.gradient_checkpointing_enable()

	# LoRA config (works for both LoRA/QLoRA)
	peft_config = None
	if args.use_lora or args.use_qlora:
		peft_config = LoraConfig(
			r=args.lora_r,
			lora_alpha=args.lora_alpha,
			lora_dropout=args.lora_dropout,
			bias="none",
			task_type="CAUSAL_LM",
			target_modules=None,  # let PEFT pick defaults for Qwen
		)

	# Load dataset
	dataset = load_dataset(args.dataset_name)

	# Set up SFT config
	sft_config = SFTConfig(
		output_dir=args.output_dir,
		dataset_text_field=None,  # we pass formatting_func
		max_seq_length=args.max_seq_length,
		num_train_epochs=args.num_train_epochs,
		per_device_train_batch_size=args.per_device_train_batch_size,
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		learning_rate=args.lr,
		warmup_ratio=args.warmup_ratio,
		weight_decay=args.weight_decay,
		logging_steps=args.logging_steps,
		save_steps=args.save_steps,
		save_total_limit=args.save_total_limit,
		seed=args.seed,
		bf16=args.bf16,
		fp16=(args.fp16 and not args.bf16),
		gradient_checkpointing=args.gradient_checkpointing,
		packing=args.packing,
		push_to_hub=args.push_to_hub,
		hub_model_id=args.hub_model_id,
		hub_private_repo=args.hub_private_repo,
	)

	trainer = SFTTrainer(
		model=model,
		tokenizer=tokenizer,
		peft_config=peft_config,
		train_dataset=dataset["train"],
		args=sft_config,
		formatting_func=build_formatting_func(tokenizer),
	)

	trainer.train()
	trainer.save_model()
	tokenizer.save_pretrained(args.output_dir)

	# Optionally push to hub
	if args.push_to_hub:
		trainer.push_to_hub()


if __name__ == "__main__":
	main()

