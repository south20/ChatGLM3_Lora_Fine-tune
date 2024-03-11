#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import typer
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

ModelType = PreTrainedModel | PeftModelForCausalLM  # 使用 Python 3.10+ 的联合类型
TokenizerType = PreTrainedTokenizer | PreTrainedTokenizerFast

app = typer.Typer(pretty_exceptions_show_locals=False)

def load_model_and_tokenizer(model_dir: Path) -> tuple[ModelType, TokenizerType]:
    model_dir = model_dir.expanduser().resolve()
    if (model_dir / 'adapter_config.json').exists():
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map='auto'
        )
        tokenizer_dir = model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
    return model, tokenizer

@app.command()
def main(
        model_dir: Path = typer.Argument(..., help='The directory of the model.'),
        out_dir: Path = typer.Option(..., help='The output directory where the model and tokenizer will be saved.'),
):
    try:
        model, tokenizer = load_model_and_tokenizer(model_dir)
        merged_model = model.merge_and_unload() 
        merged_model.save_pretrained(out_dir, safe_serialization=True)
        tokenizer.save_pretrained(out_dir)
    except Exception as e:
        typer.echo(f"Error: {str(e)}")

if __name__ == '__main__':
    app()
