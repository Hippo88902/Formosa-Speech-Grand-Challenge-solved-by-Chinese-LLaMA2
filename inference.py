# 以下範例為輸入json用以批次生成的腳本，並將結果存於 ./answer.csv
# 你也可以自行修改腳本，整理成自己習慣的資料格式(txt..)用於批次輸入
import torch
import os
import argparse
import json,csv
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    StoppingCriteria,
    BitsAndBytesConfig
)

DEFAULT_SYSTEM_PROMPT = """請認真回答題目"""

TEMPLATE_WITH_SYSTEM_PROMPT = (
    "[INST] <<SYS>>\n"
    "{system_prompt}\n"
    "<</SYS>>\n\n{instruction} [/INST]"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--base_model',
    default=None,
    type=str,
    required=True,
    help='Base model path')
parser.add_argument(
    '--gpus',
    default="0",
    type=str,
    help='If None, cuda:0 will be used. Inference using multi-cards: --gpus=0,1,... ')
parser.add_argument(
    '--load_in_8bit',
    action='store_true',
    help='Use 8 bit quantized model')
parser.add_argument(
    '--load_in_4bit',
    action='store_true',
    help='Use 4 bit quantized model')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = LlamaForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map='auto',
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bnb_4bit_compute_dtype=torch.float16
    ))

tokenizer = LlamaTokenizer.from_pretrained(args.base_model)

with open('datasets/AI1000.json', 'r') as file:
    json_data = json.load(file)
    with open('answer.csv', 'w', newline='', encoding='utf-8') as csv_file:
        writer=csv.writer(csv_file)
        writer.writerow(['ID', 'Answer'])
        for row in json_data:
            id = row['id']
            instruction = row['input'] + ':\n' + row['instruction']

            prompt = TEMPLATE_WITH_SYSTEM_PROMPT.format_map({'instruction': instruction,'system_prompt': DEFAULT_SYSTEM_PROMPT})
            inputs = tokenizer.encode(prompt+'\n', return_tensors="pt").to(DEV)

            generate_kwargs = dict(
                input_ids=inputs,
                temperature=0.2,
                top_p=0.9,
                top_k=40,
                do_sample=True,
                max_new_tokens=1, #為了回答選擇題而設定1
                repetition_penalty=1.1,
                guidance_scale=1.0
            )
            outputs = model.generate(**generate_kwargs)
            result = tokenizer.decode(outputs[0])
            print(result)
            response = result.split('[/INST]\n')[-1]
            writer.writerow([id, response[0]])