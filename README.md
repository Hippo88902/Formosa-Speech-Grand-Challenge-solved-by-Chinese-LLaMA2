# Formosa-Speech-Grand-Challenge-solved-by-Chinese-LLaMA2

## 任務目標
科技部於2018至2020年舉辦了兩屆『科技大擂台，與AI對話』競賽，旨在探討人工智慧對全球產業、經濟和社會的影響。這些競賽特別強調語音應用在AI技術中的關鍵地位，尤其是語意理解技術的重要性。

在首屆競賽中，採用獎勵賽模式，鼓勵創新者利用創意和技術來解決語音AI的挑戰。當時主要使用的模型為BERT和GPT-2，而GPU則以V100/2080Ti系列為主。

在決賽的選擇題部分，參賽隊伍的最高正確率僅為53.7%。本研究旨在探究使用Large Language Models的方法，是否能夠在此類問題上取得更顯著的進步。

任務專注於文意理解，其中數據包括一段文章、問題及四個選項。

## 準備Dataset格式

首先，使用 `prepeocessing.py` 將格式轉換成訓練所需格式
- `pandas`庫來讀取存儲在`datasets/AI.xlsx`的Excel文件
- `Dataframe`進行數據處理
- 轉換成`JSON`格式並儲存


為了建立用於訓練的指令式數據集，我們提供了一個腳本 `script/training/build_dataset.py`，這個腳本將原始數據轉換為模型訓練所需的格式。以下是建立數據集的步驟和腳本使用方法的詳細說明。

### 腳本功能

腳本`build_dataset.py` 的主要功能包括：

- 格式化原始數據，將其轉換為模型可以理解的格式。
- 使用一個定義好的prompt模板，確保所有數據都有統一的格式。
- 對輸入和輸出進行編碼，以便模型可以進行訓練。

## Training (LoRa)
### 模型選擇 (可從huggingface下載)
- `chinese-llama-2-1.3b`
- `chinese-llama-2-7b`
- `chinese-llama-2-13b`
- `chinese-alpaca-2-1.3b`
- `chinese-alpaca-2-7b`
- `chinese-alpaca-2-13b`

### max_sequence
由於使用`llamatokenizer`進行編碼後，題目的長度可能會超過預設的512個token限制。為了處理這個問題，我們將`max_sequence`的長度增加到1536。這樣可以確保即使是較長的題目也能夠被完整地編碼和處理，從而提高模型對於長文本的處理能力。

## 硬體需求

根據所使用的模型，硬體需求會有所不同：

- 對於`chinese-alpaca-1.3b`和`chinese-alpaca-7b`模型，我們使用了配置為RTX 2070 Super * 2 + 64GB CPU內存的系統。
- 而對於`chinese-alpaca-13b`模型，則使用了搭載RTX 3090 + 126GB CPU內存的硬體配置。

這些配置確保了足夠的計算能力和內存空間，以支持這些大型語言模型的訓練和運行。

## deepspeed優化

由於顯存不足的挑戰，我們採用了`deepspeed`來優化訓練流程。這是一種先進的分布式訓練框架，可以顯著減少單機顯存需求並提高訓練速度。

### 配置調整

- 將`deepspeed`配置從`zero_2_no_offload`調整為`zero_3_offload`，以便於在有限的硬體資源下支持更大的模型。
  
### 優勢

- `zero_2`模式對GPU顯存的利用更高效，但`zero_3`模式通過進一步的模型參數分割，能夠在更小的顯存空間內進行訓練。
- 啟用`offload`功能後，模型部分或全部將被移至CPU記憶體，這有助於在不增加顯存的情況下訓練更大的模型。

這些改進允許我們在有限的硬體條件下，有效提高模型訓練的效率和性能。

## Merge Model
以下為合併LoRA與原版Llama-2模型以生成完整模型權重的步驟。執行以下命令以合併LoRA權重

```bash
model_name=chinese-alpaca-2-13b
python scripts/merge_llama2_with_chinese_lora_low_mem.py \
    --base_model ../model/$model_name \
    --lora_model ../output_model/$model_name \
    --output_type huggingface \ 
    --output_dir ../merge_model/$model_name 
```

#### 參數說明
- `--base_model`：原版Llama-2模型權重和文件的目錄位置。
- `--lora_model`：指定中文LLaMA-2/Alpaca-2 LoRA模型的解壓後文件所在位置。
- `--output_type`：選擇輸出模型的格式。可選為`pth`（PyTorch格式）或`huggingface`（Hugging Face格式）。
- `--output_dir`：設定保存合併後模型權重的位置。
- （可選）`--verbose`：合併過程中顯示詳細信息。


## Inference

使用 `inference.py` 進行模型推理，可以加上自定義的prompt。

```python
# 定義自己的prompt
DEFAULT_SYSTEM_PROMPT = """請認真回答題目"""
prompt = TEMPLATE_WITH_SYSTEM_PROMPT.format_map({'instruction': instruction, 'system_prompt': DEFAULT_SYSTEM_PROMPT})
inputs = tokenizer.encode(prompt+'\n', return_tensors="pt").to(DEV)

# 設定生成參數
generate_kwargs = dict(
    input_ids=inputs,
    temperature=0.2,
    top_p=0.9,
    top_k=40,
    do_sample=True,
    max_new_tokens=1,  # 為了回答選擇題而設定1
    repetition_penalty=1.1,
    guidance_scale=1.0
)
```

## Result

| Model                    | Performance |
|--------------------------|-------------|
| chinese-alpaca-2-1.3b    | 0.33857     |
| chinese-alpaca-2-7b      | 0.82714     |
| chinese-alpaca-2-13b     | 0.88285     |

## 未來改進

為了進一步提升模型的性能和理解能力，未來我們可以進一步採取以下方式：

- 嘗試使用第一代`chinese-llama-33b`模型來看是否能在文意理解上取得更好的結果。
- 嘗試使用預訓練過更多中文資料的其他LLaMA模型，以便模型能更好地適應中文語境。
- 測試將數據從繁體中文轉換為簡體中文的效果，因為`chinese-llama`的訓練數據多以簡體中文為主，這可能有助於提升模型的文意理解能力。


## 許可證和版權信息
```
@article{Chinese-LLaMA-Alpaca,
    title={Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca},
    author={Cui, Yiming and Yang, Ziqing and Yao, Xin},
    journal={arXiv preprint arXiv:2304.08177},
    url={https://arxiv.org/abs/2304.08177},
    year={2023}
}
```
