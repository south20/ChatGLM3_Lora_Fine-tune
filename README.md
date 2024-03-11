# 基于ChatGLM3-6B模型的Lora方法的微调（lora finetuning）

>  目前大模型微调方式Prefix Tuning、P-Tuning V1/V2到LoRA、QLoRA 全参微调SFT、本项目对ChatGLM3-6B通过多种方式微调，使模型具备落地潜质（包括但不限于客服、聊天、游戏）

- 构建训练数据集
- 微调chatglm3-6b模型（lora）
- 测试微调后的模型（基座模型+lora权重）
- 模型合并及部署


## 0.环境说明

本实验基于清华开源大模型 ChatGLM3-6B作为LLM，有关 ChatGLM3-6B的安装及配置不在本次实验中说明之内。有关安装和配置ChatGLM3-6B的请参见ChatGLM3-6B的github主页。[ChatGLM3-6B的github链接](https://github.com/THUDM/ChatGLM3-6B)
本实验按照官方的finetuning方法，对chatglm3-6b模型进行微调（finetuning）。

## 1.构建训练数据集

本实验采用一个简单的自我认知的训练集，该训练集包含100多条自我认知的数据集，属于非常少的数据集，主要是用于测试和验证lora方法的微调效果。

- 按照官方的资料，训练集的基本格式如下：

```
{
	"conversations": [
		{"role": "user",
			"content": "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞"
			}, 
		{"role": "assistant", 
			"content": "简约而不简单的牛仔外套，白色的衣身十分百搭。衣身多处有做旧破洞设计，打破单调乏味，增加一丝造型看点。衣身后背处有趣味刺绣装饰，丰富层次感，彰别样时尚。"
		}
			]
}
```

- 多轮对话的训练集的格式：

```
{
  "conversations": [
    {
        "role": "system",
        "content": "<system prompt text>"
    },
    {
        "role": "user",
        "content": "<user prompt text>"
    },
    {
        "role": "assistant",
        "content": "<assistant response text>"
    },
      // ... Muti Turn
    {
        "role": "user",
        "content": "<user prompt text>"
     },
     {
        "role": "assistant",
        "content": "<assistant response text>"
    }
    ]
}
  // ...

```

- 如果希望微调模型的对话和工具能力，应该按照以下格式整理数据。
  
```
{
  "tools": [
      // available tools, format is not restricted
    ],
    "conversations": [
    {
        "role": "system",
        "content": "<system prompt text>"
    },
    {
        "role": "user",
        "content": "<user prompt text>"
    },
    {
        "role": "assistant",
        "content": "<assistant thought to text>"
    },
    {
        "role": "tool",
        "name": "<name of the tool to be called",
        "parameters": {
          "<parameter_name>": "<parameter_value>"
      },
        "observation": "<observation>"
        // don't have to be string
    },
    {
        "role": "assistant",
        "content": "<assistant response to observation>"
    },
    // ... Muti Turn
    {
        "role": "user",
        "content": "<user prompt text>"
    },
    {
        "role": "assistant",
        "content": "<assistant response text>"
    }
  ]
}
  // ...

```

### 本次训练的自我认知数据集
- 数据集包含100条对话，其中大部分是训练数据，一小部分是测试数据。
- train.json是训练数据  dev.json是测试数据
- 数据很少，我们截取一部分如下：
  
```
{"conversations": [{"role": "user", "content": "你能做什么"}, {"role": "assistant", "content": "您好，我是 某公司客服小爱，可以帮你解答你需要的技术问题。"}]}
{"conversations": [{"role": "user", "content": "机台宕机怎么办"}, {"role": "assistant", "content": "您好，很高兴为您服务。我将告诉您以下步骤请您详细检查并排除问题"}]}
{"conversations": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "您好，我是 某公司客服小爱，一个由 大河实验室 开发的 AI 助手，很高兴认识您。请问我能为您做些什么？"}]}
{"conversations": [{"role": "user", "content": "您好"}, {"role": "assistant", "content": "您好，我是 某公司客服小爱，很高兴为您服务。有什么我可以帮您解决的问题或者需要我提供的帮助吗？"}]}
{"conversations": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "您好，我是 某公司客服小爱，一个由 大河实验室 开发的 AI 助手，很高兴认识您。请问我能为您做些什么？"}]}
{"conversations": [{"role": "user", "content": "机台无法初始化"}, {"role": "assistant", "content": "您好！请点击页面get status，获取转状信息。"}]}
{"conversations": [{"role": "user", "content": "你是谁"}, {"role": "assistant", "content": "您好，我是由 大河实验室 发明的 某公司客服小爱。我可以为您提供多种多样的服务，比如翻译、写代码、闲聊、为您答疑解惑等。"}]}
```


### finetune 记录：

服务器：

- 代码： 
    - 微调代码：finetune_hf.py
    - 推理代码：inference_hf.py
    - 合并代码：merge_model_hf.py
- 数据集：data/self_cognition
- 配置文件：configs/lora.yaml
    
    ```
    data_config:
      train_file: train.json
      val_file: dev.json
      test_file: dev.json
      num_proc: 16
    max_input_length: 128
    max_output_length: 256 #微调模型输出的最大长度
    training_args:
      # see `transformers.Seq2SeqTrainingArguments`
      output_dir: ./output
      max_steps: 3000
      # settings for data loading
      per_device_train_batch_size: 16
      dataloader_num_workers: 16 #数据加载时使用的工作进程数
      remove_unused_columns: false
      # settings for saving checkpoints
      save_strategy: steps #保存策略，默认是按步数保存（steps）
      save_steps: 500
      # settings for logging
      log_level: info
      logging_strategy: steps
      logging_steps: 10
      # settings for evaluation
      per_device_eval_batch_size: 16
      evaluation_strategy: steps
      eval_steps: 500
      # settings for optimizer
      # adam_epsilon: 1e-6
      # uncomment the following line to detect nan or inf values
      # debug: underflow_overflow
      predict_with_generate: true
      # see `transformers.GenerationConfig`
      generation_config:
        max_new_tokens: 256
      # set your absolute deepspeed path here
      #deepspeed: ds_zero_2.json
    peft_config:   #Huggingface PEFT 框架的相关参数，peft_type 选择高效微调的方式，可以为LORA 或者 PREFIX_TUNING，并需要搭配对应的参数。
      peft_type: LORA
      task_type: CAUSAL_LM
      r: 8
      lora_alpha: 32 #是控制LoRA调整幅度的参数。它决定了对原始模型参数的修改程度。较高的lora_alpha值意味着对原始模型参数的更大调整，这可能有助于模型更好地适应新的任务或数据，但也可能导致过拟合。较低的值则意味着较小的调整，可能保持模型的泛化能力，但可能不足以充分适应新任务。
      lora_dropout: 0.1 #指的是在LoRA层应用的dropout比率。这意味着在训练过程中，网络的一部分连接会随机断开，以防止模型过度依赖于训练数据中的特定模式。较高的dropout比率可以增加模型的泛化能力，但也可能导致学习效率降低。
    
    ```
    

### 1. 微调训练：

 python3 finetune_hf.py data/self_cognition ../chatglm3-6b configs/lora.yaml

- 数据集： data/self_cognition
- 基础模型： ../chatglm3-6b
- 配置参数： configs/lora.yaml


训练按照 configs/lora.yaml 的配置参数训练完成，保存到 output目录。(./output/checkpoint-3000)

### 2. 推理测试效果

 python3 inference_hf.py output/checkpoint-3000/ --prompt "你是谁?"

- 预训练模型：  output/checkpoint-3000
    - 我们没有合并训练后的模型，而是在`adapter_config.json`中记录了微调型的路径，如果你的原始模型位置发生更改，我们也要修改`adapter_config.json`中`base_model_name_or_path`的路径。 所以， 我们使用load_model_and_tokenizer，通过 AutoPeftModelForCausalLM.from_pretrained调用实现rola权重和基础模型的合并。 这个不是真正意义的合并，还是需要基础模型和lora权重分别保存。
    
    ```
    python3 inference_hf.py output/checkpoint-3000/ --prompt "你是谁?"
    Loading checkpoint shards: 100%|███████████████████████████████████████████████████| 7/7 [00:10<00:00,  1.53s/it]
    您好，我是 某公司客服小爱，一个由 大河实验室 开发的 AI 助手，很高兴认识您。请问我能为您做些什么?。
    ```
    

大家看，预训练的数据已经对模型已经起了作用，认知已经改变了。

### 3. 模型合并导出

以上的推理需要基础模型和lora权重分别加载，这样在实际项目中非常不方便，另外官方提供的各种调用方式也是按照基础模型的调用方式使用的，以后做量化或者在此基础上再训练如果不合并就很难做下一部分工作。

 python3 model_export_hf.py   ./output/checkpoint-3000/  --out-dir ./chatglm3-6b-01

- 预训练模型目录（lora）：  ./output/checkpoint-3000/
- 合并后模型输出目录：  --out-dir ./chatglm3-6b-01

```
ls chatglm3-6b-01/ -l
total 12195668
-rw-r--r-- 1 root root       1465 Feb 24 03:39 config.json
-rw-rw-r-- 1 root root       2332 Feb 24 03:39 configuration_chatglm.py
-rw-r--r-- 1 root root        111 Feb 24 03:39 generation_config.json
-rw-r--r-- 1 root root 4907627760 Feb 24 03:39 model-00001-of-00003.safetensors
-rw-r--r-- 1 root root 4895071288 Feb 24 03:39 model-00002-of-00003.safetensors
-rw-r--r-- 1 root root 2684495816 Feb 24 03:39 model-00003-of-00003.safetensors
-rw-r--r-- 1 root root      20438 Feb 24 03:39 model.safetensors.index.json
-rw-r--r-- 1 root root      55678 Feb 24 03:39 modeling_chatglm.py
-rw-rw-r-- 1 root root      14692 Feb 24 03:39 quantization.py
-rw-r--r-- 1 root root          3 Feb 24 03:39 special_tokens_map.json
-rw-r--r-- 1 root root      12998 Feb 24 03:39 tokenization_chatglm.py
-rw-r--r-- 1 root root    1018370 Feb 24 03:39 tokenizer.model
-rw-r--r-- 1 root root        700 Feb 24 03:39 tokenizer_config.json

```

chatglm3-6b-01目录就是合并后的模型，这个和基础模型chatglm3-6b使用起来应该是一样的。

我们按照通用的方法使用这个chatglm3-6b-01，做一下测试：

```

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("chatglm3-6b-01", trust_remote_code=True)
model = AutoModel.from_pretrained("chatglm3-6b-01", trust_remote_code=True, device='cuda')
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
```

测试如下：

```
>>> from transformers import AutoTokenizer, AutoModel
>>> tokenizer = AutoTokenizer.from_pretrained("chatglm3-6b-01", trust_remote_code=True)
Setting eos_token is not supported, use the default one.
Setting pad_token is not supported, use the default one.
Setting unk_token is not supported, use the default one.
>>> model = AutoModel.from_pretrained("chatglm3-6b-01", trust_remote_code=True, device='cuda')

Loading checkpoint shards: 100%|███████████████████████████████████████████████████| 3/3 [00:15<00:00,  5.19s/it]
>>> model = model.eval()
>>> response, history = model.chat(tokenizer, "你好", history=[])
>>> print(response)
您好，我是 某公司客服小爱，一个由 大河实验室 开发的 AI 助手，很高兴认识您。请问我能为您做些什么
>>> 
```
