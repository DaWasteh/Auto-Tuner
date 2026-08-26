# GGUF Modelle - Metadaten Übersicht

## qwen3.6-35b-a3b-moe-mxfp4-mmproj-f16.gguf
**Pfad:** `Alibaba\MXFP\qwen3.6-35b-a3b-moe-mxfp4-mmproj-f16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[29]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  54  32  51  53  66  32  65  51  66]` |
| `general.finetune` | `[51 53 98]` |
| `general.basename` | `[ 81 119 101 110  51  46  54]` |
| `general.size_label` | `[65 51 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[1]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[2048]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  51  53  66  45  65  51  66  47  98 108 111  98  47 109  97 105 110  47
  76  73  67  69  78  83  69]
```

---

## qwen3.6-35b-a3b-moe-mxfp4.gguf
**Pfad:** `Alibaba\MXFP\qwen3.6-35b-a3b-moe-mxfp4.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[733]` |
| `GGUF.kv_count` | `[44]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  54  32  51  53  66  32  65  51  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  54]` |
| `general.size_label` | `[51 53 66 45 65 51 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `qwen35moe.block_count` | `[40]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[38]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  51  53  66  45  65  51  66  47  98 108 111  98  47 109  97 105 110  47
  76  73  67  69  78  83  69]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## mmproj-Qwen3.6-35B-A3B-F16.gguf
**Pfad:** `Alibaba\NVFP4\mmproj-Qwen3.6-35B-A3B-F16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[29]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | `[ 51  53  98  45 115 114  99]` |
| `general.basename` | `[ 81 119 101 110  51  46  54]` |
| `general.size_label` | `[65 51 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[2048]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

**general.name:**
```jinja
[ 81 119 101 110  51  46  54  32  51  53  66  32  65  51  66  32  66 102
  49  54  32  83 114  99]
```

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  51  53  66  45  65  51  66  47  98 108 111  98  47 109  97 105 110  47
  76  73  67  69  78  83  69]
```

---

## mtp-Qwen3.6-35B-A3B-NVFP4.gguf
**Pfad:** `Alibaba\NVFP4\mtp-Qwen3.6-35B-A3B-NVFP4.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[23]` |
| `GGUF.kv_count` | `[49]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | `[ 51  53  98  45  78  86  70  80  52  45 115 114  99]` |
| `general.basename` | `[ 81 119 101 110  51  46  54]` |
| `general.size_label` | `[65 51 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  54  32  51  53  66  32  65  51  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `qwen35moe.block_count` | `[41]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `general.file_type` | `[39]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `qwen35moe.nextn_predict_layers` | `[1]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**general.name:**
```jinja
[ 81 119 101 110  51  46  54  32  51  53  66  32  65  51  66  32  78  86
  70  80  52  32  83 114  99]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  51  53  66  45  65  51  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## Qwen3.6-35B-A3B-NVFP4-Q4_K_M-mtp.gguf
**Pfad:** `Alibaba\NVFP4\Qwen3.6-35B-A3B-NVFP4-Q4_K_M-mtp.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[973]` |
| `GGUF.kv_count` | `[48]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | `[ 78  86  70  80  52  45 115 114  99]` |
| `general.basename` | `[ 81 119 101 110  51  46  54]` |
| `general.size_label` | `[51 53 66 45 65 51 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  54  32  51  53  66  32  65  51  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `qwen35moe.block_count` | `[40]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |

**general.name:**
```jinja
[ 81 119 101 110  51  46  54  32  51  53  66  32  65  51  66  32  78  86
  70  80  52  32  83 114  99]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  51  53  66  45  65  51  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## Qwen3-Coder-30B-A3B-Instruct-UD_Q6_K_XL.gguf
**Pfad:** `Alibaba\Qwen3\Qwen3-Coder-30B-A3B-Instruct-UD_Q6_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[579]` |
| `GGUF.kv_count` | `[44]` |
| `general.architecture` | `[113 119 101 110  51 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | `[ 73 110 115 116 114 117  99 116]` |
| `general.basename` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[51 48 66 45 65 51 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `qwen3moe.block_count` | `[48]` |
| `qwen3moe.context_length` | `[262144]` |
| `qwen3moe.embedding_length` | `[2048]` |
| `qwen3moe.feed_forward_length` | `[5472]` |
| `qwen3moe.attention.head_count` | `[32]` |
| `qwen3moe.attention.head_count_kv` | `[4]` |
| `qwen3moe.rope.freq_base` | `[1.e+07]` |
| `qwen3moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen3moe.expert_used_count` | `[8]` |
| `qwen3moe.attention.key_length` | `[128]` |
| `qwen3moe.attention.value_length` | `[128]` |
| `qwen3moe.expert_count` | `[128]` |
| `qwen3moe.expert_feed_forward_length` | `[768]` |
| `qwen3moe.expert_shared_feed_forward_length` | `[0]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  50]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 53 49 57 51 53 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[195 162 194 189  32 196 185]` |
| `tokenizer.ggml.eos_token_id` | `[151645]` |
| `tokenizer.ggml.padding_token_id` | `[151654]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[384]` |
| `quantize.imatrix.chunks_count` | `[154]` |

**general.name:**
```jinja
[ 81 119 101 110  51  45  67 111 100 101 114  45  51  48  66  45  65  51
  66  45  73 110 115 116 114 117  99 116]
```

**general.basename:**
```jinja
[ 81 119 101 110  51  45  67 111 100 101 114  45  51  48  66  45  65  51
  66  45  73 110 115 116 114 117  99 116]
```

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  45  67 111
 100 101 114  45  51  48  66  45  65  51  66  45  73 110 115 116 114 117
  99 116  47  98 108 111  98  47 109  97 105 110  47  76  73  67  69  78
  83  69]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.name:**
```jinja
[ 81 119 101 110  51  32  67 111 100 101 114  32  51  48  66  32  65  51
  66  32  73 110 115 116 114 117  99 116]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  45  67 111
 100 101 114  45  51  48  66  45  65  51  66  45  73 110 115 116 114 117
  99 116]
```

**tokenizer.chat_template:**
```jinja
[123  35  32 ...  32  35 125]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  45  67 111 100 101 114  45  51  48  66  45  65  51
  66  45  73 110 115 116 114 117  99 116  45  71  71  85  70  47 105 109
  97 116 114 105 120  95 117 110 115 108 111 116 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  81 119 101 110  51  45  67 111 100 101 114  45  51  48  66  45
  65  51  66  45  73 110 115 116 114 117  99 116  46 116 120 116]
```

---

## Qwen3-Coder-Next-UD_IQ4_XS.gguf
**Pfad:** `Alibaba\Qwen3\Qwen3-Coder-Next-UD_IQ4_XS.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[843]` |
| `GGUF.kv_count` | `[52]` |
| `general.architecture` | `[113 119 101 110  51 110 101 120 116]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[40]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  45  67 111 100 101 114  45  78 101 120 116]` |
| `general.basename` | `[ 81 119 101 110  51  45  67 111 100 101 114  45  78 101 120 116]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[ 53  49  50 120  50  46  53  66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  32  67 111 100 101 114  32  78 101 120 116]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `qwen3next.block_count` | `[48]` |
| `qwen3next.context_length` | `[262144]` |
| `qwen3next.embedding_length` | `[2048]` |
| `qwen3next.feed_forward_length` | `[5120]` |
| `qwen3next.attention.head_count` | `[16]` |
| `qwen3next.attention.head_count_kv` | `[2]` |
| `qwen3next.rope.freq_base` | `[5.e+06]` |
| `qwen3next.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen3next.expert_count` | `[512]` |
| `qwen3next.expert_used_count` | `[10]` |
| `qwen3next.attention.key_length` | `[256]` |
| `qwen3next.attention.value_length` | `[256]` |
| `qwen3next.expert_feed_forward_length` | `[512]` |
| `qwen3next.expert_shared_feed_forward_length` | `[512]` |
| `qwen3next.ssm.conv_kernel` | `[4]` |
| `qwen3next.ssm.state_size` | `[128]` |
| `qwen3next.ssm.group_count` | `[16]` |
| `qwen3next.ssm.time_step_rank` | `[32]` |
| `qwen3next.ssm.inner_size` | `[4096]` |
| `qwen3next.full_attention_interval` | `[4]` |
| `qwen3next.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  50]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 53 49 57 51 53 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[195 162 194 189  32 196 185]` |
| `tokenizer.ggml.eos_token_id` | `[151645]` |
| `tokenizer.ggml.padding_token_id` | `[151654]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[30]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[576]` |
| `quantize.imatrix.chunks_count` | `[154]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  45  67 111
 100 101 114  45  78 101 120 116  47  98 108 111  98  47 109  97 105 110
  47  76  73  67  69  78  83  69]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  45  67 111
 100 101 114  45  78 101 120 116]
```

**tokenizer.chat_template:**
```jinja
[123  37  32 ...  37 125  10]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  45  67 111 100 101 114  45  78 101 120 116  45  71
  71  85  70  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116
 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  81 119 101 110  51  45  67 111 100 101 114  45  78 101 120 116
  46 116 120 116]
```

---

## mmproj-Qwen3.5-122B-A10B-UD_F32.gguf
**Pfad:** `Alibaba\Qwen3.5\mmproj-Qwen3.5-122B-A10B-UD_F32.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[33]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[0.6]` |
| `general.name` | `[ 81 119 101 110  51  46  53  45  49  50  50  66  45  65  49  48  66]` |
| `general.finetune` | `[49 50 50 98]` |
| `general.basename` | `[ 81 119 101 110  51  46  53  45  49  50  50  66  45  65  49  48  66]` |
| `general.size_label` | `[65 49 48 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  53  32  49  50  50  66  32  65  49  48  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[0]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[3072]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  53  45
  49  50  50  66  45  65  49  48  66  47  98 108 111  98  47 109  97 105
 110  47  76  73  67  69  78  83  69]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  53  45
  49  50  50  66  45  65  49  48  66]
```

---

## Qwen3.5-122B-A10B-UD_Q3_K_XL-00001-of-00003.gguf
**Pfad:** `Alibaba\Qwen3.5\Qwen3.5-122B-A10B-UD_Q3_K_XL-00001-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[0]` |
| `GGUF.kv_count` | `[57]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[0.6]` |
| `general.name` | `[ 81 119 101 110  51  46  53  45  49  50  50  66  45  65  49  48  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  53  45  49  50  50  66  45  65  49  48  66]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[49 50 50 66 45 65 49 48 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  53  32  49  50  50  66  32  65  49  48  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `qwen35moe.block_count` | `[49]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[3072]` |
| `qwen35moe.attention.head_count` | `[32]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[1024]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[1024]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[64]` |
| `qwen35moe.ssm.inner_size` | `[8192]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `qwen35moe.nextn_predict_layers` | `[1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[12]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[612]` |
| `quantize.imatrix.chunks_count` | `[77]` |
| `split.no` | `[0]` |
| `split.tensors.count` | `[899]` |
| `split.count` | `[3]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  53  45
  49  50  50  66  45  65  49  48  66  47  98 108 111  98  47 109  97 105
 110  47  76  73  67  69  78  83  69]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  53  45
  49  50  50  66  45  65  49  48  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  35 125  10]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  53  45  49  50  50  66  45  65  49  48  66  45
  71  71  85  70  47 105 109  97 116 114 105 120  95 117 110 115 108 111
 116 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  81 119 101 110  51  46  53  45  49  50  50  66  45  65  49  48
  66  46 116 120 116]
```

---

## Qwen3.5-122B-A10B-UD_Q3_K_XL-00002-of-00003.gguf
**Pfad:** `Alibaba\Qwen3.5\Qwen3.5-122B-A10B-UD_Q3_K_XL-00002-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[774]` |
| `GGUF.kv_count` | `[3]` |
| `split.no` | `[1]` |
| `split.tensors.count` | `[899]` |
| `split.count` | `[3]` |

---

## Qwen3.5-122B-A10B-UD_Q3_K_XL-00003-of-00003.gguf
**Pfad:** `Alibaba\Qwen3.5\Qwen3.5-122B-A10B-UD_Q3_K_XL-00003-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[125]` |
| `GGUF.kv_count` | `[3]` |
| `split.no` | `[2]` |
| `split.tensors.count` | `[899]` |
| `split.count` | `[3]` |

---

## mmproj-Qwen3.6-27B-MTP-BF16.gguf
**Pfad:** `Alibaba\Qwen3.6\mmproj-Qwen3.6-27B-MTP-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[33]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  54  45  50  55  66]` |
| `general.finetune` | `[50 55 98]` |
| `general.basename` | `[ 81 119 101 110  51  46  54  45  50  55  66]` |
| `general.size_label` | `[52 54 49 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  54  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[5120]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  50  55  66  47  98 108 111  98  47 109  97 105 110  47  76  73  67  69
  78  83  69]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  50  55  66]
```

---

## mmproj-Qwen3.6-35B-A3B-MTP-UD_BF16.gguf
**Pfad:** `Alibaba\Qwen3.6\mmproj-Qwen3.6-35B-A3B-MTP-UD_BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[33]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  54  45  51  53  66  45  65  51  66]` |
| `general.finetune` | `[51 53 98]` |
| `general.basename` | `[ 81 119 101 110  51  46  54  45  51  53  66  45  65  51  66]` |
| `general.size_label` | `[65 51 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  54  32  51  53  66  32  65  51  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[2048]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  51  53  66  45  65  51  66  47  98 108 111  98  47 109  97 105 110  47
  76  73  67  69  78  83  69]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  51  53  66  45  65  51  66]
```

---

## Qwen3.6-27B-MTP-IQ4_XS.gguf
**Pfad:** `Alibaba\Qwen3.6\Qwen3.6-27B-MTP-IQ4_XS.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[866]` |
| `GGUF.kv_count` | `[52]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  54  45  50  55  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  54  45  50  55  66]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 55 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  54  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[30]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[496]` |
| `quantize.imatrix.chunks_count` | `[76]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  50  55  66  47  98 108 111  98  47 109  97 105 110  47  76  73  67  69
  78  83  69]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  54  45  50  55  66  45  71  71  85  70  47 105
 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46 103 103 117
 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  81 119 101 110  51  46  54  45  50  55  66  46 116 120 116]
```

---

## Qwen3.6-27B-MTP-UD_Q6_K_XL.gguf
**Pfad:** `Alibaba\Qwen3.6\Qwen3.6-27B-MTP-UD_Q6_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[866]` |
| `GGUF.kv_count` | `[52]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  54  45  50  55  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  54  45  50  55  66]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 55 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  54  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[496]` |
| `quantize.imatrix.chunks_count` | `[76]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  50  55  66  47  98 108 111  98  47 109  97 105 110  47  76  73  67  69
  78  83  69]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  54  45  50  55  66  45  71  71  85  70  47 105
 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46 103 103 117
 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  81 119 101 110  51  46  54  45  50  55  66  46 116 120 116]
```

---

## Qwen3.6-35B-A3B-MTP-UD_Q6_K.gguf
**Pfad:** `Alibaba\Qwen3.6\Qwen3.6-35B-A3B-MTP-UD_Q6_K.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[753]` |
| `GGUF.kv_count` | `[55]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  54  45  51  53  66  45  65  51  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  54  45  51  53  66  45  65  51  66]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[51 53 66 45 65 51 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  54  32  51  53  66  32  65  51  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `qwen35moe.block_count` | `[41]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `qwen35moe.nextn_predict_layers` | `[1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[510]` |
| `quantize.imatrix.chunks_count` | `[77]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  51  53  66  45  65  51  66  47  98 108 111  98  47 109  97 105 110  47
  76  73  67  69  78  83  69]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  51  53  66  45  65  51  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  54  45  51  53  66  45  65  51  66  45  71  71
  85  70  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116 104
  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  81 119 101 110  51  46  54  45  51  53  66  45  65  51  66  46
 116 120 116]
```

---

## Qwen3.6-35B-A3B-MTP-UD_Q8_K_XL.gguf
**Pfad:** `Alibaba\Qwen3.6\Qwen3.6-35B-A3B-MTP-UD_Q8_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[753]` |
| `GGUF.kv_count` | `[55]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  54  45  51  53  66  45  65  51  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  54  45  51  53  66  45  65  51  66]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[51 53 66 45 65 51 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  54  32  51  53  66  32  65  51  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `qwen35moe.block_count` | `[41]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `qwen35moe.nextn_predict_layers` | `[1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[510]` |
| `quantize.imatrix.chunks_count` | `[77]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  51  53  66  45  65  51  66  47  98 108 111  98  47 109  97 105 110  47
  76  73  67  69  78  83  69]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  51  53  66  45  65  51  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  54  45  51  53  66  45  65  51  66  45  71  71
  85  70  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116 104
  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  81 119 101 110  51  46  54  45  51  53  66  45  65  51  66  46
 116 120 116]
```

---

## mmproj-Qwen3.8-27B-BF16.gguf
**Pfad:** `Alibaba\Qwen3.8\mmproj-Qwen3.8-27B-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[35]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.finetune` | `[50 55 98]` |
| `general.basename` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.description` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[52 54 49 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  56  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[5120]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

**general.description:**
```jinja
[ 82 101 110 101 119  97 108  32 111 102  32 116 104 101  32  98 101 108
 111 118 101 100  32  81 119 101 110  32 109 111 100 101 108  44  32 100
 101 108 105 118 101 114 105 110 103  32 117 110 109  97 116  99 104 101
 100  32 105 110 116 101 108 108 105 103 101 110  99 101  32 100 101 110
 115 105 116 121  46]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  56  45
  50  55  66]
```

---

## mmproj-Qwen3.8-27B-Ridge-BF16.gguf
**Pfad:** `Alibaba\Qwen3.8\mmproj-Qwen3.8-27B-Ridge-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[28]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  56  32  50  55  66  32  66 102  49  54]` |
| `general.finetune` | `[50 55 98]` |
| `general.basename` | `[ 81 119 101 110  51  46  56]` |
| `general.size_label` | `[52 54 49 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[5120]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

---

## Qwen3.8-27B-DFlash2-BF16.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-27B-DFlash2-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[81]` |
| `GGUF.kv_count` | `[47]` |
| `general.architecture` | `[100 102 108  97 115 104]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 81 119 101 110  51  46  56  45  50  55  66  45  68  70 108  97 115 104   50]` |
| `general.author` | `[ 73 110  99 111  32  65  73]` |
| `general.organization` | `[122  45 108  97  98]` |
| `general.finetune` | `[ 68  70 108  97 115 104  50]` |
| `general.basename` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.size_label` | `[49 46 57 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.source.url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  56  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `dflash.block_count` | `[5]` |
| `dflash.context_length` | `[262144]` |
| `dflash.embedding_length` | `[5120]` |
| `dflash.feed_forward_length` | `[17408]` |
| `dflash.attention.head_count` | `[32]` |
| `dflash.attention.head_count_kv` | `[8]` |
| `dflash.attention.causal` | `[False]` |
| `dflash.rope.freq_base` | `[1.e+07]` |
| `dflash.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `dflash.attention.key_length` | `[128]` |
| `dflash.attention.value_length` | `[128]` |
| `general.file_type` | `[32]` |
| `dflash.block_size` | `[8]` |
| `dflash.conv_kernel_size` | `[2]` |
| `dflash.conv_group_size` | `[16]` |
| `dflash.selector_rank` | `[256]` |
| `dflash.selector_top_k` | `[16]` |
| `dflash.target_layers` | `[62]` |
| `dflash.attention.sliding_window` | `[2048]` |
| `dflash.attention.sliding_window_pattern` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.mask_token_id` | `[248070]` |

**general.source.url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 122  45 108  97  98  47  81 119 101 110  51  46  56
  45  50  55  66  45  68  70 108  97 115 104  50]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  56  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## Qwen3.8-27B-DFlash2-Q4_K_M.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-27B-DFlash2-Q4_K_M.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[81]` |
| `GGUF.kv_count` | `[47]` |
| `general.architecture` | `[100 102 108  97 115 104]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 81 119 101 110  51  46  56  45  50  55  66  45  68  70 108  97 115 104   50]` |
| `general.author` | `[ 73 110  99 111  32  65  73]` |
| `general.organization` | `[122  45 108  97  98]` |
| `general.finetune` | `[ 68  70 108  97 115 104  50]` |
| `general.basename` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.size_label` | `[49 46 57 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.source.url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  56  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `dflash.block_count` | `[5]` |
| `dflash.context_length` | `[262144]` |
| `dflash.embedding_length` | `[5120]` |
| `dflash.feed_forward_length` | `[17408]` |
| `dflash.attention.head_count` | `[32]` |
| `dflash.attention.head_count_kv` | `[8]` |
| `dflash.attention.causal` | `[False]` |
| `dflash.rope.freq_base` | `[1.e+07]` |
| `dflash.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `dflash.attention.key_length` | `[128]` |
| `dflash.attention.value_length` | `[128]` |
| `dflash.block_size` | `[8]` |
| `dflash.conv_kernel_size` | `[2]` |
| `dflash.conv_group_size` | `[16]` |
| `dflash.selector_rank` | `[256]` |
| `dflash.selector_top_k` | `[16]` |
| `dflash.target_layers` | `[62]` |
| `dflash.attention.sliding_window` | `[2048]` |
| `dflash.attention.sliding_window_pattern` | `[ True]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.mask_token_id` | `[248070]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |

**general.source.url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 122  45 108  97  98  47  81 119 101 110  51  46  56
  45  50  55  66  45  68  70 108  97 115 104  50]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  56  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## Qwen3.8-27B-DFlash2-Q8_0.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-27B-DFlash2-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[81]` |
| `GGUF.kv_count` | `[47]` |
| `general.architecture` | `[100 102 108  97 115 104]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 81 119 101 110  51  46  56  45  50  55  66  45  68  70 108  97 115 104   50]` |
| `general.author` | `[ 73 110  99 111  32  65  73]` |
| `general.organization` | `[122  45 108  97  98]` |
| `general.finetune` | `[ 68  70 108  97 115 104  50]` |
| `general.basename` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.size_label` | `[49 46 57 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.source.url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  56  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `dflash.block_count` | `[5]` |
| `dflash.context_length` | `[262144]` |
| `dflash.embedding_length` | `[5120]` |
| `dflash.feed_forward_length` | `[17408]` |
| `dflash.attention.head_count` | `[32]` |
| `dflash.attention.head_count_kv` | `[8]` |
| `dflash.attention.causal` | `[False]` |
| `dflash.rope.freq_base` | `[1.e+07]` |
| `dflash.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `dflash.attention.key_length` | `[128]` |
| `dflash.attention.value_length` | `[128]` |
| `dflash.block_size` | `[8]` |
| `dflash.conv_kernel_size` | `[2]` |
| `dflash.conv_group_size` | `[16]` |
| `dflash.selector_rank` | `[256]` |
| `dflash.selector_top_k` | `[16]` |
| `dflash.target_layers` | `[62]` |
| `dflash.attention.sliding_window` | `[2048]` |
| `dflash.attention.sliding_window_pattern` | `[ True]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.mask_token_id` | `[248070]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**general.source.url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 122  45 108  97  98  47  81 119 101 110  51  46  56
  45  50  55  66  45  68  70 108  97 115 104  50]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  56  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## Qwen3.8-27B-Q8_0.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-27B-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[866]` |
| `GGUF.kv_count` | `[45]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  56  32  50  55  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  56]` |
| `general.size_label` | `[50 55 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[496]` |
| `quantize.imatrix.chunks_count` | `[582]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 47 109 111 100 101 108 115  95 111 117 116  47  81 119 101 110  51  46
  56  45  50  55  66  45  71  71  85  70  47  81 119 101 110  51  46  56
  45  50  55  66  45 105 109  97 116 114 105 120  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[ 47 109 111 100 101 108 115  95 111 117 116  47  81 119 101 110  51  46
  56  45  50  55  66  45  71  71  85  70  47  81 119 101 110  51  46  56
  45  50  55  66  45  99  97 108 105  98 114  97 116 105 111 110  45 118
  54  46 116 120 116]
```

---

## Qwen3.8-27B-Ridge-3.7bpw.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-27B-Ridge-3.7bpw.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[866]` |
| `GGUF.kv_count` | `[45]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  56  32  50  55  66  32  66 102  49  54]` |
| `general.basename` | `[ 81 119 101 110  51  46  56]` |
| `general.size_label` | `[50 55 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[29]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[497]` |
| `quantize.imatrix.chunks_count` | `[80]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 47 119 111 114 107 115 112  97  99 101  47 113 119 101 110  51  56  45
 101 120 108  51  47 119 111 114 107  47 114 105 100 103 101  46 105 109
  97 116 114 105 120]
```

**quantize.imatrix.dataset:**
```jinja
[ 47 119 111 114 107 115 112  97  99 101  47 113 119 101 110  51  56  45
 101 120 108  51  47 119 111 114 107  47 114 105 100 103 101  45  99  97
 108  46 116 120 116]
```

---

## Qwen3.8-27B-UD-IQ4_XS.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-27B-UD-IQ4_XS.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[866]` |
| `GGUF.kv_count` | `[50]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.description` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 55 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  56  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[30]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[496]` |
| `quantize.imatrix.chunks_count` | `[1251]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**general.description:**
```jinja
[ 82 101 110 101 119  97 108  32 111 102  32 116 104 101  32  98 101 108
 111 118 101 100  32  81 119 101 110  32 109 111 100 101 108  44  32 100
 101 108 105 118 101 114 105 110 103  32 117 110 109  97 116  99 104 101
 100  32 105 110 116 101 108 108 105 103 101 110  99 101  32 100 101 110
 115 105 116 121  46]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  56  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  56  45  50  55  66  45  71  71  85  70  47 105
 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46 103 103 117
 102]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

---

## Qwen3.8-27B-UD-Q3_K_XL.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-27B-UD-Q3_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[866]` |
| `GGUF.kv_count` | `[50]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.description` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 55 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  56  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[13]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[496]` |
| `quantize.imatrix.chunks_count` | `[1251]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**general.description:**
```jinja
[ 82 101 110 101 119  97 108  32 111 102  32 116 104 101  32  98 101 108
 111 118 101 100  32  81 119 101 110  32 109 111 100 101 108  44  32 100
 101 108 105 118 101 114 105 110 103  32 117 110 109  97 116  99 104 101
 100  32 105 110 116 101 108 108 105 103 101 110  99 101  32 100 101 110
 115 105 116 121  46]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  56  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  56  45  50  55  66  45  71  71  85  70  47 105
 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46 103 103 117
 102]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

---

## Qwen3.8-27B-UD-Q4_K_XL.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-27B-UD-Q4_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[866]` |
| `GGUF.kv_count` | `[50]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.description` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 55 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  56  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[496]` |
| `quantize.imatrix.chunks_count` | `[1251]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**general.description:**
```jinja
[ 82 101 110 101 119  97 108  32 111 102  32 116 104 101  32  98 101 108
 111 118 101 100  32  81 119 101 110  32 109 111 100 101 108  44  32 100
 101 108 105 118 101 114 105 110 103  32 117 110 109  97 116  99 104 101
 100  32 105 110 116 101 108 108 105 103 101 110  99 101  32 100 101 110
 115 105 116 121  46]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  56  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  56  45  50  55  66  45  71  71  85  70  47 105
 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46 103 103 117
 102]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

---

## Qwen3.8-27B-UD-Q5_K_XL.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-27B-UD-Q5_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[866]` |
| `GGUF.kv_count` | `[50]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.description` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 55 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  56  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[17]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[496]` |
| `quantize.imatrix.chunks_count` | `[1251]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**general.description:**
```jinja
[ 82 101 110 101 119  97 108  32 111 102  32 116 104 101  32  98 101 108
 111 118 101 100  32  81 119 101 110  32 109 111 100 101 108  44  32 100
 101 108 105 118 101 114 105 110 103  32 117 110 109  97 116  99 104 101
 100  32 105 110 116 101 108 108 105 103 101 110  99 101  32 100 101 110
 115 105 116 121  46]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  56  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  56  45  50  55  66  45  71  71  85  70  47 105
 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46 103 103 117
 102]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

---

## Qwen3.8-27B-UD-Q6_K_XL.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-27B-UD-Q6_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[866]` |
| `GGUF.kv_count` | `[50]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.description` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 55 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  56  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[496]` |
| `quantize.imatrix.chunks_count` | `[1251]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**general.description:**
```jinja
[ 82 101 110 101 119  97 108  32 111 102  32 116 104 101  32  98 101 108
 111 118 101 100  32  81 119 101 110  32 109 111 100 101 108  44  32 100
 101 108 105 118 101 114 105 110 103  32 117 110 109  97 116  99 104 101
 100  32 105 110 116 101 108 108 105 103 101 110  99 101  32 100 101 110
 115 105 116 121  46]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  56  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  56  45  50  55  66  45  71  71  85  70  47 105
 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46 103 103 117
 102]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

---

## Qwen3.8-27B-UD-Q8_K_XL.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-27B-UD-Q8_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[866]` |
| `GGUF.kv_count` | `[51]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.basename` | `[ 81 119 101 110  51  46  56  45  50  55  66]` |
| `general.description` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 55 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  56  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[496]` |
| `quantize.imatrix.chunks_count` | `[45]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**general.description:**
```jinja
[ 82 101 110 101 119  97 108  32 111 102  32 116 104 101  32  98 101 108
 111 118 101 100  32  81 119 101 110  32 109 111 100 101 108  44  32 100
 101 108 105 118 101 114 105 110 103  32 117 110 109  97 116  99 104 101
 100  32 105 110 116 101 108 108 105 103 101 110  99 101  32 100 101 110
 115 105 116 121  46]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  56  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**quantize.imatrix.file:**
```jinja
[107 108 100  95 109 111 100 101 108 115  47 113 119 101 110  51  56  45
  50  55  98  45 103 103 117 102  47 105 109  97 116 114 105 120  47  81
 119 101 110  51  46  56  45  50  55  66  45  71  71  85  70  47 105 109
  97 116 114 105 120  95 117 110 115 108 111 116 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  81 119 101 110  51  46  56  45  50  55  66  46 116 120 116]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

---

## Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-Flash-Next-UD-IQ1_S-00001-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[0]` |
| `GGUF.kv_count` | `[67]` |
| `general.architecture` | `[113 119 101 110  52 101 120 112]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 81 119 101 110  51  46  56  32  70 108  97 115 104  32  78 101 120 116]` |
| `general.description` | *Siehe Code-Block unten* |
| `general.size_label` | `[ 53  49  50 120  53  54  66]` |
| `qwen4exp.block_count` | `[48]` |
| `qwen4exp.context_length` | `[262144]` |
| `qwen4exp.embedding_length` | `[2560]` |
| `qwen4exp.attention.head_count` | `[24]` |
| `qwen4exp.attention.head_count_kv` | `[2]` |
| `qwen4exp.rope.dimension_sections` | `[0]` |
| `qwen4exp.rope.freq_base` | `[1.e+07]` |
| `qwen4exp.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen4exp.expert_count` | `[512]` |
| `qwen4exp.expert_used_count` | `[10]` |
| `qwen4exp.attention.key_length` | `[256]` |
| `qwen4exp.attention.value_length` | `[256]` |
| `qwen4exp.expert_feed_forward_length` | `[640]` |
| `qwen4exp.expert_shared_feed_forward_length` | `[640]` |
| `qwen4exp.ssm.conv_kernel` | `[4]` |
| `qwen4exp.ssm.state_size` | `[128]` |
| `qwen4exp.ssm.group_count` | `[16]` |
| `qwen4exp.ssm.time_step_rank` | `[48]` |
| `qwen4exp.ssm.inner_size` | `[6144]` |
| `qwen4exp.full_attention_interval` | `[4]` |
| `qwen4exp.rope.dimension_count` | `[64]` |
| `qwen4exp.hyper_connection.count` | `[4]` |
| `qwen4exp.hyper_connection.low_rank` | `[320]` |
| `qwen4exp.attention.indexer.head_count` | `[4]` |
| `qwen4exp.attention.indexer.key_length` | `[128]` |
| `qwen4exp.attention.indexer.top_k` | `[2048]` |
| `qwen4exp.attention.compress_ratios` | `[4]` |
| `qwen4exp.ple.layers` | `[1]` |
| `qwen4exp.ple.ngram_size` | `[3]` |
| `qwen4exp.ple.heads_per_ngram` | `[8]` |
| `qwen4exp.ple.conv_kernel` | `[4]` |
| `qwen4exp.ple.eos_token_id` | `[248044]` |
| `qwen4exp.embedding_length_per_layer_input` | `[160]` |
| `qwen4exp.ple.layer_multipliers` | `[8052911324071]` |
| `qwen4exp.ple.head_offsets` | `[300001275]` |
| `qwen4exp.ple.head_vocab_sizes` | `[20000171]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[24]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[926]` |
| `quantize.imatrix.chunks_count` | `[45]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `qwen4exp.ple.image_token_id` | `[248056]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `split.no` | `[0]` |
| `split.tensors.count` | `[1224]` |
| `split.count` | `[3]` |

**general.description:**
```jinja
[ 65  32  80 114 101 118 105 101 119  32 111 102  32 116 104 101  32  81
 119 101 110  52  32  65 114  99 104 105 116 101  99 116 117 114 101]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  51  46  56  45  70 108  97 115 104  45  78 101 120 116
  45  71  71  85  70  47 105 109  97 116 114 105 120  95 117 110 115 108
 111 116 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  81 119 101 110  51  46  56  45  70 108  97 115 104  45  78 101
 120 116  46 116 120 116]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

---

## Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[595]` |
| `GGUF.kv_count` | `[3]` |
| `split.no` | `[1]` |
| `split.tensors.count` | `[1224]` |
| `split.count` | `[3]` |

---

## Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf
**Pfad:** `Alibaba\Qwen3.8\Qwen3.8-Flash-Next-UD-IQ1_S-00003-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[629]` |
| `GGUF.kv_count` | `[3]` |
| `split.no` | `[2]` |
| `split.tensors.count` | `[1224]` |
| `split.count` | `[3]` |

---

## Qwen-AgentWorld-35B-A3B-UD_Q6_K.gguf
**Pfad:** `Alibaba\QwenAgentWorld\Qwen-AgentWorld-35B-A3B-UD_Q6_K.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[733]` |
| `GGUF.kv_count` | `[57]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[0.6]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.basename` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[51 53 66 45 65 51 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.count` | `[1]` |
| `general.dataset.0.name` | `[ 65 103 101 110 116  87 111 114 108 100  66 101 110  99 104]` |
| `general.dataset.0.organization` | `[ 81 119 101 110]` |
| `general.dataset.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `qwen35moe.block_count` | `[40]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[510]` |
| `quantize.imatrix.chunks_count` | `[76]` |

**general.name:**
```jinja
[ 81 119 101 110  45  65 103 101 110 116 119 111 114 108 100  45  51  53
  66  45  65  51  66]
```

**general.basename:**
```jinja
[ 81 119 101 110  45  65 103 101 110 116 119 111 114 108 100  45  51  53
  66  45  65  51  66]
```

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  45  65 103 101
 110 116  87 111 114 108 100  45  51  53  66  45  65  51  66  47  98 108
 111  98  47 109  97 105 110  47  76  73  67  69  78  83  69]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.name:**
```jinja
[ 81 119 101 110  32  65 103 101 110 116  87 111 114 108 100  32  51  53
  66  32  65  51  66]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  45  65 103 101
 110 116  87 111 114 108 100  45  51  53  66  45  65  51  66]
```

**general.dataset.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  65 103 101 110 116  87 111 114
 108 100  66 101 110  99 104]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  45  65 103 101 110 116  87 111 114 108 100  45  51  53
  66  45  65  51  66  45  71  71  85  70  47 105 109  97 116 114 105 120
  95 117 110 115 108 111 116 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  81 119 101 110  45  65 103 101 110 116  87 111 114 108 100  45
  51  53  66  45  65  51  66  46 116 120 116]
```

---

## Qwen-AgentWorld-35B-A3B-UD_Q8_K_XL.gguf
**Pfad:** `Alibaba\QwenAgentWorld\Qwen-AgentWorld-35B-A3B-UD_Q8_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[733]` |
| `GGUF.kv_count` | `[57]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[0.6]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.basename` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[51 53 66 45 65 51 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.count` | `[1]` |
| `general.dataset.0.name` | `[ 65 103 101 110 116  87 111 114 108 100  66 101 110  99 104]` |
| `general.dataset.0.organization` | `[ 81 119 101 110]` |
| `general.dataset.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `qwen35moe.block_count` | `[40]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248055]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[510]` |
| `quantize.imatrix.chunks_count` | `[76]` |

**general.name:**
```jinja
[ 81 119 101 110  45  65 103 101 110 116 119 111 114 108 100  45  51  53
  66  45  65  51  66]
```

**general.basename:**
```jinja
[ 81 119 101 110  45  65 103 101 110 116 119 111 114 108 100  45  51  53
  66  45  65  51  66]
```

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  45  65 103 101
 110 116  87 111 114 108 100  45  51  53  66  45  65  51  66  47  98 108
 111  98  47 109  97 105 110  47  76  73  67  69  78  83  69]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.name:**
```jinja
[ 81 119 101 110  32  65 103 101 110 116  87 111 114 108 100  32  51  53
  66  32  65  51  66]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  45  65 103 101
 110 116  87 111 114 108 100  45  51  53  66  45  65  51  66]
```

**general.dataset.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  65 103 101 110 116  87 111 114
 108 100  66 101 110  99 104]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 81 119 101 110  45  65 103 101 110 116  87 111 114 108 100  45  51  53
  66  45  65  51  66  45  71  71  85  70  47 105 109  97 116 114 105 120
  95 117 110 115 108 111 116 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  81 119 101 110  45  65 103 101 110 116  87 111 114 108 100  45
  51  53  66  45  65  51  66  46 116 120 116]
```

---

## mmproj-Unlimited-OCR-F16.gguf
**Pfad:** `Baidu\mmproj-Unlimited-OCR-F16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[476]` |
| `GGUF.kv_count` | `[27]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | `[ 85 110 108 105 109 105 116 101 100  32  79  67  82]` |
| `general.size_label` | `[52 48 49 77]` |
| `general.license` | `[109 105 116]` |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.languages` | `[109 117 108 116 105 108 105 110 103 117  97 108]` |
| `general.file_type` | `[1]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[1280]` |
| `clip.vision.image_size` | `[224]` |
| `clip.vision.patch_size` | `[14]` |
| `clip.vision.embedding_length` | `[1024]` |
| `clip.vision.feed_forward_length` | `[64]` |
| `clip.vision.block_count` | `[24]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[100 101 101 112 115 101 101 107 111  99 114]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.projector.scale_factor` | `[1]` |
| `clip.vision.window_size` | `[14]` |
| `clip.vision.sam.block_count` | `[12]` |
| `clip.vision.sam.embedding_length` | `[768]` |
| `clip.vision.sam.head_count` | `[12]` |
| `general.quantization_version` | `[2]` |

---

## Unlimited-OCR-BF16.gguf
**Pfad:** `Baidu\Unlimited-OCR-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[155]` |
| `GGUF.kv_count` | `[37]` |
| `general.architecture` | `[100 101 101 112 115 101 101 107  50  45 111  99 114]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 85 110 108 105 109 105 116 101 100  32  79  67  82]` |
| `general.size_label` | `[ 54  52 120  53  53  48  77]` |
| `general.license` | `[109 105 116]` |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.languages` | `[109 117 108 116 105 108 105 110 103 117  97 108]` |
| `deepseek2-ocr.block_count` | `[12]` |
| `deepseek2-ocr.context_length` | `[32768]` |
| `deepseek2-ocr.embedding_length` | `[1280]` |
| `deepseek2-ocr.feed_forward_length` | `[6848]` |
| `deepseek2-ocr.attention.head_count` | `[10]` |
| `deepseek2-ocr.attention.head_count_kv` | `[10]` |
| `deepseek2-ocr.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `deepseek2-ocr.expert_used_count` | `[6]` |
| `deepseek2-ocr.expert_group_count` | `[1]` |
| `deepseek2-ocr.expert_group_used_count` | `[1]` |
| `general.file_type` | `[32]` |
| `deepseek2-ocr.leading_dense_block_count` | `[1]` |
| `deepseek2-ocr.vocab_size` | `[129280]` |
| `deepseek2-ocr.expert_feed_forward_length` | `[896]` |
| `deepseek2-ocr.expert_count` | `[64]` |
| `deepseek2-ocr.expert_shared_count` | `[2]` |
| `deepseek2-ocr.rope.dimension_count` | `[0]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[100 101 101 112 115 101 101 107  45 118  51]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 50 57 50 55 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[195 165 194 177 196 173  32 195 166 194 170 196 178]` |
| `tokenizer.ggml.bos_token_id` | `[0]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.padding_token_id` | `[2]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_sep_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |

**tokenizer.chat_template:**
```jinja
[123  37  32 102 111 114  32 109  32 105 110  32 109 101 115 115  97 103
 101 115  32  37 125 123 123 109  91  39  99 111 110 116 101 110 116  39
  93 125 125 123  37  32 101 110 100 102 111 114  32  37 125]
```

---

## mmproj-ThinkingCap-Qwen3.6-27B-f16.gguf
**Pfad:** `bottlecapAI\mmproj-ThinkingCap-Qwen3.6-27B-f16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[25]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | *Siehe Code-Block unten* |
| `general.size_label` | `[52 54 49 77]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[5120]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

**general.name:**
```jinja
[ 51 100  97  57  98 101  51  57  57  97  49  57  53  49  97  99  57 102
  99  99  54 100  49  98  55  54  97 102  57  53  54  53  99 101 100  50
  56  53 102  49]
```

**general.finetune:**
```jinja
[ 51 100  97  57  98 101  51  57  57  97  49  57  53  49  97  99  57 102
  99  99  54 100  49  98  55  54  97 102  57  53  54  53  99 101 100  50
  56  53 102  49]
```

---

## ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf
**Pfad:** `bottlecapAI\ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[866]` |
| `GGUF.kv_count` | `[37]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 67 107 112 116  50  49  48  48]` |
| `general.size_label` | `[50 55 66]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## North-Mini-Code-1.0-UD_Q6_K_XL.gguf
**Pfad:** `CohereLabs\North-Mini-Code-1.0-UD_Q6_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[442]` |
| `GGUF.kv_count` | `[57]` |
| `general.architecture` | `[ 99 111 104 101 114 101  50 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 78 111 114 116 104  45  77 105 110 105  45  67 111 100 101  45  49  46   48]` |
| `general.version` | `[49 46 48]` |
| `general.finetune` | `[ 67 111 100 101]` |
| `general.basename` | `[ 78 111 114 116 104  45  77 105 110 105  45  67 111 100 101  45  49  46   48]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[ 77 105 110 105]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 78 111 114 116 104  32  77 105 110 105  32  67 111 100 101  32  49  46   48]` |
| `general.base_model.0.version` | `[49 46 48]` |
| `general.base_model.0.organization` | `[ 67 111 104 101 114 101  76  97  98 115]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 103 101 110 116]` |
| `cohere2moe.block_count` | `[49]` |
| `cohere2moe.context_length` | `[500000]` |
| `cohere2moe.embedding_length` | `[2048]` |
| `cohere2moe.feed_forward_length` | `[3072]` |
| `cohere2moe.attention.head_count` | `[32]` |
| `cohere2moe.attention.head_count_kv` | `[4]` |
| `cohere2moe.rope.freq_base` | `[50000.]` |
| `cohere2moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `cohere2moe.attention.layer_norm_epsilon` | `[1.e-05]` |
| `cohere2moe.expert_count` | `[128]` |
| `cohere2moe.expert_used_count` | `[8]` |
| `cohere2moe.attention.key_length` | `[128]` |
| `cohere2moe.attention.value_length` | `[128]` |
| `cohere2moe.logit_scale` | `[1.]` |
| `cohere2moe.attention.sliding_window` | `[4096]` |
| `cohere2moe.attention.sliding_window_pattern` | `[False]` |
| `cohere2moe.vocab_size` | `[262144]` |
| `cohere2moe.expert_feed_forward_length` | `[768]` |
| `cohere2moe.leading_dense_block_count` | `[1]` |
| `cohere2moe.expert_weights_norm` | `[False]` |
| `cohere2moe.expert_gating_func` | `[2]` |
| `cohere2moe.rope.dimension_count` | `[128]` |
| `cohere2moe.rope.scaling.type` | `[110 111 110 101]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[116 105 110 121  95  97 121  97]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 54 50 49 52 51 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[196 160  98 114  32  97 110 105]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[255001]` |
| `tokenizer.ggml.unknown_token_id` | `[4]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_sep_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[391]` |
| `quantize.imatrix.chunks_count` | `[200]` |

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  67 111 104 101 114 101  76  97  98 115  47  78 111
 114 116 104  45  77 105 110 105  45  67 111 100 101  45  49  46  48]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 78 111 114 116 104  45  77 105 110 105  45  67 111 100 101  45  49  46
  48  45  71  71  85  70  47 105 109  97 116 114 105 120  95 117 110 115
 108 111 116 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  78 111 114 116 104  45  77 105 110 105  45  67 111 100 101  45
  49  46  48  46 116 120 116]
```

---

## North-Mini-Code-1.0-UD_Q8_K_XL.gguf
**Pfad:** `CohereLabs\North-Mini-Code-1.0-UD_Q8_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[442]` |
| `GGUF.kv_count` | `[57]` |
| `general.architecture` | `[ 99 111 104 101 114 101  50 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 78 111 114 116 104  45  77 105 110 105  45  67 111 100 101  45  49  46   48]` |
| `general.version` | `[49 46 48]` |
| `general.finetune` | `[ 67 111 100 101]` |
| `general.basename` | `[ 78 111 114 116 104  45  77 105 110 105  45  67 111 100 101  45  49  46   48]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[ 77 105 110 105]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 78 111 114 116 104  32  77 105 110 105  32  67 111 100 101  32  49  46   48]` |
| `general.base_model.0.version` | `[49 46 48]` |
| `general.base_model.0.organization` | `[ 67 111 104 101 114 101  76  97  98 115]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 103 101 110 116]` |
| `cohere2moe.block_count` | `[49]` |
| `cohere2moe.context_length` | `[500000]` |
| `cohere2moe.embedding_length` | `[2048]` |
| `cohere2moe.feed_forward_length` | `[3072]` |
| `cohere2moe.attention.head_count` | `[32]` |
| `cohere2moe.attention.head_count_kv` | `[4]` |
| `cohere2moe.rope.freq_base` | `[50000.]` |
| `cohere2moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `cohere2moe.attention.layer_norm_epsilon` | `[1.e-05]` |
| `cohere2moe.expert_count` | `[128]` |
| `cohere2moe.expert_used_count` | `[8]` |
| `cohere2moe.attention.key_length` | `[128]` |
| `cohere2moe.attention.value_length` | `[128]` |
| `cohere2moe.logit_scale` | `[1.]` |
| `cohere2moe.attention.sliding_window` | `[4096]` |
| `cohere2moe.attention.sliding_window_pattern` | `[False]` |
| `cohere2moe.vocab_size` | `[262144]` |
| `cohere2moe.expert_feed_forward_length` | `[768]` |
| `cohere2moe.leading_dense_block_count` | `[1]` |
| `cohere2moe.expert_weights_norm` | `[False]` |
| `cohere2moe.expert_gating_func` | `[2]` |
| `cohere2moe.rope.dimension_count` | `[128]` |
| `cohere2moe.rope.scaling.type` | `[110 111 110 101]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[116 105 110 121  95  97 121  97]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 54 50 49 52 51 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[196 160  98 114  32  97 110 105]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[255001]` |
| `tokenizer.ggml.unknown_token_id` | `[4]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_sep_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[391]` |
| `quantize.imatrix.chunks_count` | `[200]` |

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  67 111 104 101 114 101  76  97  98 115  47  78 111
 114 116 104  45  77 105 110 105  45  67 111 100 101  45  49  46  48]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 78 111 114 116 104  45  77 105 110 105  45  67 111 100 101  45  49  46
  48  45  71  71  85  70  47 105 109  97 116 114 105 120  95 117 110 115
 108 111 116 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  78 111 114 116 104  45  77 105 110 105  45  67 111 100 101  45
  49  46  48  46 116 120 116]
```

---

## VibeCoder-20b-RL1.0-MOE-MXFP4.gguf
**Pfad:** `EpistemeAI\VibeCoder-20b-RL1.0-MOE-MXFP4.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[459]` |
| `GGUF.kv_count` | `[46]` |
| `general.architecture` | `[103 112 116  45 111 115 115]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 73 110]` |
| `general.size_label` | `[ 51  50 120  50  46  52  66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.version` | `[48 46 48 50]` |
| `general.base_model.0.organization` | `[ 69 112 105 115 116 101 109 101  65  73]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.dataset.count` | `[1]` |
| `general.dataset.0.name` | *Siehe Code-Block unten* |
| `general.dataset.0.organization` | `[ 69 112 105 115 116 101 109 101  65  73]` |
| `general.dataset.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[103 112 116  95 111 115 115]` |
| `general.languages` | `[101 110]` |
| `gpt-oss.block_count` | `[24]` |
| `gpt-oss.context_length` | `[131072]` |
| `gpt-oss.embedding_length` | `[2880]` |
| `gpt-oss.feed_forward_length` | `[2880]` |
| `gpt-oss.attention.head_count` | `[64]` |
| `gpt-oss.attention.head_count_kv` | `[8]` |
| `gpt-oss.rope.scaling.type` | `[121  97 114 110]` |
| `gpt-oss.rope.scaling.factor` | `[32.]` |
| `gpt-oss.rope.scaling.original_context_length` | `[4096]` |
| `gpt-oss.rope.scaling.yarn_beta_fast` | `[32.]` |
| `gpt-oss.rope.scaling.yarn_beta_slow` | `[1.]` |
| `gpt-oss.rope.freq_base` | `[150000.]` |
| `gpt-oss.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `gpt-oss.expert_count` | `[32]` |
| `gpt-oss.expert_used_count` | `[4]` |
| `gpt-oss.attention.key_length` | `[64]` |
| `gpt-oss.attention.value_length` | `[64]` |
| `gpt-oss.attention.sliding_window` | `[128]` |
| `gpt-oss.expert_feed_forward_length` | `[2880]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[103 112 116  45  52 111]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 48 49 48 56 55 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[196 160  99 111  99 111  32 115]` |
| `tokenizer.ggml.bos_token_id` | `[199998]` |
| `tokenizer.ggml.eos_token_id` | `[200002]` |
| `tokenizer.ggml.padding_token_id` | `[200017]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[38]` |

**general.base_model.0.name:**
```jinja
[ 86 105  98 101  67 111 100 101 114  32  50  48  98  32  48  46  48  50
  32  68 101  98 117 103 103 101 114]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  69 112 105 115 116 101 109 101  65  73  47  86 105
  98 101  67 111 100 101 114  45  50  48  98  45  48  46  48  50  45  68
 101  98 117 103 103 101 114]
```

**general.dataset.0.name:**
```jinja
[ 86 105  98 101  32  67 111 100 101 114  32  80  97 114 116  32  68 101
  98 117 103]
```

**general.dataset.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  69 112 105 115 116 101 109 101  65  73  47 118 105
  98 101  45  99 111 100 101 114  45 112  97 114 116  45 100 101  98 117
 103]
```

**tokenizer.chat_template:**
```jinja
[123  35  32 ...  32  35 125]
```

---

## Kwaipilot_KAT-Coder-V2.5-Dev-Q6_K_L.gguf
**Pfad:** `Finetunes\Kwaipilot_KAT-Coder-V2.5-Dev-Q6_K_L.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[733]` |
| `GGUF.kv_count` | `[49]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 75  65  84  32  67 111 100 101 114  32  86  50  46  53  32  68 101 118]` |
| `general.size_label` | `[ 50  53  54 120  50  46  54  66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  54  32  51  53  66  32  65  51  66]` |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `general.languages` | `[122 104]` |
| `qwen35moe.block_count` | `[40]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[510]` |
| `quantize.imatrix.chunks_count` | `[802]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 47 109 111 100 101 108 115  95 111 117 116  47  75  65  84  45  67 111
 100 101 114  45  86  50  46  53  45  68 101 118  45  71  71  85  70  47
  75 119  97 105 112 105 108 111 116  95  75  65  84  45  67 111 100 101
 114  45  86  50  46  53  45  68 101 118  45 105 109  97 116 114 105 120
  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[ 47 116 114  97 105 110 105 110 103  95 100 105 114  47  99  97 108 105
  98 114  97 116 105 111 110  95 100  97 116  97 118  53  46 116 120 116]
```

---

## diffusiongemma-26B-A4B-it-Q8_0.gguf
**Pfad:** `Google\Gemma 4 Diffusion\diffusiongemma-26B-A4B-it-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[692]` |
| `GGUF.kv_count` | `[44]` |
| `general.architecture` | `[100 105 102 102 117 115 105 111 110  45 103 101 109 109  97]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 68 103  95  82  99  48  80  49  95  80  97 116  99 104 101 100]` |
| `general.size_label` | `[ 49  50  56 120  50  46  54  66]` |
| `diffusion-gemma.block_count` | `[30]` |
| `diffusion-gemma.context_length` | `[262144]` |
| `diffusion-gemma.embedding_length` | `[2816]` |
| `diffusion-gemma.feed_forward_length` | `[2112]` |
| `diffusion-gemma.attention.head_count` | `[16]` |
| `diffusion-gemma.attention.head_count_kv` | `[2]` |
| `diffusion-gemma.rope.freq_base` | `[1.e+06]` |
| `diffusion-gemma.rope.freq_base_swa` | `[10000.]` |
| `diffusion-gemma.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `diffusion-gemma.expert_count` | `[128]` |
| `diffusion-gemma.expert_used_count` | `[8]` |
| `diffusion-gemma.attention.key_length` | `[512]` |
| `diffusion-gemma.attention.value_length` | `[512]` |
| `diffusion-gemma.final_logit_softcapping` | `[30.]` |
| `diffusion-gemma.attention.sliding_window` | `[1024]` |
| `diffusion-gemma.attention.shared_kv_layers` | `[0]` |
| `diffusion-gemma.embedding_length_per_layer_input` | `[0]` |
| `diffusion-gemma.attention.sliding_window_pattern` | `[False]` |
| `diffusion-gemma.attention.key_length_swa` | `[256]` |
| `diffusion-gemma.attention.value_length_swa` | `[256]` |
| `diffusion-gemma.expert_feed_forward_length` | `[704]` |
| `diffusion-gemma.rope.dimension_count` | `[512]` |
| `diffusion-gemma.rope.dimension_count_swa` | `[256]` |
| `diffusion-gemma.attention.causal` | `[False]` |
| `diffusion.canvas_length` | `[256]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  45  37 125]
```

---

## gemma-4-12b-it-v2-UD-Q8_K_XL.gguf
**Pfad:** `Google\Gemma 4 v2\gemma-4-12b-it-v2-UD-Q8_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[667]` |
| `GGUF.kv_count` | `[58]` |
| `general.architecture` | `[103 101 109 109  97  52]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 71 101 109 109  97  45  52  45  49  50  66  45  73 116]` |
| `general.finetune` | `[105 116]` |
| `general.basename` | `[ 71 101 109 109  97  45  52  45  49  50  66  45  73 116]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[49 50 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  49  50  98  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 110 121  45 116 111  45  97 110 121]` |
| `gemma4.block_count` | `[48]` |
| `gemma4.context_length` | `[262144]` |
| `gemma4.embedding_length` | `[3840]` |
| `gemma4.feed_forward_length` | `[15360]` |
| `gemma4.attention.head_count` | `[16]` |
| `gemma4.attention.head_count_kv` | `[1]` |
| `gemma4.rope.freq_base` | `[1.e+06]` |
| `gemma4.rope.freq_base_swa` | `[10000.]` |
| `gemma4.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4.attention.key_length` | `[512]` |
| `gemma4.attention.value_length` | `[512]` |
| `gemma4.final_logit_softcapping` | `[30.]` |
| `gemma4.attention.sliding_window` | `[1024]` |
| `gemma4.attention.shared_kv_layers` | `[0]` |
| `gemma4.embedding_length_per_layer_input` | `[0]` |
| `gemma4.attention.sliding_window_pattern` | `[False]` |
| `gemma4.attention.key_length_swa` | `[256]` |
| `gemma4.attention.value_length_swa` | `[256]` |
| `gemma4.rope.dimension_count` | `[512]` |
| `gemma4.rope.dimension_count_swa` | `[256]` |
| `tokenizer.ggml.suppress_tokens` | `[258882]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[106]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[328]` |
| `quantize.imatrix.chunks_count` | `[141]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  49  50  98  45 105 116]
```

**quantize.imatrix.file:**
```jinja
[103 101 109 109  97  45  52  45  49  50  98  45 105 116  45  71  71  85
  70  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46
 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95 103 101 109 109  97  45  52  45  49  50  98  45 105 116  46 116
 120 116]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  37 125  10]
```

---

## gemma-4-26B-A4B-it-v2-UD-Q8_K_XL.gguf
**Pfad:** `Google\Gemma 4 v2\gemma-4-26B-A4B-it-v2-UD-Q8_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[658]` |
| `GGUF.kv_count` | `[60]` |
| `general.architecture` | `[103 101 109 109  97  52]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 71 101 109 109  97  45  52  45  50  54  66  45  65  52  66  45  73 116]` |
| `general.finetune` | `[105 116]` |
| `general.basename` | `[ 71 101 109 109  97  45  52  45  50  54  66  45  65  52  66  45  73 116]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 54 66 45 65 52 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  50  54  66  32  65  52  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `gemma4.block_count` | `[30]` |
| `gemma4.context_length` | `[262144]` |
| `gemma4.embedding_length` | `[2816]` |
| `gemma4.feed_forward_length` | `[2112]` |
| `gemma4.attention.head_count` | `[16]` |
| `gemma4.attention.head_count_kv` | `[2]` |
| `gemma4.rope.freq_base` | `[1.e+06]` |
| `gemma4.rope.freq_base_swa` | `[10000.]` |
| `gemma4.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4.expert_count` | `[128]` |
| `gemma4.expert_used_count` | `[8]` |
| `gemma4.attention.key_length` | `[512]` |
| `gemma4.attention.value_length` | `[512]` |
| `gemma4.final_logit_softcapping` | `[30.]` |
| `gemma4.attention.sliding_window` | `[1024]` |
| `gemma4.attention.shared_kv_layers` | `[0]` |
| `gemma4.embedding_length_per_layer_input` | `[0]` |
| `gemma4.attention.sliding_window_pattern` | `[False]` |
| `gemma4.attention.key_length_swa` | `[256]` |
| `gemma4.attention.value_length_swa` | `[256]` |
| `gemma4.expert_feed_forward_length` | `[704]` |
| `gemma4.rope.dimension_count` | `[512]` |
| `gemma4.rope.dimension_count_swa` | `[256]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[106]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[295]` |
| `quantize.imatrix.chunks_count` | `[141]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  50  54  66  45  65  52  66  45 105 116]
```

**quantize.imatrix.file:**
```jinja
[103 101 109 109  97  45  52  45  50  54  66  45  65  52  66  45 105 116
  45  71  71  85  70  47 105 109  97 116 114 105 120  95 117 110 115 108
 111 116 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95 103 101 109 109  97  45  52  45  50  54  66  45  65  52  66  45
 105 116  46 116 120 116]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  37 125  10]
```

---

## gemma-4-31B-it-v2-UD-Q5_K_XL.gguf
**Pfad:** `Google\Gemma 4 v2\gemma-4-31B-it-v2-UD-Q5_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[833]` |
| `GGUF.kv_count` | `[57]` |
| `general.architecture` | `[103 101 109 109  97  52]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 71 101 109 109  97  45  52  45  51  49  66  45  73 116]` |
| `general.finetune` | `[105 116]` |
| `general.basename` | `[ 71 101 109 109  97  45  52  45  51  49  66  45  73 116]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[51 49 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  51  49  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `gemma4.block_count` | `[60]` |
| `gemma4.context_length` | `[262144]` |
| `gemma4.embedding_length` | `[5376]` |
| `gemma4.feed_forward_length` | `[21504]` |
| `gemma4.attention.head_count` | `[32]` |
| `gemma4.attention.head_count_kv` | `[4]` |
| `gemma4.rope.freq_base` | `[1.e+06]` |
| `gemma4.rope.freq_base_swa` | `[10000.]` |
| `gemma4.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4.attention.key_length` | `[512]` |
| `gemma4.attention.value_length` | `[512]` |
| `gemma4.final_logit_softcapping` | `[30.]` |
| `gemma4.attention.sliding_window` | `[1024]` |
| `gemma4.attention.shared_kv_layers` | `[0]` |
| `gemma4.embedding_length_per_layer_input` | `[0]` |
| `gemma4.attention.sliding_window_pattern` | `[False]` |
| `gemma4.attention.key_length_swa` | `[256]` |
| `gemma4.attention.value_length_swa` | `[256]` |
| `gemma4.rope.dimension_count` | `[512]` |
| `gemma4.rope.dimension_count_swa` | `[256]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[106]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[17]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[410]` |
| `quantize.imatrix.chunks_count` | `[141]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  51  49  66  45 105 116]
```

**quantize.imatrix.file:**
```jinja
[103 101 109 109  97  45  52  45  51  49  66  45 105 116  45  71  71  85
  70  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46
 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95 103 101 109 109  97  45  52  45  51  49  66  45 105 116  46 116
 120 116]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  37 125  10]
```

---

## mmproj-gemma-4-12b-it-v2-F32.gguf
**Pfad:** `Google\Gemma 4 v2\mmproj-gemma-4-12b-it-v2-F32.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[11]` |
| `GGUF.kv_count` | `[41]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 71 101 109 109  97  45  52  45  49  50  66  45  73 116]` |
| `general.finetune` | `[ 49  50  98  45 105 116]` |
| `general.basename` | `[ 71 101 109 109  97  45  52  45  49  50  66  45  73 116]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[53 50 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  49  50  98  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 110 121  45 116 111  45  97 110 121]` |
| `general.file_type` | `[0]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[3840]` |
| `clip.vision.image_size` | `[224]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[3840]` |
| `clip.vision.feed_forward_length` | `[0]` |
| `clip.vision.block_count` | `[0]` |
| `clip.vision.attention.head_count` | `[1]` |
| `clip.vision.image_mean` | `[0.]` |
| `clip.vision.image_std` | `[1.]` |
| `clip.has_audio_encoder` | `[ True]` |
| `clip.audio.projection_dim` | `[3840]` |
| `clip.audio.embedding_length` | `[640]` |
| `clip.audio.feed_forward_length` | `[0]` |
| `clip.audio.block_count` | `[0]` |
| `clip.audio.attention.head_count` | `[1]` |
| `clip.vision.projector_type` | `[103 101 109 109  97  52 117 118]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.audio.projector_type` | `[103 101 109 109  97  52 117  97]` |
| `clip.audio.num_mel_bins` | `[128]` |
| `clip.audio.attention.layer_norm_epsilon` | `[1.e-06]` |
| `general.quantization_version` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  49  50  98  45 105 116]
```

---

## mmproj-gemma-4-26B-A4B-it-v2-F32.gguf
**Pfad:** `Google\Gemma 4 v2\mmproj-gemma-4-26B-A4B-it-v2-F32.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[356]` |
| `GGUF.kv_count` | `[30]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 71 101 109 109  97  45  52  45  50  54  66  45  65  52  66  45  73 116]` |
| `general.finetune` | `[ 50  54  98  45 105 116]` |
| `general.basename` | `[ 71 101 109 109  97  45  52  45  50  54  66  45  65  52  66  45  73 116]` |
| `general.size_label` | `[65 52 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  50  54  66  32  65  52  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[0]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[2816]` |
| `clip.vision.image_size` | `[224]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.]` |
| `clip.vision.image_std` | `[1.]` |
| `clip.vision.projector_type` | `[103 101 109 109  97  52 118]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `general.quantization_version` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  50  54  66  45  65  52  66  45 105 116]
```

---

## mmproj-gemma-4-31b-it-v2-F32.gguf
**Pfad:** `Google\Gemma 4 v2\mmproj-gemma-4-31b-it-v2-F32.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[356]` |
| `GGUF.kv_count` | `[30]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 71 101 109 109  97  45  52  45  51  49  66  45  73 116]` |
| `general.finetune` | `[ 51  49  98  45 105 116]` |
| `general.basename` | `[ 71 101 109 109  97  45  52  45  51  49  66  45  73 116]` |
| `general.size_label` | `[53 55 54 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  51  49  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[0]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[5376]` |
| `clip.vision.image_size` | `[224]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.]` |
| `clip.vision.image_std` | `[1.]` |
| `clip.vision.projector_type` | `[103 101 109 109  97  52 118]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `general.quantization_version` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  51  49  66  45 105 116]
```

---

## mtp-gemma-4-12b-it-v2.gguf
**Pfad:** `Google\Gemma 4 v2\mtp-gemma-4-12b-it-v2.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[49]` |
| `GGUF.kv_count` | `[48]` |
| `general.architecture` | `[103 101 109 109  97  52  45  97 115 115 105 115 116  97 110 116]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | `[ 49  50  98  45 105 116  45  97 115 115 105 115 116  97 110 116]` |
| `general.basename` | `[103 101 109 109  97  45  52]` |
| `general.size_label` | `[52 50 51 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 110 121  45 116 111  45  97 110 121]` |
| `gemma4-assistant.block_count` | `[4]` |
| `gemma4-assistant.context_length` | `[262144]` |
| `gemma4-assistant.embedding_length` | `[1024]` |
| `gemma4-assistant.feed_forward_length` | `[8192]` |
| `gemma4-assistant.attention.head_count` | `[16]` |
| `gemma4-assistant.attention.head_count_kv` | `[1]` |
| `gemma4-assistant.rope.freq_base` | `[1.e+06]` |
| `gemma4-assistant.rope.freq_base_swa` | `[10000.]` |
| `gemma4-assistant.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4-assistant.attention.key_length` | `[512]` |
| `gemma4-assistant.attention.value_length` | `[512]` |
| `general.file_type` | `[7]` |
| `gemma4-assistant.attention.sliding_window` | `[1024]` |
| `gemma4-assistant.attention.shared_kv_layers` | `[4]` |
| `gemma4-assistant.embedding_length_per_layer_input` | `[0]` |
| `gemma4-assistant.attention.sliding_window_pattern` | `[False]` |
| `gemma4-assistant.attention.key_length_swa` | `[256]` |
| `gemma4-assistant.attention.value_length_swa` | `[256]` |
| `gemma4-assistant.rope.dimension_count` | `[512]` |
| `gemma4-assistant.rope.dimension_count_swa` | `[256]` |
| `gemma4-assistant.embedding_length_out` | `[3840]` |
| `gemma4-assistant.nextn_predict_layers` | `[4]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_sep_token` | `[False]` |
| `tokenizer.ggml.add_space_prefix` | `[False]` |

**general.name:**
```jinja
[ 71 101 109 109  97  32  52  32  49  50  66  32  73 116  32  65 115 115
 105 115 116  97 110 116]
```

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

---

## mtp-gemma-4-26B-A4B-it-v2.gguf
**Pfad:** `Google\Gemma 4 v2\mtp-gemma-4-26B-A4B-it-v2.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[49]` |
| `GGUF.kv_count` | `[46]` |
| `general.architecture` | `[103 101 109 109  97  52  45  97 115 115 105 115 116  97 110 116]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[50 54 66 32 65 52 66]` |
| `general.finetune` | `[50 54 66]` |
| `general.size_label` | `[65 52 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 110 121  45 116 111  45  97 110 121]` |
| `gemma4-assistant.block_count` | `[4]` |
| `gemma4-assistant.context_length` | `[262144]` |
| `gemma4-assistant.embedding_length` | `[1024]` |
| `gemma4-assistant.feed_forward_length` | `[8192]` |
| `gemma4-assistant.attention.head_count` | `[16]` |
| `gemma4-assistant.attention.head_count_kv` | `[2]` |
| `gemma4-assistant.rope.freq_base` | `[1.e+06]` |
| `gemma4-assistant.rope.freq_base_swa` | `[10000.]` |
| `gemma4-assistant.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4-assistant.attention.key_length` | `[512]` |
| `gemma4-assistant.attention.value_length` | `[512]` |
| `general.file_type` | `[7]` |
| `gemma4-assistant.attention.sliding_window` | `[1024]` |
| `gemma4-assistant.attention.shared_kv_layers` | `[4]` |
| `gemma4-assistant.embedding_length_per_layer_input` | `[0]` |
| `gemma4-assistant.attention.sliding_window_pattern` | `[False]` |
| `gemma4-assistant.attention.key_length_swa` | `[256]` |
| `gemma4-assistant.attention.value_length_swa` | `[256]` |
| `gemma4-assistant.rope.dimension_count` | `[512]` |
| `gemma4-assistant.rope.dimension_count_swa` | `[256]` |
| `gemma4-assistant.embedding_length_out` | `[2816]` |
| `gemma4-assistant.nextn_predict_layers` | `[4]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

---

## mtp-gemma-4-31B-it-v2.gguf
**Pfad:** `Google\Gemma 4 v2\mtp-gemma-4-31B-it-v2.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[49]` |
| `GGUF.kv_count` | `[46]` |
| `general.architecture` | `[103 101 109 109  97  52  45  97 115 115 105 115 116  97 110 116]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[51 49 66]` |
| `general.finetune` | `[51 49 66]` |
| `general.size_label` | `[52 55 48 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 110 121  45 116 111  45  97 110 121]` |
| `gemma4-assistant.block_count` | `[4]` |
| `gemma4-assistant.context_length` | `[262144]` |
| `gemma4-assistant.embedding_length` | `[1024]` |
| `gemma4-assistant.feed_forward_length` | `[8192]` |
| `gemma4-assistant.attention.head_count` | `[32]` |
| `gemma4-assistant.attention.head_count_kv` | `[4]` |
| `gemma4-assistant.rope.freq_base` | `[1.e+06]` |
| `gemma4-assistant.rope.freq_base_swa` | `[10000.]` |
| `gemma4-assistant.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4-assistant.attention.key_length` | `[512]` |
| `gemma4-assistant.attention.value_length` | `[512]` |
| `general.file_type` | `[7]` |
| `gemma4-assistant.attention.sliding_window` | `[1024]` |
| `gemma4-assistant.attention.shared_kv_layers` | `[4]` |
| `gemma4-assistant.embedding_length_per_layer_input` | `[0]` |
| `gemma4-assistant.attention.sliding_window_pattern` | `[False]` |
| `gemma4-assistant.attention.key_length_swa` | `[256]` |
| `gemma4-assistant.attention.value_length_swa` | `[256]` |
| `gemma4-assistant.rope.dimension_count` | `[512]` |
| `gemma4-assistant.rope.dimension_count_swa` | `[256]` |
| `gemma4-assistant.embedding_length_out` | `[5376]` |
| `gemma4-assistant.nextn_predict_layers` | `[4]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

---

## gemma-4-12b-it-v2-qat-q4_0.gguf
**Pfad:** `Google\Gemma 4 v2 QAT\gemma-4-12b-it-v2-qat-q4_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[667]` |
| `GGUF.kv_count` | `[50]` |
| `general.architecture` | `[103 101 109 109  97  52]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 72 102]` |
| `general.size_label` | `[49 50 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  49  50  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 110 121  45 116 111  45  97 110 121]` |
| `gemma4.block_count` | `[48]` |
| `gemma4.context_length` | `[262144]` |
| `gemma4.embedding_length` | `[3840]` |
| `gemma4.feed_forward_length` | `[15360]` |
| `gemma4.attention.head_count` | `[16]` |
| `gemma4.attention.head_count_kv` | `[1]` |
| `gemma4.rope.freq_base` | `[1.e+06]` |
| `gemma4.rope.freq_base_swa` | `[10000.]` |
| `gemma4.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4.attention.key_length` | `[512]` |
| `gemma4.attention.value_length` | `[512]` |
| `gemma4.final_logit_softcapping` | `[30.]` |
| `gemma4.attention.sliding_window` | `[1024]` |
| `gemma4.attention.shared_kv_layers` | `[0]` |
| `gemma4.embedding_length_per_layer_input` | `[0]` |
| `gemma4.attention.sliding_window_pattern` | `[False]` |
| `gemma4.attention.key_length_swa` | `[256]` |
| `gemma4.attention.value_length_swa` | `[256]` |
| `gemma4.rope.dimension_count` | `[512]` |
| `gemma4.rope.dimension_count_swa` | `[256]` |
| `tokenizer.ggml.suppress_tokens` | `[258882]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  49  50  66  45 105 116]
```

**tokenizer.chat_template:**
```jinja
[123  35  10 ...  37 125  10]
```

---

## gemma-4-26B-it-v2-mmproj.gguf
**Pfad:** `Google\Gemma 4 v2 QAT\gemma-4-26B-it-v2-mmproj.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[356]` |
| `GGUF.kv_count` | `[28]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 72 102]` |
| `general.size_label` | `[53 55 51 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  50  54  66  32  65  52  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[2816]` |
| `clip.vision.image_size` | `[224]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.]` |
| `clip.vision.image_std` | `[1.]` |
| `clip.vision.projector_type` | `[103 101 109 109  97  52 118]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `general.quantization_version` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  50  54  66  45  65  52  66  45 105 116]
```

---

## gemma-4-26B-it-v2_q4_0.gguf
**Pfad:** `Google\Gemma 4 v2 QAT\gemma-4-26B-it-v2_q4_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[658]` |
| `GGUF.kv_count` | `[52]` |
| `general.architecture` | `[103 101 109 109  97  52]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 72 102]` |
| `general.size_label` | `[ 49  50  56 120  50  46  54  66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  50  54  66  32  65  52  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `gemma4.block_count` | `[30]` |
| `gemma4.context_length` | `[262144]` |
| `gemma4.embedding_length` | `[2816]` |
| `gemma4.feed_forward_length` | `[2112]` |
| `gemma4.attention.head_count` | `[16]` |
| `gemma4.attention.head_count_kv` | `[2]` |
| `gemma4.rope.freq_base` | `[1.e+06]` |
| `gemma4.rope.freq_base_swa` | `[10000.]` |
| `gemma4.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4.expert_count` | `[128]` |
| `gemma4.expert_used_count` | `[8]` |
| `gemma4.attention.key_length` | `[512]` |
| `gemma4.attention.value_length` | `[512]` |
| `gemma4.final_logit_softcapping` | `[30.]` |
| `gemma4.attention.sliding_window` | `[1024]` |
| `gemma4.attention.shared_kv_layers` | `[0]` |
| `gemma4.embedding_length_per_layer_input` | `[0]` |
| `gemma4.attention.sliding_window_pattern` | `[False]` |
| `gemma4.attention.key_length_swa` | `[256]` |
| `gemma4.attention.value_length_swa` | `[256]` |
| `gemma4.expert_feed_forward_length` | `[704]` |
| `gemma4.rope.dimension_count` | `[512]` |
| `gemma4.rope.dimension_count_swa` | `[256]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  50  54  66  45  65  52  66  45 105 116]
```

**tokenizer.chat_template:**
```jinja
[123  35  10 ...  37 125  10]
```

---

## gemma-4-31B-it-v2-mmproj.gguf
**Pfad:** `Google\Gemma 4 v2 QAT\gemma-4-31B-it-v2-mmproj.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[356]` |
| `GGUF.kv_count` | `[28]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 72 102]` |
| `general.size_label` | `[53 55 54 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  51  49  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[5376]` |
| `clip.vision.image_size` | `[224]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.]` |
| `clip.vision.image_std` | `[1.]` |
| `clip.vision.projector_type` | `[103 101 109 109  97  52 118]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `general.quantization_version` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  51  49  66  45 105 116]
```

---

## gemma-4-31B-it-v2_q4_0.gguf
**Pfad:** `Google\Gemma 4 v2 QAT\gemma-4-31B-it-v2_q4_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[833]` |
| `GGUF.kv_count` | `[49]` |
| `general.architecture` | `[103 101 109 109  97  52]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 72 102]` |
| `general.size_label` | `[51 49 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  51  49  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `gemma4.block_count` | `[60]` |
| `gemma4.context_length` | `[262144]` |
| `gemma4.embedding_length` | `[5376]` |
| `gemma4.feed_forward_length` | `[21504]` |
| `gemma4.attention.head_count` | `[32]` |
| `gemma4.attention.head_count_kv` | `[4]` |
| `gemma4.rope.freq_base` | `[1.e+06]` |
| `gemma4.rope.freq_base_swa` | `[10000.]` |
| `gemma4.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `gemma4.attention.key_length` | `[512]` |
| `gemma4.attention.value_length` | `[512]` |
| `gemma4.final_logit_softcapping` | `[30.]` |
| `gemma4.attention.sliding_window` | `[1024]` |
| `gemma4.attention.shared_kv_layers` | `[0]` |
| `gemma4.embedding_length_per_layer_input` | `[0]` |
| `gemma4.attention.sliding_window_pattern` | `[False]` |
| `gemma4.attention.key_length_swa` | `[256]` |
| `gemma4.attention.value_length_swa` | `[256]` |
| `gemma4.rope.dimension_count` | `[512]` |
| `gemma4.rope.dimension_count_swa` | `[256]` |
| `tokenizer.ggml.model` | `[103 101 109 109  97  52]` |
| `tokenizer.ggml.tokens` | `[ 60 117 110 117 115 101 100  54  50  50  54  62]` |
| `tokenizer.ggml.scores` | `[-1000.]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[226 150 129 224 164 166  32 224 164 191 224 164 178 224 164 190 224 164  136]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[1]` |
| `tokenizer.ggml.unknown_token_id` | `[3]` |
| `tokenizer.ggml.padding_token_id` | `[0]` |
| `tokenizer.ggml.mask_token_id` | `[4]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  51  49  66  45 105 116]
```

**tokenizer.chat_template:**
```jinja
[123  35  10 ...  37 125  10]
```

---

## mmproj-gemma-4-12b-it-v2-qat-q4_0.gguf
**Pfad:** `Google\Gemma 4 v2 QAT\mmproj-gemma-4-12b-it-v2-qat-q4_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[11]` |
| `GGUF.kv_count` | `[37]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[64]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 72 102]` |
| `general.size_label` | `[53 50 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.license.link` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 101 109 109  97  32  52  32  49  50  66  32  73 116]` |
| `general.base_model.0.organization` | `[ 71 111 111 103 108 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[ 97 110 121  45 116 111  45  97 110 121]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[3840]` |
| `clip.vision.image_size` | `[224]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[3840]` |
| `clip.vision.feed_forward_length` | `[0]` |
| `clip.vision.block_count` | `[0]` |
| `clip.vision.attention.head_count` | `[0]` |
| `clip.vision.image_mean` | `[0.]` |
| `clip.vision.image_std` | `[1.]` |
| `clip.has_audio_encoder` | `[ True]` |
| `clip.audio.projection_dim` | `[3840]` |
| `clip.audio.embedding_length` | `[640]` |
| `clip.audio.feed_forward_length` | `[0]` |
| `clip.audio.block_count` | `[0]` |
| `clip.audio.attention.head_count` | `[0]` |
| `clip.vision.projector_type` | `[103 101 109 109  97  52 117 118]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.audio.projector_type` | `[103 101 109 109  97  52 117  97]` |
| `clip.audio.num_mel_bins` | `[128]` |
| `clip.audio.attention.layer_norm_epsilon` | `[1.e-06]` |
| `general.quantization_version` | `[2]` |

**general.license.link:**
```jinja
[104 116 116 112 115  58  47  47  97 105  46 103 111 111 103 108 101  46
 100 101 118  47 103 101 109 109  97  47 100 111  99 115  47 103 101 109
 109  97  95  52  95 108 105  99 101 110 115 101]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 103 111 111 103 108 101  47 103 101 109 109  97  45
  52  45  49  50  66  45 105 116]
```

---

## granite-4.1-30b-IQ4_XS.gguf
**Pfad:** `IBM\granite-4.1-30b-IQ4_XS.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[578]` |
| `GGUF.kv_count` | `[45]` |
| `general.architecture` | `[103 114  97 110 105 116 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 71 114  97 110 105 116 101  45  52  46  49  45  51  48  66]` |
| `general.basename` | `[ 71 114  97 110 105 116 101  45  52  46  49  45  51  48  66]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[51 48 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 114  97 110 105 116 101  32  52  46  49  32  51  48  98]` |
| `general.base_model.0.organization` | `[ 73  98 109  32  71 114  97 110 105 116 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[103 114  97 110 105 116 101  45  52  46  48]` |
| `granite.block_count` | `[64]` |
| `granite.context_length` | `[131072]` |
| `granite.embedding_length` | `[4096]` |
| `granite.feed_forward_length` | `[32768]` |
| `granite.attention.head_count` | `[32]` |
| `granite.attention.head_count_kv` | `[8]` |
| `granite.rope.freq_base` | `[5.e+07]` |
| `granite.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `granite.vocab_size` | `[100352]` |
| `granite.rope.dimension_count` | `[128]` |
| `granite.attention.scale` | `[0.0078125]` |
| `granite.embedding_scale` | `[12.]` |
| `granite.residual_scale` | `[0.175]` |
| `granite.logit_scale` | `[16.]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[100  98 114 120]` |
| `tokenizer.ggml.tokens` | `[ 60 124 117 110 117 115 101 100  95  56  50 124  62]` |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[196 160  67 111 110  32 118 101 121 111 114]` |
| `tokenizer.ggml.bos_token_id` | `[100257]` |
| `tokenizer.ggml.eos_token_id` | `[100257]` |
| `tokenizer.ggml.unknown_token_id` | `[100269]` |
| `tokenizer.ggml.padding_token_id` | `[100256]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[30]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[448]` |
| `quantize.imatrix.chunks_count` | `[209]` |

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 105  98 109  45 103 114  97 110 105 116 101  47 103
 114  97 110 105 116 101  45  52  46  49  45  51  48  98]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

**quantize.imatrix.file:**
```jinja
[103 114  97 110 105 116 101  45  52  46  49  45  51  48  98  45  71  71
  85  70  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116 104
  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95 103 114  97 110 105 116 101  45  52  46  49  45  51  48  98  46
 116 120 116]
```

---

## granite-4.1-30b-UD_Q8_K_XL.gguf
**Pfad:** `IBM\granite-4.1-30b-UD_Q8_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[578]` |
| `GGUF.kv_count` | `[45]` |
| `general.architecture` | `[103 114  97 110 105 116 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 71 114  97 110 105 116 101  45  52  46  49  45  51  48  66]` |
| `general.basename` | `[ 71 114  97 110 105 116 101  45  52  46  49  45  51  48  66]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[51 48 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 114  97 110 105 116 101  32  52  46  49  32  51  48  98]` |
| `general.base_model.0.organization` | `[ 73  98 109  32  71 114  97 110 105 116 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[103 114  97 110 105 116 101  45  52  46  48]` |
| `granite.block_count` | `[64]` |
| `granite.context_length` | `[131072]` |
| `granite.embedding_length` | `[4096]` |
| `granite.feed_forward_length` | `[32768]` |
| `granite.attention.head_count` | `[32]` |
| `granite.attention.head_count_kv` | `[8]` |
| `granite.rope.freq_base` | `[5.e+07]` |
| `granite.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `granite.vocab_size` | `[100352]` |
| `granite.rope.dimension_count` | `[128]` |
| `granite.attention.scale` | `[0.0078125]` |
| `granite.embedding_scale` | `[12.]` |
| `granite.residual_scale` | `[0.175]` |
| `granite.logit_scale` | `[16.]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[100  98 114 120]` |
| `tokenizer.ggml.tokens` | `[ 60 124 117 110 117 115 101 100  95  56  50 124  62]` |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[196 160  67 111 110  32 118 101 121 111 114]` |
| `tokenizer.ggml.bos_token_id` | `[100257]` |
| `tokenizer.ggml.eos_token_id` | `[100257]` |
| `tokenizer.ggml.unknown_token_id` | `[100269]` |
| `tokenizer.ggml.padding_token_id` | `[100256]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[448]` |
| `quantize.imatrix.chunks_count` | `[209]` |

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 105  98 109  45 103 114  97 110 105 116 101  47 103
 114  97 110 105 116 101  45  52  46  49  45  51  48  98]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

**quantize.imatrix.file:**
```jinja
[103 114  97 110 105 116 101  45  52  46  49  45  51  48  98  45  71  71
  85  70  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116 104
  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95 103 114  97 110 105 116 101  45  52  46  49  45  51  48  98  46
 116 120 116]
```

---

## granite-4.1-8b-UD_Q8_K_XL.gguf
**Pfad:** `IBM\granite-4.1-8b-UD_Q8_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[363]` |
| `GGUF.kv_count` | `[45]` |
| `general.architecture` | `[103 114  97 110 105 116 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 71 114  97 110 105 116 101  45  52  46  49  45  56  66]` |
| `general.basename` | `[ 71 114  97 110 105 116 101  45  52  46  49  45  56  66]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[56 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71 114  97 110 105 116 101  32  52  46  49  32  56  98]` |
| `general.base_model.0.organization` | `[ 73  98 109  32  71 114  97 110 105 116 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[103 114  97 110 105 116 101  45  52  46  49]` |
| `granite.block_count` | `[40]` |
| `granite.context_length` | `[131072]` |
| `granite.embedding_length` | `[4096]` |
| `granite.feed_forward_length` | `[12800]` |
| `granite.attention.head_count` | `[32]` |
| `granite.attention.head_count_kv` | `[8]` |
| `granite.rope.freq_base` | `[1.e+07]` |
| `granite.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `granite.vocab_size` | `[100352]` |
| `granite.rope.dimension_count` | `[128]` |
| `granite.attention.scale` | `[0.0078125]` |
| `granite.embedding_scale` | `[12.]` |
| `granite.residual_scale` | `[0.22]` |
| `granite.logit_scale` | `[16.]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[100  98 114 120]` |
| `tokenizer.ggml.tokens` | `[ 60 124 117 110 117 115 101 100  95  56  50 124  62]` |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[196 160  67 111 110  32 118 101 121 111 114]` |
| `tokenizer.ggml.bos_token_id` | `[100257]` |
| `tokenizer.ggml.eos_token_id` | `[100257]` |
| `tokenizer.ggml.unknown_token_id` | `[100269]` |
| `tokenizer.ggml.padding_token_id` | `[100256]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.add_space_prefix` | `[False]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[280]` |
| `quantize.imatrix.chunks_count` | `[209]` |

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 105  98 109  45 103 114  97 110 105 116 101  47 103
 114  97 110 105 116 101  45  52  46  49  45  56  98]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

**quantize.imatrix.file:**
```jinja
[103 114  97 110 105 116 101  45  52  46  49  45  56  98  45  71  71  85
  70  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46
 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95 103 114  97 110 105 116 101  45  52  46  49  45  56  98  46 116
 120 116]
```

---

## Ling-3.0-flash-AD-IQ3_M-00001-of-00002.gguf
**Pfad:** `inclusionAI\Ling-3.0-flash-AD-IQ3_M-00001-of-00002.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[682]` |
| `GGUF.kv_count` | `[58]` |
| `general.architecture` | `[ 98  97 105 108 105 110 103 109 111 101  51]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 76 105 110 103  32  51  46  48  32  70 108  97 115 104]` |
| `general.size_label` | `[ 53  49  50 120  51  46  57  66]` |
| `general.license` | `[109 105 116]` |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `bailingmoe3.block_count` | `[42]` |
| `bailingmoe3.context_length` | `[131072]` |
| `bailingmoe3.embedding_length` | `[2560]` |
| `bailingmoe3.feed_forward_length` | `[6144]` |
| `bailingmoe3.attention.head_count` | `[32]` |
| `bailingmoe3.attention.head_count_kv` | `[1]` |
| `bailingmoe3.rope.freq_base` | `[6.e+06]` |
| `bailingmoe3.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `bailingmoe3.expert_count` | `[512]` |
| `bailingmoe3.expert_used_count` | `[8]` |
| `bailingmoe3.expert_group_count` | `[8]` |
| `bailingmoe3.expert_group_used_count` | `[4]` |
| `bailingmoe3.expert_gating_func` | `[2]` |
| `bailingmoe3.attention.key_length` | `[576]` |
| `bailingmoe3.attention.value_length` | `[512]` |
| `bailingmoe3.vocab_size` | `[157184]` |
| `bailingmoe3.ssm.conv_kernel` | `[4]` |
| `bailingmoe3.kda.head_dim` | `[128]` |
| `bailingmoe3.attention.kv_lora_rank` | `[512]` |
| `bailingmoe3.attention.key_length_mla` | `[192]` |
| `bailingmoe3.attention.value_length_mla` | `[128]` |
| `bailingmoe3.rope.dimension_count` | `[64]` |
| `bailingmoe3.leading_dense_block_count` | `[2]` |
| `bailingmoe3.expert_feed_forward_length` | `[768]` |
| `bailingmoe3.expert_shared_feed_forward_length` | `[768]` |
| `bailingmoe3.expert_shared_count` | `[1]` |
| `bailingmoe3.expert_weights_scale` | `[2.5]` |
| `bailingmoe3.expert_weights_norm` | `[ True]` |
| `bailingmoe3.nextn_predict_layers` | `[1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[ 98  97 105 108 105 110 103 109 111 101  50]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 53 55 49 56 51 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[196 160  97 110  32 108  97 121]` |
| `tokenizer.ggml.bos_token_id` | `[156891]` |
| `tokenizer.ggml.eos_token_id` | `[156895]` |
| `tokenizer.ggml.padding_token_id` | `[156892]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `bailingmoe3.kda.gate_lower_bound` | `[-5.]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[27]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[573]` |
| `quantize.imatrix.chunks_count` | `[522]` |
| `split.no` | `[0]` |
| `split.tensors.count` | `[917]` |
| `split.count` | `[2]` |
| `bailingmoe3.swiglu_clamp_exp` | `[4.]` |
| `bailingmoe3.swiglu_clamp_shexp` | `[7.]` |

**tokenizer.chat_template:**
```jinja
[123  35  45 ...  32  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 47 114 111 111 116  47 119 111 114 107  47 108 105 110 103  51  45 105
 109  97 116 114 105 120  45  98 102  49  54  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[ 47 114 111 111 116  47 119 111 114 107  47  99  97 108 105  98  47  99
  97 108 105  98  95 116 114  97 105 110  46 116 120 116]
```

---

## Ling-3.0-flash-AD-IQ3_M-00002-of-00002.gguf
**Pfad:** `inclusionAI\Ling-3.0-flash-AD-IQ3_M-00002-of-00002.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[235]` |
| `GGUF.kv_count` | `[3]` |
| `split.no` | `[1]` |
| `split.tensors.count` | `[917]` |
| `split.count` | `[2]` |

---

## Agents-A1-mmproj.gguf
**Pfad:** `InternScience\Agents-A1-mmproj.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[23]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | `[ 65 103 101 110 116 115  32  65  49]` |
| `general.size_label` | `[52 52 55 77]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `general.file_type` | `[1]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[2048]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

---

## Agents-A1-Q4_K_M.gguf
**Pfad:** `InternScience\Agents-A1-Q4_K_M.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[733]` |
| `GGUF.kv_count` | `[38]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 65 103 101 110 116 115  32  65  49]` |
| `general.size_label` | `[ 50  53  54 120  50  46  54  66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `qwen35moe.block_count` | `[40]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## Agents-A1-Q8_0.gguf
**Pfad:** `InternScience\Agents-A1-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[733]` |
| `GGUF.kv_count` | `[38]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 65 103 101 110 116 115  32  65  49]` |
| `general.size_label` | `[ 50  53  54 120  50  46  54  66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `qwen35moe.block_count` | `[40]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## dflash-Muse-Glimmer-30B-kquant.gguf
**Pfad:** `Meta\dflash-Muse-Glimmer-30B-kquant.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[58]` |
| `GGUF.kv_count` | `[33]` |
| `general.architecture` | `[100 102 108  97 115 104]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 72 102  95  77 117 115 101 103 108 105 109 109 101 114]` |
| `general.size_label` | `[50 46 54 66]` |
| `dflash.block_count` | `[5]` |
| `dflash.context_length` | `[131072]` |
| `dflash.embedding_length` | `[6656]` |
| `dflash.feed_forward_length` | `[19968]` |
| `dflash.attention.head_count` | `[32]` |
| `dflash.attention.head_count_kv` | `[8]` |
| `dflash.rope.freq_base` | `[500000.]` |
| `dflash.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `dflash.attention.key_length` | `[128]` |
| `dflash.attention.value_length` | `[128]` |
| `dflash.block_size` | `[16]` |
| `dflash.target_layers` | `[50]` |
| `dflash.attention.sliding_window` | `[2048]` |
| `dflash.attention.sliding_window_pattern` | `[ True]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[108 108  97 109  97  52]` |
| `tokenizer.ggml.tokens` | *Siehe Code-Block unten* |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[ 40 103  32 101]` |
| `tokenizer.ggml.bos_token_id` | `[200000]` |
| `tokenizer.ggml.eos_token_id` | `[200001]` |
| `tokenizer.ggml.padding_token_id` | `[200018]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_sep_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eot_token_id` | `[200008]` |
| `tokenizer.ggml.mask_token_id` | `[201818]` |
| `general.file_type` | `[15]` |

**tokenizer.ggml.tokens:**
```jinja
[ 60 124 114 101 115 101 114 118 101 100  95 115 112 101  99 105  97 108
  95 116 111 107 101 110  95  50  48  52  55 124  62]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  45  37 125]
```

---

## mmproj-Muse-Glimmer-30B-BF16.gguf
**Pfad:** `Meta\mmproj-Muse-Glimmer-30B-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[809]` |
| `GGUF.kv_count` | `[19]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | `[ 77 117 115 101  45  71 108 105 109 109 101 114  45  51  48  66]` |
| `general.size_label` | `[49 46 57 66]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[6656]` |
| `clip.vision.image_size` | `[896]` |
| `clip.vision.patch_size` | `[14]` |
| `clip.vision.embedding_length` | `[1536]` |
| `clip.vision.feed_forward_length` | `[8960]` |
| `clip.vision.block_count` | `[50]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[109 117 115 101  45 103 108 105 109 109 101 114]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-05]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `general.quantization_version` | `[2]` |

---

## Muse-Glimmer-30B-UD-Q5_K_XL.gguf
**Pfad:** `Meta\Muse-Glimmer-30B-UD-Q5_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[731]` |
| `GGUF.kv_count` | `[36]` |
| `general.architecture` | `[109 117 115 101  45 103 108 105 109 109 101 114]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 77 117 115 101  45  71 108 105 109 109 101 114  45  51  48  66]` |
| `general.size_label` | `[50 56 66]` |
| `muse-glimmer.block_count` | `[52]` |
| `muse-glimmer.context_length` | `[131072]` |
| `muse-glimmer.embedding_length` | `[6656]` |
| `muse-glimmer.feed_forward_length` | `[19968]` |
| `muse-glimmer.attention.head_count` | `[32]` |
| `muse-glimmer.attention.head_count_kv` | `[2]` |
| `muse-glimmer.rope.freq_base` | `[500000.]` |
| `muse-glimmer.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `muse-glimmer.attention.key_length` | `[128]` |
| `muse-glimmer.attention.value_length` | `[128]` |
| `muse-glimmer.final_logit_softcapping` | `[20.]` |
| `muse-glimmer.logit_scale` | `[0.19611613]` |
| `muse-glimmer.attention.sliding_window` | `[2048]` |
| `muse-glimmer.attention.sliding_window_pattern` | `[4]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[108 108  97 109  97  52]` |
| `tokenizer.ggml.tokens` | *Siehe Code-Block unten* |
| `tokenizer.ggml.token_type` | `[3]` |
| `tokenizer.ggml.merges` | `[ 40 103  32 101]` |
| `tokenizer.ggml.bos_token_id` | `[200000]` |
| `tokenizer.ggml.eos_token_id` | `[200001]` |
| `tokenizer.ggml.padding_token_id` | `[200018]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_sep_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eot_token_id` | `[200008]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[17]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[416]` |
| `quantize.imatrix.chunks_count` | `[166]` |

**tokenizer.ggml.tokens:**
```jinja
[ 60 124 114 101 115 101 114 118 101 100  95 115 112 101  99 105  97 108
  95 116 111 107 101 110  95  50  48  52  55 124  62]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  45  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 77 117 115 101  45  71 108 105 109 109 101 114  45 105 109  97 116 114
 105 120  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[ 77 117 115 101  45  71 108 105 109 109 101 114  45  99  97 108 105  98
 114  97 116 105 111 110  46 116 120 116]
```

---

## fastcontext-1.0-4b-rl-q8_0.gguf
**Pfad:** `Microsoft\fastcontext-1.0-4b-rl-q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[398]` |
| `GGUF.kv_count` | `[34]` |
| `general.architecture` | `[113 119 101 110  51]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.8]` |
| `general.sampling.temp` | `[0.7]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | `[82 76]` |
| `general.basename` | `[ 70  97 115 116  67 111 110 116 101 120 116  45  49  46  48]` |
| `general.size_label` | `[52 66]` |
| `general.license` | `[109 105 116]` |
| `general.tags` | *Siehe Code-Block unten* |
| `general.languages` | `[101 110]` |
| `qwen3.block_count` | `[36]` |
| `qwen3.context_length` | `[262144]` |
| `qwen3.embedding_length` | `[2560]` |
| `qwen3.feed_forward_length` | `[9728]` |
| `qwen3.attention.head_count` | `[32]` |
| `qwen3.attention.head_count_kv` | `[8]` |
| `qwen3.rope.freq_base` | `[5.e+06]` |
| `qwen3.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen3.attention.key_length` | `[128]` |
| `qwen3.attention.value_length` | `[128]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  50]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 53 49 57 51 53 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[195 162 194 189  32 196 185]` |
| `tokenizer.ggml.eos_token_id` | `[151645]` |
| `tokenizer.ggml.padding_token_id` | `[151643]` |
| `tokenizer.ggml.bos_token_id` | `[151643]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**general.name:**
```jinja
[ 70  97 115 116  67 111 110 116 101 120 116  32  49  46  48  32  52  66
  32  82  76]
```

**general.tags:**
```jinja
[ 82 101 112 111 115 105 116 111 114 121  32  69 120 112 108 111 114  97
 116 105 111 110]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## FastContext-1.0-4B-SFT-Q8_0.gguf
**Pfad:** `Microsoft\FastContext-1.0-4B-SFT-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[398]` |
| `GGUF.kv_count` | `[39]` |
| `general.architecture` | `[113 119 101 110  51]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.8]` |
| `general.sampling.temp` | `[0.7]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | `[83 70 84]` |
| `general.basename` | `[ 70  97 115 116  67 111 110 116 101 120 116  45  49  46  48]` |
| `general.size_label` | `[52 66]` |
| `general.license` | `[109 105 116]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.version` | `[50 53 48 55]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | *Siehe Code-Block unten* |
| `general.languages` | `[101 110]` |
| `qwen3.block_count` | `[36]` |
| `qwen3.context_length` | `[262144]` |
| `qwen3.embedding_length` | `[2560]` |
| `qwen3.feed_forward_length` | `[9728]` |
| `qwen3.attention.head_count` | `[32]` |
| `qwen3.attention.head_count_kv` | `[8]` |
| `qwen3.rope.freq_base` | `[5.e+06]` |
| `qwen3.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen3.attention.key_length` | `[128]` |
| `qwen3.attention.value_length` | `[128]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  50]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 53 49 57 51 53 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[195 162 194 189  32 196 185]` |
| `tokenizer.ggml.eos_token_id` | `[151645]` |
| `tokenizer.ggml.padding_token_id` | `[151643]` |
| `tokenizer.ggml.bos_token_id` | `[151643]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**general.name:**
```jinja
[ 70  97 115 116  67 111 110 116 101 120 116  32  49  46  48  32  52  66
  32  83  70  84]
```

**general.base_model.0.name:**
```jinja
[ 81 119 101 110  51  32  52  66  32  73 110 115 116 114 117  99 116  32
  50  53  48  55]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  45  52  66
  45  73 110 115 116 114 117  99 116  45  50  53  48  55]
```

**general.tags:**
```jinja
[ 82 101 112 111 115 105 116 111 114 121  32  69 120 112 108 111 114  97
 116 105 111 110]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## mmproj-Tess-4-27B-F16.gguf
**Pfad:** `migtissera\mmproj-Tess-4-27B-F16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[23]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | `[ 84 101 115 115  32  52  32  50  55  66]` |
| `general.finetune` | `[50 55 98]` |
| `general.basename` | `[ 84 101 115 115  45  52]` |
| `general.size_label` | `[52 54 49 77]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[5120]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

---

## mtp-Tess-4-27B-Q4_K_M.gguf
**Pfad:** `migtissera\mtp-Tess-4-27B-Q4_K_M.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[18]` |
| `GGUF.kv_count` | `[43]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 84 101 115 115  32  52  32  50  55  66]` |
| `general.finetune` | `[50 55 98]` |
| `general.basename` | `[ 84 101 115 115  45  52]` |
| `general.size_label` | `[51 46 48 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  54  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## mtp-Tess-4-27B-Q8_0.gguf
**Pfad:** `migtissera\mtp-Tess-4-27B-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[18]` |
| `GGUF.kv_count` | `[43]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 84 101 115 115  32  52  32  50  55  66]` |
| `general.finetune` | `[50 55 98]` |
| `general.basename` | `[ 84 101 115 115  45  52]` |
| `general.size_label` | `[51 46 48 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 81 119 101 110  51  46  54  32  50  55  66]` |
| `general.base_model.0.organization` | `[ 81 119 101 110]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[105 109  97 103 101  45 116 101 120 116  45 116 111  45 116 101 120 116]` |
| `qwen35.block_count` | `[65]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `qwen35.nextn_predict_layers` | `[1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47  81 119 101 110  47  81 119 101 110  51  46  54  45
  50  55  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## Tess-4-27B-Q6_K.gguf
**Pfad:** `migtissera\Tess-4-27B-Q6_K.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[851]` |
| `GGUF.kv_count` | `[35]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 84 101 115 115  32  52  32  50  55  66]` |
| `general.basename` | `[ 84 101 115 115  45  52]` |
| `general.size_label` | `[50 55 66]` |
| `qwen35.block_count` | `[64]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[5120]` |
| `qwen35.feed_forward_length` | `[17408]` |
| `qwen35.attention.head_count` | `[24]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[48]` |
| `qwen35.ssm.inner_size` | `[6144]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  37 125]
```

---

## Devstral-Small-2-24B-Instruct-2512-UD_Q4_K_XL.gguf
**Pfad:** `Mistral AI\Devstral-Small 2\Devstral-Small-2-24B-Instruct-2512-UD_Q4_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[363]` |
| `GGUF.kv_count` | `[54]` |
| `general.architecture` | `[109 105 115 116 114  97 108  51]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.version` | `[50 53 49 50]` |
| `general.finetune` | `[ 73 110 115 116 114 117  99 116]` |
| `general.basename` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 52 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.version` | `[50 53 49 50]` |
| `general.base_model.0.organization` | `[ 77 105 115 116 114  97 108  97 105]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `mistral3.block_count` | `[40]` |
| `mistral3.context_length` | `[393216]` |
| `mistral3.embedding_length` | `[5120]` |
| `mistral3.feed_forward_length` | `[32768]` |
| `mistral3.attention.head_count` | `[32]` |
| `mistral3.attention.head_count_kv` | `[8]` |
| `mistral3.rope.freq_base` | `[1.e+08]` |
| `mistral3.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `mistral3.attention.key_length` | `[128]` |
| `mistral3.attention.value_length` | `[128]` |
| `mistral3.rope.dimension_count` | `[128]` |
| `mistral3.rope.scaling.type` | `[121  97 114 110]` |
| `mistral3.rope.scaling.factor` | `[48.]` |
| `mistral3.rope.scaling.yarn_beta_fast` | `[32.]` |
| `mistral3.rope.scaling.yarn_beta_slow` | `[1.]` |
| `mistral3.rope.scaling.yarn_log_multiplier` | `[1.]` |
| `mistral3.rope.scaling.original_context_length` | `[8192]` |
| `mistral3.attention.temperature_scale` | `[0.1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[116 101 107 107 101 110]` |
| `tokenizer.ggml.merges` | `[195 165 196 178 196 176  32 195 166 194 177 196 171 195 164 194 185 194  166]` |
| `tokenizer.ggml.bos_token_id` | `[1]` |
| `tokenizer.ggml.eos_token_id` | `[2]` |
| `tokenizer.ggml.unknown_token_id` | `[0]` |
| `tokenizer.ggml.padding_token_id` | `[11]` |
| `tokenizer.ggml.tokens` | `[195 165 196 178 196 176 195 166 194 177 196 171 195 164 194 185 194 166]` |
| `tokenizer.ggml.scores` | `[0]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `mistral3.vocab_size` | `[131072]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[280]` |
| `quantize.imatrix.chunks_count` | `[75]` |

**general.name:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**general.basename:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.name:**
```jinja
[ 68 101 118 115 116 114  97 108  32  83 109  97 108 108  32  50  32  50
  52  66  32  73 110 115 116 114 117  99 116  32  50  53  49  50]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  68 101 118
 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50  52  66  45
  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**tokenizer.chat_template:**
```jinja
[123  35  45 ...  32  35 125]
```

**quantize.imatrix.file:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50  45  71
  71  85  70  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116
 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50
  45  50  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50
  46 116 120 116]
```

---

## Devstral-Small-2-24B-Instruct-2512-UD_Q8_K_XL.gguf
**Pfad:** `Mistral AI\Devstral-Small 2\Devstral-Small-2-24B-Instruct-2512-UD_Q8_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[363]` |
| `GGUF.kv_count` | `[54]` |
| `general.architecture` | `[109 105 115 116 114  97 108  51]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.version` | `[50 53 49 50]` |
| `general.finetune` | `[ 73 110 115 116 114 117  99 116]` |
| `general.basename` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 52 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.version` | `[50 53 49 50]` |
| `general.base_model.0.organization` | `[ 77 105 115 116 114  97 108  97 105]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `mistral3.block_count` | `[40]` |
| `mistral3.context_length` | `[393216]` |
| `mistral3.embedding_length` | `[5120]` |
| `mistral3.feed_forward_length` | `[32768]` |
| `mistral3.attention.head_count` | `[32]` |
| `mistral3.attention.head_count_kv` | `[8]` |
| `mistral3.rope.freq_base` | `[1.e+08]` |
| `mistral3.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `mistral3.attention.key_length` | `[128]` |
| `mistral3.attention.value_length` | `[128]` |
| `mistral3.rope.dimension_count` | `[128]` |
| `mistral3.rope.scaling.type` | `[121  97 114 110]` |
| `mistral3.rope.scaling.factor` | `[48.]` |
| `mistral3.rope.scaling.yarn_beta_fast` | `[32.]` |
| `mistral3.rope.scaling.yarn_beta_slow` | `[1.]` |
| `mistral3.rope.scaling.yarn_log_multiplier` | `[1.]` |
| `mistral3.rope.scaling.original_context_length` | `[8192]` |
| `mistral3.attention.temperature_scale` | `[0.1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[116 101 107 107 101 110]` |
| `tokenizer.ggml.merges` | `[195 165 196 178 196 176  32 195 166 194 177 196 171 195 164 194 185 194  166]` |
| `tokenizer.ggml.bos_token_id` | `[1]` |
| `tokenizer.ggml.eos_token_id` | `[2]` |
| `tokenizer.ggml.unknown_token_id` | `[0]` |
| `tokenizer.ggml.padding_token_id` | `[11]` |
| `tokenizer.ggml.tokens` | `[195 165 196 178 196 176 195 166 194 177 196 171 195 164 194 185 194 166]` |
| `tokenizer.ggml.scores` | `[0]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `mistral3.vocab_size` | `[131072]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[280]` |
| `quantize.imatrix.chunks_count` | `[75]` |

**general.name:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**general.basename:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.name:**
```jinja
[ 68 101 118 115 116 114  97 108  32  83 109  97 108 108  32  50  32  50
  52  66  32  73 110 115 116 114 117  99 116  32  50  53  49  50]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  68 101 118
 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50  52  66  45
  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**tokenizer.chat_template:**
```jinja
[123  35  45 ...  32  35 125]
```

**quantize.imatrix.file:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50  45  71
  71  85  70  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116
 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50
  45  50  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50
  46 116 120 116]
```

---

## mmproj-Devstral-Small-2-24B-Instruct-2512-BF16.gguf
**Pfad:** `Mistral AI\Devstral-Small 2\mmproj-Devstral-Small-2-24B-Instruct-2512-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[223]` |
| `GGUF.kv_count` | `[33]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.version` | `[50 45 50 53 49 50]` |
| `general.finetune` | `[ 50  52  98  45  73 110 115 116 114 117  99 116]` |
| `general.basename` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[ 83 109  97 108 108]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.version` | `[50 45 50 53 49 50]` |
| `general.base_model.0.organization` | `[ 77 105 115 116 114  97 108  97 105]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[32768]` |
| `clip.vision.image_size` | `[1540]` |
| `clip.vision.patch_size` | `[14]` |
| `clip.vision.embedding_length` | `[1024]` |
| `clip.vision.feed_forward_length` | `[4096]` |
| `clip.vision.block_count` | `[24]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.40821072]` |
| `clip.vision.image_std` | `[0.2757771]` |
| `clip.projector_type` | `[112 105 120 116 114  97 108]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-05]` |
| `clip.rope.freq_base` | `[10000.]` |
| `clip.use_silu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `general.quantization_version` | `[2]` |

**general.name:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**general.basename:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.name:**
```jinja
[ 68 101 118 115 116 114  97 108  32  83 109  97 108 108  32  50  32  50
  52  66  32  73 110 115 116 114 117  99 116  32  50  53  49  50]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  68 101 118
 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50  52  66  45
  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

---

## mmproj-Devstral-Small-2-24B-Instruct-2512-UD_F32.gguf
**Pfad:** `Mistral AI\Devstral-Small 2\mmproj-Devstral-Small-2-24B-Instruct-2512-UD_F32.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[223]` |
| `GGUF.kv_count` | `[33]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.version` | `[50 45 50 53 49 50]` |
| `general.finetune` | `[ 50  52  98  45  73 110 115 116 114 117  99 116]` |
| `general.basename` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[ 83 109  97 108 108]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.version` | `[50 45 50 53 49 50]` |
| `general.base_model.0.organization` | `[ 77 105 115 116 114  97 108  97 105]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `general.file_type` | `[0]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[32768]` |
| `clip.vision.image_size` | `[1540]` |
| `clip.vision.patch_size` | `[14]` |
| `clip.vision.embedding_length` | `[1024]` |
| `clip.vision.feed_forward_length` | `[4096]` |
| `clip.vision.block_count` | `[24]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.40821072]` |
| `clip.vision.image_std` | `[0.2757771]` |
| `clip.projector_type` | `[112 105 120 116 114  97 108]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-05]` |
| `clip.rope.freq_base` | `[10000.]` |
| `clip.use_silu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `general.quantization_version` | `[2]` |

**general.name:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**general.basename:**
```jinja
[ 68 101 118 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50
  52  66  45  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.name:**
```jinja
[ 68 101 118 115 116 114  97 108  32  83 109  97 108 108  32  50  32  50
  52  66  32  73 110 115 116 114 117  99 116  32  50  53  49  50]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  68 101 118
 115 116 114  97 108  45  83 109  97 108 108  45  50  45  50  52  66  45
  73 110 115 116 114 117  99 116  45  50  53  49  50]
```

---

## Ministral-3-14B-Instruct-2512-BF16-mmproj.gguf
**Pfad:** `Mistral AI\Ministral 3\Ministral-3-14B-Instruct-2512-BF16-mmproj.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[223]` |
| `GGUF.kv_count` | `[36]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.version` | `[51 46 48]` |
| `general.organization` | `[ 77 105 115 116 114  97 108  32  65  73]` |
| `general.finetune` | `[50 53 49 50]` |
| `general.basename` | `[ 77 105 110 105 115 116 114  97 108]` |
| `general.description` | *Siehe Code-Block unten* |
| `general.size_label` | `[49 52 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.url` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.version` | `[50 53 49 50]` |
| `general.base_model.0.organization` | `[ 77 105 115 116 114  97 108  97 105]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[109 105 115 116 114  97 108  45  99 111 109 109 111 110]` |
| `general.languages` | `[ 97 114]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[16384]` |
| `clip.vision.image_size` | `[1540]` |
| `clip.vision.patch_size` | `[14]` |
| `clip.vision.embedding_length` | `[1024]` |
| `clip.vision.feed_forward_length` | `[4096]` |
| `clip.vision.block_count` | `[24]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.40821072]` |
| `clip.vision.image_std` | `[0.2757771]` |
| `clip.projector_type` | `[112 105 120 116 114  97 108]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-05]` |
| `clip.rope.freq_base` | `[10000.]` |
| `clip.use_silu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `general.quantization_version` | `[2]` |

**general.name:**
```jinja
[109 105 110 105 115 116 114  97 108  45  49  52  66  45  73 110 115 116
 114 117  99 116  45  50  53  49  50]
```

**general.description:**
```jinja
[ 84 104 101  32  77 105 110 105 115 116 114  97 108  32  51  32 102  97
 109 105 108 121  32 105 115  32 100 101 115 105 103 110 101 100  32 102
 111 114  32 101 100 103 101  32 100 101 112 108 111 121 109 101 110 116
  44  32  99  97 112  97  98 108 101  32 111 102  32 114 117 110 110 105
 110 103  32 111 110  32  97  32 119 105 100 101  32 114  97 110 103 101
  32 111 102  32 104  97 114 100 119  97 114 101  46  32  84 104 105 115
  32 109 111 100 101 108  32 105 115  32 116 104 101  32  49  52  66  32
 105 110 115 116 114 117  99 116  32 112 111 115 116  45 116 114  97 105
 110 101 100  32 118 101 114 115 105 111 110  32 105 110  32  70  80  56
  44  32 102 105 110 101  45 116 117 110 101 100  32 102 111 114  32 105
 110 115 116 114 117  99 116 105 111 110  32 116  97 115 107 115  44  32
 109  97 107 105 110 103  32 105 116  32 105 100 101  97 108  32 102 111
 114  32  99 104  97 116  32  97 110 100  32 105 110 115 116 114 117  99
 116 105 111 110  32  98  97 115 101 100  32 117 115 101  32  99  97 115
 101 115  46]
```

**general.url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  77 105 110
 105 115 116 114  97 108  45  51  45  49  52  66  45  73 110 115 116 114
 117  99 116  45  50  53  49  50  45  71  71  85  70]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  77 105 110
 105 115 116 114  97 108  45  51  45  49  52  66  45  73 110 115 116 114
 117  99 116  45  50  53  49  50  45  71  71  85  70]
```

**general.base_model.0.name:**
```jinja
[ 77 105 110 105 115 116 114  97 108  32  51  32  49  52  66  32  66  97
 115 101  32  50  53  49  50]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  77 105 110
 105 115 116 114  97 108  45  51  45  49  52  66  45  66  97 115 101  45
  50  53  49  50]
```

---

## Ministral-3-14B-Instruct-2512-Q8_0.gguf
**Pfad:** `Mistral AI\Ministral 3\Ministral-3-14B-Instruct-2512-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[363]` |
| `GGUF.kv_count` | `[53]` |
| `general.architecture` | `[109 105 115 116 114  97 108  51]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.version` | `[51 46 48]` |
| `general.organization` | `[ 77 105 115 116 114  97 108  32  65  73]` |
| `general.finetune` | `[50 53 49 50]` |
| `general.basename` | `[ 77 105 110 105 115 116 114  97 108]` |
| `general.description` | *Siehe Code-Block unten* |
| `general.size_label` | `[49 52 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.url` | *Siehe Code-Block unten* |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.version` | `[50 53 49 50]` |
| `general.base_model.0.organization` | `[ 77 105 115 116 114  97 108  97 105]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[109 105 115 116 114  97 108  45  99 111 109 109 111 110]` |
| `general.languages` | `[ 97 114]` |
| `mistral3.block_count` | `[40]` |
| `mistral3.context_length` | `[262144]` |
| `mistral3.embedding_length` | `[5120]` |
| `mistral3.feed_forward_length` | `[16384]` |
| `mistral3.attention.head_count` | `[32]` |
| `mistral3.attention.head_count_kv` | `[8]` |
| `mistral3.rope.freq_base` | `[1.e+09]` |
| `mistral3.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `mistral3.attention.key_length` | `[128]` |
| `mistral3.attention.value_length` | `[128]` |
| `mistral3.rope.dimension_count` | `[128]` |
| `mistral3.rope.scaling.type` | `[121  97 114 110]` |
| `mistral3.rope.scaling.factor` | `[16.]` |
| `mistral3.rope.scaling.yarn_beta_fast` | `[32.]` |
| `mistral3.rope.scaling.yarn_beta_slow` | `[1.]` |
| `mistral3.rope.scaling.yarn_log_multiplier` | `[1.]` |
| `mistral3.rope.scaling.original_context_length` | `[16384]` |
| `mistral3.attention.temperature_scale` | `[0.1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[116 101 107 107 101 110]` |
| `tokenizer.ggml.merges` | `[195 165 196 178 196 176  32 195 166 194 177 196 171 195 164 194 185 194  166]` |
| `tokenizer.ggml.bos_token_id` | `[1]` |
| `tokenizer.ggml.eos_token_id` | `[2]` |
| `tokenizer.ggml.unknown_token_id` | `[0]` |
| `tokenizer.ggml.padding_token_id` | `[11]` |
| `tokenizer.ggml.tokens` | `[195 165 196 178 196 176 195 166 194 177 196 171 195 164 194 185 194 166]` |
| `tokenizer.ggml.scores` | `[0]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `mistral3.vocab_size` | `[131072]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**general.name:**
```jinja
[109 105 110 105 115 116 114  97 108  45  49  52  66  45  73 110 115 116
 114 117  99 116  45  50  53  49  50]
```

**general.description:**
```jinja
[ 84 104 101  32  77 105 110 105 115 116 114  97 108  32  51  32 102  97
 109 105 108 121  32 105 115  32 100 101 115 105 103 110 101 100  32 102
 111 114  32 101 100 103 101  32 100 101 112 108 111 121 109 101 110 116
  44  32  99  97 112  97  98 108 101  32 111 102  32 114 117 110 110 105
 110 103  32 111 110  32  97  32 119 105 100 101  32 114  97 110 103 101
  32 111 102  32 104  97 114 100 119  97 114 101  46  32  84 104 105 115
  32 109 111 100 101 108  32 105 115  32 116 104 101  32  49  52  66  32
 105 110 115 116 114 117  99 116  32 112 111 115 116  45 116 114  97 105
 110 101 100  32 118 101 114 115 105 111 110  32 105 110  32  70  80  56
  44  32 102 105 110 101  45 116 117 110 101 100  32 102 111 114  32 105
 110 115 116 114 117  99 116 105 111 110  32 116  97 115 107 115  44  32
 109  97 107 105 110 103  32 105 116  32 105 100 101  97 108  32 102 111
 114  32  99 104  97 116  32  97 110 100  32 105 110 115 116 114 117  99
 116 105 111 110  32  98  97 115 101 100  32 117 115 101  32  99  97 115
 101 115  46]
```

**general.url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  77 105 110
 105 115 116 114  97 108  45  51  45  49  52  66  45  73 110 115 116 114
 117  99 116  45  50  53  49  50  45  71  71  85  70]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  77 105 110
 105 115 116 114  97 108  45  51  45  49  52  66  45  73 110 115 116 114
 117  99 116  45  50  53  49  50  45  71  71  85  70]
```

**general.base_model.0.name:**
```jinja
[ 77 105 110 105 115 116 114  97 108  32  51  32  49  52  66  32  66  97
 115 101  32  50  53  49  50]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  77 105 110
 105 115 116 114  97 108  45  51  45  49  52  66  45  66  97 115 101  45
  50  53  49  50]
```

**tokenizer.chat_template:**
```jinja
[123  35  45 ...  37 125  10]
```

---

## Mistral-Medium-3.5-128B-UD_Q3_K_XL-00001-of-00003.gguf
**Pfad:** `Mistral AI\Mistral-Medium\Mistral-Medium-3.5-128B-UD_Q3_K_XL-00001-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[0]` |
| `GGUF.kv_count` | `[52]` |
| `general.architecture` | `[109 105 115 116 114  97 108  51]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.basename` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[49 50 56 66]` |
| `general.license` | `[111 116 104 101 114]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.organization` | `[ 77 105 115 116 114  97 108  97 105]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `general.languages` | `[ 98 110]` |
| `mistral3.block_count` | `[88]` |
| `mistral3.context_length` | `[262144]` |
| `mistral3.embedding_length` | `[12288]` |
| `mistral3.feed_forward_length` | `[28672]` |
| `mistral3.attention.head_count` | `[96]` |
| `mistral3.attention.head_count_kv` | `[8]` |
| `mistral3.rope.scaling.type` | `[121  97 114 110]` |
| `mistral3.rope.scaling.factor` | `[64.]` |
| `mistral3.rope.scaling.original_context_length` | `[4096]` |
| `mistral3.rope.scaling.yarn_beta_fast` | `[4.]` |
| `mistral3.rope.scaling.yarn_beta_slow` | `[1.]` |
| `mistral3.rope.freq_base` | `[1.e+06]` |
| `mistral3.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `mistral3.attention.key_length` | `[128]` |
| `mistral3.attention.value_length` | `[128]` |
| `mistral3.vocab_size` | `[131072]` |
| `mistral3.rope.dimension_count` | `[128]` |
| `mistral3.rope.scaling.yarn_log_multiplier` | `[0.]` |
| `mistral3.attention.temperature_scale` | `[0.]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[112 105 120 116 114  97 108]` |
| `tokenizer.ggml.tokens` | `[195 165 196 178 196 176 195 166 194 177 196 171 195 164 194 185 194 166]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[195 165 196 178 196 176  32 195 166 194 177 196 171 195 164 194 185 194  166]` |
| `tokenizer.ggml.bos_token_id` | `[1]` |
| `tokenizer.ggml.eos_token_id` | `[2]` |
| `tokenizer.ggml.unknown_token_id` | `[0]` |
| `tokenizer.ggml.padding_token_id` | `[11]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[12]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[616]` |
| `quantize.imatrix.chunks_count` | `[123]` |
| `split.no` | `[0]` |
| `split.tensors.count` | `[795]` |
| `split.count` | `[3]` |

**general.name:**
```jinja
[ 77 105 115 116 114  97 108  45  77 101 100 105 117 109  45  51  46  53
  45  49  50  56  66]
```

**general.basename:**
```jinja
[ 77 105 115 116 114  97 108  45  77 101 100 105 117 109  45  51  46  53
  45  49  50  56  66]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.name:**
```jinja
[ 77 105 115 116 114  97 108  32  77 101 100 105 117 109  32  51  46  53
  32  49  50  56  66]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  77 105 115
 116 114  97 108  45  77 101 100 105 117 109  45  51  46  53  45  49  50
  56  66]
```

**tokenizer.chat_template:**
```jinja
[123  35  45 ...  37 125  10]
```

**quantize.imatrix.file:**
```jinja
[ 77 105 115 116 114  97 108  45  77 101 100 105 117 109  45  51  46  53
  45  49  50  56  66  45  71  71  85  70  47 105 109  97 116 114 105 120
  95 117 110 115 108 111 116 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  77 105 115 116 114  97 108  45  77 101 100 105 117 109  45  51
  46  53  45  49  50  56  66  46 116 120 116]
```

---

## Mistral-Medium-3.5-128B-UD_Q3_K_XL-00002-of-00003.gguf
**Pfad:** `Mistral AI\Mistral-Medium\Mistral-Medium-3.5-128B-UD_Q3_K_XL-00002-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[638]` |
| `GGUF.kv_count` | `[3]` |
| `split.no` | `[1]` |
| `split.tensors.count` | `[795]` |
| `split.count` | `[3]` |

---

## Mistral-Medium-3.5-128B-UD_Q3_K_XL-00003-of-00003.gguf
**Pfad:** `Mistral AI\Mistral-Medium\Mistral-Medium-3.5-128B-UD_Q3_K_XL-00003-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[157]` |
| `GGUF.kv_count` | `[3]` |
| `split.no` | `[2]` |
| `split.tensors.count` | `[795]` |
| `split.count` | `[3]` |

---

## Mistral-Small-4-119B-2603-UD_Q3_K_XL-00001-of-00003.gguf
**Pfad:** `Mistral AI\Mistral-Smal 4\Mistral-Small-4-119B-2603-UD_Q3_K_XL-00001-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[0]` |
| `GGUF.kv_count` | `[66]` |
| `general.architecture` | `[109 105 115 116 114  97 108  52]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.version` | `[50 54 48 51]` |
| `general.basename` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[49 49 57 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.version` | `[50 54 48 51]` |
| `general.base_model.0.organization` | `[ 77 105 115 116 114  97 108  97 105]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `general.languages` | `[122 104]` |
| `mistral4.block_count` | `[36]` |
| `mistral4.context_length` | `[1048576]` |
| `mistral4.embedding_length` | `[4096]` |
| `mistral4.feed_forward_length` | `[12288]` |
| `mistral4.attention.head_count` | `[32]` |
| `mistral4.attention.head_count_kv` | `[1]` |
| `mistral4.rope.scaling.type` | `[121  97 114 110]` |
| `mistral4.rope.scaling.factor` | `[128.]` |
| `mistral4.rope.scaling.original_context_length` | `[8192]` |
| `mistral4.rope.scaling.yarn_beta_fast` | `[32.]` |
| `mistral4.rope.scaling.yarn_beta_slow` | `[1.]` |
| `mistral4.rope.freq_base` | `[10000.]` |
| `mistral4.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `mistral4.expert_used_count` | `[4]` |
| `mistral4.expert_group_count` | `[1]` |
| `mistral4.expert_group_used_count` | `[1]` |
| `mistral4.attention.key_length` | `[320]` |
| `mistral4.attention.value_length` | `[256]` |
| `mistral4.leading_dense_block_count` | `[0]` |
| `mistral4.vocab_size` | `[131072]` |
| `mistral4.attention.q_lora_rank` | `[1024]` |
| `mistral4.attention.kv_lora_rank` | `[256]` |
| `mistral4.attention.key_length_mla` | `[128]` |
| `mistral4.attention.value_length_mla` | `[128]` |
| `mistral4.expert_feed_forward_length` | `[2048]` |
| `mistral4.expert_count` | `[128]` |
| `mistral4.expert_shared_count` | `[1]` |
| `mistral4.expert_weights_scale` | `[1.]` |
| `mistral4.expert_weights_norm` | `[ True]` |
| `mistral4.rope.dimension_count` | `[64]` |
| `mistral4.rope.scaling.yarn_log_multiplier` | `[0.1]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[112 105 120 116 114  97 108]` |
| `tokenizer.ggml.tokens` | `[195 165 196 178 196 176 195 166 194 177 196 171 195 164 194 185 194 166]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[195 165 196 178 196 176  32 195 166 194 177 196 171 195 164 194 185 194  166]` |
| `tokenizer.ggml.bos_token_id` | `[1]` |
| `tokenizer.ggml.eos_token_id` | `[2]` |
| `tokenizer.ggml.unknown_token_id` | `[0]` |
| `tokenizer.ggml.padding_token_id` | `[11]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[12]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[432]` |
| `quantize.imatrix.chunks_count` | `[102]` |
| `split.no` | `[0]` |
| `split.tensors.count` | `[579]` |
| `split.count` | `[3]` |

**general.name:**
```jinja
[ 77 105 115 116 114  97 108  45  83 109  97 108 108  45  52  45  49  49
  57  66  45  50  54  48  51]
```

**general.basename:**
```jinja
[ 77 105 115 116 114  97 108  45  83 109  97 108 108  45  52  45  49  49
  57  66  45  50  54  48  51]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.name:**
```jinja
[ 77 105 115 116 114  97 108  32  83 109  97 108 108  32  52  32  49  49
  57  66  32  50  54  48  51]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  77 105 115
 116 114  97 108  45  83 109  97 108 108  45  52  45  49  49  57  66  45
  50  54  48  51]
```

**tokenizer.chat_template:**
```jinja
[123  35  45 ...  32  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 77 105 115 116 114  97 108  45  83 109  97 108 108  45  52  45  49  49
  57  66  45  50  54  48  51  45  71  71  85  70  47 105 109  97 116 114
 105 120  95 117 110 115 108 111 116 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  77 105 115 116 114  97 108  45  83 109  97 108 108  45  52  45
  49  49  57  66  45  50  54  48  51  46 116 120 116]
```

---

## Mistral-Small-4-119B-2603-UD_Q3_K_XL-00002-of-00003.gguf
**Pfad:** `Mistral AI\Mistral-Smal 4\Mistral-Small-4-119B-2603-UD_Q3_K_XL-00002-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[528]` |
| `GGUF.kv_count` | `[3]` |
| `split.no` | `[1]` |
| `split.tensors.count` | `[579]` |
| `split.count` | `[3]` |

---

## Mistral-Small-4-119B-2603-UD_Q3_K_XL-00003-of-00003.gguf
**Pfad:** `Mistral AI\Mistral-Smal 4\Mistral-Small-4-119B-2603-UD_Q3_K_XL-00003-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[51]` |
| `GGUF.kv_count` | `[3]` |
| `split.no` | `[2]` |
| `split.tensors.count` | `[579]` |
| `split.count` | `[3]` |

---

## mmproj-Mistral-Small-4-119B-2603-UD_F32.gguf
**Pfad:** `Mistral AI\Mistral-Smal 4\mmproj-Mistral-Small-4-119B-2603-UD_F32.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[223]` |
| `GGUF.kv_count` | `[33]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.version` | `[52 45 50 54 48 51]` |
| `general.finetune` | `[49 49 57 98]` |
| `general.basename` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[ 83 109  97 108 108]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | *Siehe Code-Block unten* |
| `general.base_model.0.version` | `[52 45 50 54 48 51]` |
| `general.base_model.0.organization` | `[ 77 105 115 116 114  97 108  97 105]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[117 110 115 108 111 116 104]` |
| `general.languages` | `[122 104]` |
| `general.file_type` | `[0]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[4096]` |
| `clip.vision.image_size` | `[1540]` |
| `clip.vision.patch_size` | `[14]` |
| `clip.vision.embedding_length` | `[1024]` |
| `clip.vision.feed_forward_length` | `[4096]` |
| `clip.vision.block_count` | `[24]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.40821072]` |
| `clip.vision.image_std` | `[0.2757771]` |
| `clip.projector_type` | `[112 105 120 116 114  97 108]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-05]` |
| `clip.use_silu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `general.quantization_version` | `[2]` |

**general.name:**
```jinja
[ 77 105 115 116 114  97 108  45  83 109  97 108 108  45  52  45  49  49
  57  66  45  50  54  48  51]
```

**general.basename:**
```jinja
[ 77 105 115 116 114  97 108  45  83 109  97 108 108  45  52  45  49  49
  57  66  45  50  54  48  51]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.name:**
```jinja
[ 77 105 115 116 114  97 108  32  83 109  97 108 108  32  52  32  49  49
  57  66  32  50  54  48  51]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 109 105 115 116 114  97 108  97 105  47  77 105 115
 116 114  97 108  45  83 109  97 108 108  45  52  45  49  49  57  66  45
  50  54  48  51]
```

---

## mmproj-NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16.gguf
**Pfad:** `NVIDIA\mmproj-NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[390]` |
| `GGUF.kv_count` | `[24]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[0.6]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | `[ 51  48  98  45  82 101  97 115 111 110 105 110 103]` |
| `general.basename` | *Siehe Code-Block unten* |
| `general.size_label` | `[65 51 66]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[2688]` |
| `clip.vision.image_size` | `[512]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1280]` |
| `clip.vision.feed_forward_length` | `[5120]` |
| `clip.vision.block_count` | `[32]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.40821072]` |
| `clip.vision.image_std` | `[0.2757771]` |
| `clip.projector_type` | `[110 101 109 111 116 114 111 110  95 118  50  95 118 108]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.projector.scale_factor` | `[2]` |
| `general.quantization_version` | `[2]` |

**general.name:**
```jinja
[ 78 118 105 100 105  97  45  78 101 109 111 116 114 111 110  45  51  45
  78  97 110 111  45  79 109 110 105  45  51  48  66  45  65  51  66  45
  82 101  97 115 111 110 105 110 103]
```

**general.basename:**
```jinja
[ 78 118 105 100 105  97  45  78 101 109 111 116 114 111 110  45  51  45
  78  97 110 111  45  79 109 110 105  45  51  48  66  45  65  51  66  45
  82 101  97 115 111 110 105 110 103]
```

---

## NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD_Q6_K.gguf
**Pfad:** `NVIDIA\NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD_Q6_K.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[401]` |
| `GGUF.kv_count` | `[56]` |
| `general.architecture` | `[110 101 109 111 116 114 111 110  95 104  95 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[0.6]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.finetune` | `[ 82 101  97 115 111 110 105 110 103]` |
| `general.basename` | *Siehe Code-Block unten* |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[51 48 66 45 65 51 66]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `nemotron_h_moe.block_count` | `[52]` |
| `nemotron_h_moe.context_length` | `[1048576]` |
| `nemotron_h_moe.embedding_length` | `[2688]` |
| `nemotron_h_moe.feed_forward_length` | `[1856]` |
| `nemotron_h_moe.attention.head_count` | `[32]` |
| `nemotron_h_moe.attention.head_count_kv` | `[0]` |
| `nemotron_h_moe.rope.freq_base` | `[10000.]` |
| `nemotron_h_moe.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `nemotron_h_moe.attention.layer_norm_epsilon` | `[1.e-05]` |
| `nemotron_h_moe.expert_used_count` | `[6]` |
| `nemotron_h_moe.expert_group_count` | `[1]` |
| `nemotron_h_moe.expert_group_used_count` | `[1]` |
| `nemotron_h_moe.vocab_size` | `[131072]` |
| `nemotron_h_moe.rope.dimension_count` | `[84]` |
| `nemotron_h_moe.ssm.conv_kernel` | `[4]` |
| `nemotron_h_moe.ssm.state_size` | `[128]` |
| `nemotron_h_moe.ssm.group_count` | `[8]` |
| `nemotron_h_moe.ssm.inner_size` | `[4096]` |
| `nemotron_h_moe.ssm.time_step_rank` | `[64]` |
| `nemotron_h_moe.rope.scaling.finetuned` | `[False]` |
| `nemotron_h_moe.attention.key_length` | `[128]` |
| `nemotron_h_moe.attention.value_length` | `[128]` |
| `nemotron_h_moe.expert_feed_forward_length` | `[1856]` |
| `nemotron_h_moe.expert_shared_feed_forward_length` | `[3712]` |
| `nemotron_h_moe.expert_count` | `[128]` |
| `nemotron_h_moe.expert_shared_count` | `[1]` |
| `nemotron_h_moe.expert_weights_norm` | `[ True]` |
| `nemotron_h_moe.expert_weights_scale` | `[2.5]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[112 105 120 116 114  97 108]` |
| `tokenizer.ggml.tokens` | `[195 165 196 178 196 176 195 166 194 177 196 171 195 164 194 185 194 166]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[195 165 196 178 196 176  32 195 166 194 177 196 171 195 164 194 185 194  166]` |
| `tokenizer.ggml.bos_token_id` | `[1]` |
| `tokenizer.ggml.eos_token_id` | `[11]` |
| `tokenizer.ggml.unknown_token_id` | `[0]` |
| `tokenizer.ggml.padding_token_id` | `[999]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.ggml.add_eos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[185]` |
| `quantize.imatrix.chunks_count` | `[72]` |

**general.name:**
```jinja
[ 78 118 105 100 105  97  45  78 101 109 111 116 114 111 110  45  51  45
  78  97 110 111  45  79 109 110 105  45  51  48  66  45  65  51  66  45
  82 101  97 115 111 110 105 110 103]
```

**general.basename:**
```jinja
[ 78 118 105 100 105  97  45  78 101 109 111 116 114 111 110  45  51  45
  78  97 110 111  45  79 109 110 105  45  51  48  66  45  65  51  66  45
  82 101  97 115 111 110 105 110 103]
```

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**tokenizer.chat_template:**
```jinja
[123  37  32 ...  32  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 78  86  73  68  73  65  45  78 101 109 111 116 114 111 110  45  51  45
  78  97 110 111  45  79 109 110 105  45  51  48  66  45  65  51  66  45
  82 101  97 115 111 110 105 110 103  45  71  71  85  70  47 105 109  97
 116 114 105 120  95 117 110 115 108 111 116 104  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  78  86  73  68  73  65  45  78 101 109 111 116 114 111 110  45
  51  45  78  97 110 111  45  79 109 110 105  45  51  48  66  45  65  51
  66  45  82 101  97 115 111 110 105 110 103  46 116 120 116]
```

---

## gpt-oss-20b-UD_Q8_K_XL.gguf
**Pfad:** `OpenAI\gpt-oss-20b-UD_Q8_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[459]` |
| `GGUF.kv_count` | `[37]` |
| `general.architecture` | `[103 112 116  45 111 115 115]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 71 112 116  45  79 115 115  45  50  48  66]` |
| `general.basename` | `[ 71 112 116  45  79 115 115  45  50  48  66]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[50 48 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `gpt-oss.block_count` | `[24]` |
| `gpt-oss.context_length` | `[131072]` |
| `gpt-oss.embedding_length` | `[2880]` |
| `gpt-oss.feed_forward_length` | `[2880]` |
| `gpt-oss.attention.head_count` | `[64]` |
| `gpt-oss.attention.head_count_kv` | `[8]` |
| `gpt-oss.rope.freq_base` | `[150000.]` |
| `gpt-oss.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `gpt-oss.expert_count` | `[32]` |
| `gpt-oss.expert_used_count` | `[4]` |
| `gpt-oss.attention.key_length` | `[64]` |
| `gpt-oss.attention.value_length` | `[64]` |
| `gpt-oss.attention.sliding_window` | `[128]` |
| `gpt-oss.expert_feed_forward_length` | `[2880]` |
| `gpt-oss.rope.scaling.type` | `[121  97 114 110]` |
| `gpt-oss.rope.scaling.factor` | `[32.]` |
| `gpt-oss.rope.scaling.original_context_length` | `[4096]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[103 112 116  45  52 111]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 48 49 48 56 55 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[196 160  99 111  99 111  32 115]` |
| `tokenizer.ggml.bos_token_id` | `[199998]` |
| `tokenizer.ggml.eos_token_id` | `[200002]` |
| `tokenizer.ggml.padding_token_id` | `[200017]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**tokenizer.chat_template:**
```jinja
[123  35  32 ...  32  35 125]
```

---

## mmproj-Ornith-1.5-35B-BF16.gguf
**Pfad:** `ornith-ai\mmproj-Ornith-1.5-35B-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[26]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[51 53 98 32 52 48 48 48]` |
| `general.version` | `[52 48 48 48]` |
| `general.finetune` | `[51 53 98]` |
| `general.size_label` | `[52 52 55 77]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[2048]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

---

## mmproj-Ornith-1.5-9B-BF16.gguf
**Pfad:** `ornith-ai\mmproj-Ornith-1.5-9B-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[334]` |
| `GGUF.kv_count` | `[22]` |
| `general.architecture` | `[ 99 108 105 112]` |
| `general.type` | `[109 109 112 114 111 106]` |
| `general.name` | `[ 57  98  32  67 107 112 116  49  48  48  48]` |
| `general.finetune` | `[ 57  98  45  99 107 112 116  49  48  48  48]` |
| `general.size_label` | `[52 53 54 77]` |
| `general.file_type` | `[32]` |
| `clip.has_vision_encoder` | `[ True]` |
| `clip.vision.projection_dim` | `[4096]` |
| `clip.vision.image_size` | `[768]` |
| `clip.vision.patch_size` | `[16]` |
| `clip.vision.embedding_length` | `[1152]` |
| `clip.vision.feed_forward_length` | `[4304]` |
| `clip.vision.block_count` | `[27]` |
| `clip.vision.attention.head_count` | `[16]` |
| `clip.vision.image_mean` | `[0.5]` |
| `clip.vision.image_std` | `[0.5]` |
| `clip.projector_type` | `[113 119 101 110  51 118 108  95 109 101 114 103 101 114]` |
| `clip.use_gelu` | `[ True]` |
| `clip.vision.spatial_merge_size` | `[2]` |
| `clip.vision.attention.layer_norm_epsilon` | `[1.e-06]` |
| `clip.vision.is_deepstack_layers` | `[False]` |
| `general.quantization_version` | `[2]` |

---

## ornith-1.0-35b-Q6_K.gguf
**Pfad:** `ornith-ai\ornith-1.0-35b-Q6_K.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[733]` |
| `GGUF.kv_count` | `[44]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 79 114 110 105 116 104  32  49  46  48  32  51  53  66]` |
| `general.basename` | `[ 79 114 110 105 116 104  45  49  46  48]` |
| `general.size_label` | `[51 53 66]` |
| `qwen35moe.block_count` | `[40]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[510]` |
| `quantize.imatrix.chunks_count` | `[1608]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  37 125  10]
```

**quantize.imatrix.file:**
```jinja
[111 114 110 105 116 104  95  51  53  98  47 105 109  97 116 114 105 120
  46 100  97 116]
```

**quantize.imatrix.dataset:**
```jinja
[ 47 109 103 102 115  47 115 104  97 114 101 100  47  71 114 111 117 112
  95  71  89  47 119 101 110  99 104  97 111  47 115 101  47 103 103 117
 102  47  99  97 108 105  98 114  97 116 105 111 110  95  51  53  98  95
  56  48  48 107  46 116 120 116]
```

---

## ornith-1.0-35b-Q8_0.gguf
**Pfad:** `ornith-ai\ornith-1.0-35b-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[733]` |
| `GGUF.kv_count` | `[40]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 79 114 110 105 116 104  32  49  46  48  32  51  53  66]` |
| `general.basename` | `[ 79 114 110 105 116 104  45  49  46  48]` |
| `general.size_label` | `[51 53 66]` |
| `qwen35moe.block_count` | `[40]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  37 125  10]
```

---

## Ornith-1.5-35B-Q4_K_M.gguf
**Pfad:** `ornith-ai\Ornith-1.5-35B-Q4_K_M.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[753]` |
| `GGUF.kv_count` | `[47]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 79 114 110 105 116 104  45  49  46  53  45  51  53  66]` |
| `general.version` | `[52 48 48 48]` |
| `general.finetune` | `[51 53 98]` |
| `general.size_label` | `[ 50  53  54 120  50  46  54  66]` |
| `qwen35moe.block_count` | `[41]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.nextn_predict_layers` | `[1]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[15]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[510]` |
| `quantize.imatrix.chunks_count` | `[3636]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

**quantize.imatrix.file:**
```jinja
[ 47 109 103 102 115  47 115 104  97 114 101 100  47  71 114 111 117 112
  95  71  89  47 119 101 110  99 104  97 111  47 103 103 117 102  95 119
 111 114 107  47 111 117 116  47  51  53  98  45  52  48  48  48  47  79
 114 110 105 116 104  45  49  46  53  45  51  53  66  46 105 109  97 116
 114 105 120  46 103 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[ 47 109 103 102 115  47 115 104  97 114 101 100  47  71 114 111 117 112
  95  71  89  47 119 101 110  99 104  97 111  47 103 103 117 102  95 119
 111 114 107  47  99  97 108 105  98  47 109 101 114 103 101 100  47 115
 104  97 114 100 115  45  51  53  98  45  52  48  48  48  47 115  95  48
  48]
```

---

## Ornith-1.5-35B-Q8_0.gguf
**Pfad:** `ornith-ai\Ornith-1.5-35B-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[753]` |
| `GGUF.kv_count` | `[43]` |
| `general.architecture` | `[113 119 101 110  51  53 109 111 101]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 79 114 110 105 116 104  45  49  46  53  45  51  53  66]` |
| `general.version` | `[52 48 48 48]` |
| `general.finetune` | `[51 53 98]` |
| `general.size_label` | `[ 50  53  54 120  50  46  54  66]` |
| `qwen35moe.block_count` | `[41]` |
| `qwen35moe.context_length` | `[262144]` |
| `qwen35moe.embedding_length` | `[2048]` |
| `qwen35moe.attention.head_count` | `[16]` |
| `qwen35moe.attention.head_count_kv` | `[2]` |
| `qwen35moe.rope.dimension_sections` | `[0]` |
| `qwen35moe.rope.freq_base` | `[1.e+07]` |
| `qwen35moe.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35moe.expert_count` | `[256]` |
| `qwen35moe.expert_used_count` | `[8]` |
| `qwen35moe.attention.key_length` | `[256]` |
| `qwen35moe.attention.value_length` | `[256]` |
| `qwen35moe.expert_feed_forward_length` | `[512]` |
| `qwen35moe.expert_shared_feed_forward_length` | `[512]` |
| `qwen35moe.nextn_predict_layers` | `[1]` |
| `qwen35moe.ssm.conv_kernel` | `[4]` |
| `qwen35moe.ssm.state_size` | `[128]` |
| `qwen35moe.ssm.group_count` | `[16]` |
| `qwen35moe.ssm.time_step_rank` | `[32]` |
| `qwen35moe.ssm.inner_size` | `[4096]` |
| `qwen35moe.full_attention_interval` | `[4]` |
| `qwen35moe.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.ggml.bos_token_id` | `[248044]` |
| `tokenizer.ggml.add_bos_token` | `[False]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

---

## Ornith-1.5-9B-BF16.gguf
**Pfad:** `ornith-ai\Ornith-1.5-9B-BF16.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[427]` |
| `GGUF.kv_count` | `[33]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 79 114 110 105 116 104  45  49  46  53  45  57  66]` |
| `general.finetune` | `[ 57  98  45  99 107 112 116  49  48  48  48]` |
| `general.size_label` | `[57 46 48 66]` |
| `qwen35.block_count` | `[32]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[4096]` |
| `qwen35.feed_forward_length` | `[12288]` |
| `qwen35.attention.head_count` | `[16]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `general.file_type` | `[32]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[32]` |
| `qwen35.ssm.inner_size` | `[4096]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `general.quantization_version` | `[2]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

---

## Ornith-1.5-9B-Q8_0.gguf
**Pfad:** `ornith-ai\Ornith-1.5-9B-Q8_0.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[427]` |
| `GGUF.kv_count` | `[33]` |
| `general.architecture` | `[113 119 101 110  51  53]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | `[ 79 114 110 105 116 104  45  49  46  53  45  57  66]` |
| `general.finetune` | `[ 57  98  45  99 107 112 116  49  48  48  48]` |
| `general.size_label` | `[57 46 48 66]` |
| `qwen35.block_count` | `[32]` |
| `qwen35.context_length` | `[262144]` |
| `qwen35.embedding_length` | `[4096]` |
| `qwen35.feed_forward_length` | `[12288]` |
| `qwen35.attention.head_count` | `[16]` |
| `qwen35.attention.head_count_kv` | `[4]` |
| `qwen35.rope.dimension_sections` | `[0]` |
| `qwen35.rope.freq_base` | `[1.e+07]` |
| `qwen35.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `qwen35.attention.key_length` | `[256]` |
| `qwen35.attention.value_length` | `[256]` |
| `qwen35.ssm.conv_kernel` | `[4]` |
| `qwen35.ssm.state_size` | `[128]` |
| `qwen35.ssm.group_count` | `[16]` |
| `qwen35.ssm.time_step_rank` | `[32]` |
| `qwen35.ssm.inner_size` | `[4096]` |
| `qwen35.full_attention_interval` | `[4]` |
| `qwen35.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[113 119 101 110  51  53]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 50 52 56 51 49 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eos_token_id` | `[248046]` |
| `tokenizer.ggml.padding_token_id` | `[248044]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[7]` |

**tokenizer.ggml.merges:**
```jinja
[195 163 196 163 196 164 195 163 196 164 196 172 195 163 196 163 194 190
 195 163 196 163 196 187  32 195 163 196 163 196 173]
```

**tokenizer.chat_template:**
```jinja
[123  37  45 ...  32  35 125]
```

---

## Laguna-S-2.1-UD-Q3_K_XL-00001-of-00003.gguf
**Pfad:** `poolside\Laguna-S-2.1-UD-Q3_K_XL-00001-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[0]` |
| `GGUF.kv_count` | `[72]` |
| `general.architecture` | `[108  97 103 117 110  97]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_k` | `[20]` |
| `general.sampling.top_p` | `[1.]` |
| `general.sampling.min_p` | `[0.]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 76  97 103 117 110  97  45  83  45  50  46  49]` |
| `general.version` | `[50 46 49]` |
| `general.basename` | `[ 76  97 103 117 110  97  45  83  45  50  46  49]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[ 50  53  54 120  52  46  53  66]` |
| `general.license` | `[111 112 101 110 109 100 119  45  49  46  49]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 76  97 103 117 110  97  32  83  32  50  46  49]` |
| `general.base_model.0.version` | `[50 46 49]` |
| `general.base_model.0.organization` | `[ 80 111 111 108 115 105 100 101]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `laguna.block_count` | `[48]` |
| `laguna.context_length` | `[1048576]` |
| `laguna.embedding_length` | `[3072]` |
| `laguna.feed_forward_length` | `[12288]` |
| `laguna.attention.head_count` | `[72]` |
| `laguna.attention.head_count_kv` | `[8]` |
| `laguna.rope.scaling.type` | `[121  97 114 110]` |
| `laguna.rope.scaling.factor` | `[128.]` |
| `laguna.rope.scaling.original_context_length` | `[8192]` |
| `laguna.rope.scaling.yarn_attn_factor` | `[1.485203]` |
| `laguna.rope.scaling.yarn_beta_fast` | `[32.]` |
| `laguna.rope.scaling.yarn_beta_slow` | `[1.]` |
| `laguna.rope.freq_base` | `[500000.]` |
| `laguna.rope.freq_base_swa` | `[10000.]` |
| `laguna.attention.layer_norm_rms_epsilon` | `[1.e-06]` |
| `laguna.expert_count` | `[256]` |
| `laguna.expert_used_count` | `[10]` |
| `laguna.attention.key_length` | `[128]` |
| `laguna.attention.value_length` | `[128]` |
| `laguna.vocab_size` | `[100352]` |
| `laguna.attention.sliding_window` | `[512]` |
| `laguna.expert_feed_forward_length` | `[1024]` |
| `laguna.expert_shared_feed_forward_length` | `[1024]` |
| `laguna.expert_weights_norm` | `[ True]` |
| `laguna.expert_weights_scale` | `[2.5]` |
| `laguna.expert_gating_func` | `[2]` |
| `laguna.leading_dense_block_count` | `[1]` |
| `laguna.rope.dimension_count` | `[64]` |
| `laguna.rope.dimension_count_swa` | `[128]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[108  97 103 117 110  97]` |
| `tokenizer.ggml.tokens` | `[ 73 115  78 111 116  78 117 108 108]` |
| `tokenizer.ggml.token_type` | `[1]` |
| `tokenizer.ggml.merges` | `[ 73 115  32  78 111 116  78 117 108 108]` |
| `tokenizer.ggml.bos_token_id` | `[2]` |
| `tokenizer.ggml.eos_token_id` | `[2]` |
| `tokenizer.ggml.unknown_token_id` | `[0]` |
| `tokenizer.ggml.seperator_token_id` | `[8]` |
| `tokenizer.ggml.padding_token_id` | `[9]` |
| `tokenizer.ggml.mask_token_id` | `[12]` |
| `tokenizer.ggml.add_bos_token` | `[ True]` |
| `tokenizer.ggml.add_sep_token` | `[ True]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `tokenizer.ggml.eot_token_id` | `[24]` |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[12]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[572]` |
| `quantize.imatrix.chunks_count` | `[46]` |
| `split.no` | `[0]` |
| `split.tensors.count` | `[814]` |
| `split.count` | `[3]` |

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 112 111 111 108 115 105 100 101  47  76  97 103 117
 110  97  45  83  45  50  46  49]
```

**tokenizer.chat_template:**
```jinja
[123  35  45 ...  45  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 76  97 103 117 110  97  45  83  45  50  46  49  45  71  71  85  70  47
 105 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46 103 103
 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  76  97 103 117 110  97  45  83  45  50  46  49  46 116 120 116]
```

---

## Laguna-S-2.1-UD-Q3_K_XL-00002-of-00003.gguf
**Pfad:** `poolside\Laguna-S-2.1-UD-Q3_K_XL-00002-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[761]` |
| `GGUF.kv_count` | `[3]` |
| `split.no` | `[1]` |
| `split.tensors.count` | `[814]` |
| `split.count` | `[3]` |

---

## Laguna-S-2.1-UD-Q3_K_XL-00003-of-00003.gguf
**Pfad:** `poolside\Laguna-S-2.1-UD-Q3_K_XL-00003-of-00003.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[53]` |
| `GGUF.kv_count` | `[3]` |
| `split.no` | `[2]` |
| `split.tensors.count` | `[814]` |
| `split.count` | `[3]` |

---

## Hy-MT2-30B-A3B-Q6_K.gguf
**Pfad:** `tencent\Hy-MT2-30B-A3B-Q6_K.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[766]` |
| `GGUF.kv_count` | `[37]` |
| `general.architecture` | `[104 121  95 118  51]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.name` | *Siehe Code-Block unten* |
| `general.basename` | `[116 101 110  99 101 110 116  45  72 121  45  77  84  50]` |
| `general.size_label` | `[51 48 66 45 65 51 66]` |
| `general.license` | `[ 97 112  97  99 104 101  45  50  46  48]` |
| `general.tags` | `[116 114  97 110 115 108  97 116 105 111 110]` |
| `general.languages` | `[117 103]` |
| `hy_v3.block_count` | `[48]` |
| `hy_v3.context_length` | `[262144]` |
| `hy_v3.embedding_length` | `[2048]` |
| `hy_v3.feed_forward_length` | `[6912]` |
| `hy_v3.attention.head_count` | `[32]` |
| `hy_v3.attention.head_count_kv` | `[4]` |
| `hy_v3.rope.freq_base` | `[1.115884e+07]` |
| `hy_v3.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `hy_v3.expert_count` | `[128]` |
| `hy_v3.expert_used_count` | `[8]` |
| `hy_v3.attention.key_length` | `[128]` |
| `hy_v3.attention.value_length` | `[128]` |
| `hy_v3.expert_feed_forward_length` | `[768]` |
| `hy_v3.expert_shared_feed_forward_length` | `[768]` |
| `hy_v3.expert_weights_norm` | `[ True]` |
| `hy_v3.expert_weights_scale` | `[2.826]` |
| `hy_v3.expert_gating_func` | `[2]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[104 117 110 121 117  97 110  45 100 101 110 115 101]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 50 48 56 51 49 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | *Siehe Code-Block unten* |
| `tokenizer.ggml.bos_token_id` | `[120000]` |
| `tokenizer.ggml.eos_token_id` | `[120025]` |
| `tokenizer.ggml.padding_token_id` | `[120002]` |
| `tokenizer.ggml.seperator_token_id` | `[120007]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |

**general.name:**
```jinja
[ 84 101 110  99 101 110 116  32  72 121  32  77  84  50  32  51  48  66
  32  65  51  66]
```

**tokenizer.ggml.merges:**
```jinja
[195 168 196 178 194 165 195 169 196 162 197 130  32 195 168 196 171 194
 175 195 165 194 165 194 189]
```

**tokenizer.chat_template:**
```jinja
[123  35  32 ...  32  37 125]
```

---

## GLM-4.7-Flash-UD_Q6_K_XL.gguf
**Pfad:** `Zhipu AI\GLM 4.7 Flash\GLM-4.7-Flash-UD_Q6_K_XL.gguf`

| Metadaten-Schlüssel | Wert |
|---|---|
| `GGUF.version` | `[3]` |
| `GGUF.tensor_count` | `[844]` |
| `GGUF.kv_count` | `[60]` |
| `general.architecture` | `[100 101 101 112 115 101 101 107  50]` |
| `general.type` | `[109 111 100 101 108]` |
| `general.sampling.top_p` | `[0.95]` |
| `general.sampling.temp` | `[1.]` |
| `general.name` | `[ 71 108 109  45  52  46  55  45  70 108  97 115 104]` |
| `general.basename` | `[ 71 108 109  45  52  46  55  45  70 108  97 115 104]` |
| `general.quantized_by` | `[ 85 110 115 108 111 116 104]` |
| `general.size_label` | `[ 54  52 120  50  46  54  66]` |
| `general.license` | `[109 105 116]` |
| `general.repo_url` | *Siehe Code-Block unten* |
| `general.base_model.count` | `[1]` |
| `general.base_model.0.name` | `[ 71  76  77  32  52  46  55  32  70 108  97 115 104]` |
| `general.base_model.0.organization` | `[ 90  97 105  32  79 114 103]` |
| `general.base_model.0.repo_url` | *Siehe Code-Block unten* |
| `general.tags` | `[116 101 120 116  45 103 101 110 101 114  97 116 105 111 110]` |
| `general.languages` | `[122 104]` |
| `deepseek2.block_count` | `[47]` |
| `deepseek2.context_length` | `[202752]` |
| `deepseek2.embedding_length` | `[2048]` |
| `deepseek2.feed_forward_length` | `[10240]` |
| `deepseek2.attention.head_count` | `[20]` |
| `deepseek2.attention.head_count_kv` | `[1]` |
| `deepseek2.rope.freq_base` | `[1.e+06]` |
| `deepseek2.attention.layer_norm_rms_epsilon` | `[1.e-05]` |
| `deepseek2.expert_used_count` | `[4]` |
| `deepseek2.expert_group_count` | `[1]` |
| `deepseek2.expert_group_used_count` | `[1]` |
| `deepseek2.expert_gating_func` | `[2]` |
| `deepseek2.leading_dense_block_count` | `[1]` |
| `deepseek2.vocab_size` | `[154880]` |
| `deepseek2.attention.q_lora_rank` | `[768]` |
| `deepseek2.attention.kv_lora_rank` | `[512]` |
| `deepseek2.attention.key_length` | `[576]` |
| `deepseek2.attention.value_length` | `[512]` |
| `deepseek2.attention.key_length_mla` | `[256]` |
| `deepseek2.attention.value_length_mla` | `[256]` |
| `deepseek2.expert_feed_forward_length` | `[1536]` |
| `deepseek2.expert_count` | `[64]` |
| `deepseek2.expert_shared_count` | `[1]` |
| `deepseek2.expert_weights_scale` | `[1.8]` |
| `deepseek2.expert_weights_norm` | `[ True]` |
| `deepseek2.rope.dimension_count` | `[64]` |
| `tokenizer.ggml.model` | `[103 112 116  50]` |
| `tokenizer.ggml.pre` | `[103 108 109  52]` |
| `tokenizer.ggml.tokens` | `[91 80 65 68 49 53 52 56 55 57 93]` |
| `tokenizer.ggml.token_type` | `[5]` |
| `tokenizer.ggml.merges` | `[195 162 196 189  32 194 191]` |
| `tokenizer.ggml.eos_token_id` | `[154820]` |
| `tokenizer.ggml.padding_token_id` | `[154821]` |
| `tokenizer.ggml.bos_token_id` | `[154822]` |
| `tokenizer.ggml.eot_token_id` | `[154827]` |
| `tokenizer.ggml.unknown_token_id` | `[154820]` |
| `tokenizer.ggml.eom_token_id` | `[154829]` |
| `tokenizer.chat_template` | *Siehe Code-Block unten* |
| `general.quantization_version` | `[2]` |
| `general.file_type` | `[18]` |
| `quantize.imatrix.file` | *Siehe Code-Block unten* |
| `quantize.imatrix.dataset` | *Siehe Code-Block unten* |
| `quantize.imatrix.entries_count` | `[607]` |
| `quantize.imatrix.chunks_count` | `[85]` |

**general.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 117 110 115 108 111 116 104]
```

**general.base_model.0.repo_url:**
```jinja
[104 116 116 112 115  58  47  47 104 117 103 103 105 110 103 102  97  99
 101  46  99 111  47 122  97 105  45 111 114 103  47  71  76  77  45  52
  46  55  45  70 108  97 115 104]
```

**tokenizer.chat_template:**
```jinja
[ 91 103  77 ...  45  37 125]
```

**quantize.imatrix.file:**
```jinja
[ 71  76  77  45  52  46  55  45  70 108  97 115 104  45  71  71  85  70
  47 105 109  97 116 114 105 120  95 117 110 115 108 111 116 104  46 103
 103 117 102]
```

**quantize.imatrix.dataset:**
```jinja
[117 110 115 108 111 116 104  95  99  97 108 105  98 114  97 116 105 111
 110  95  71  76  77  45  52  46  55  45  70 108  97 115 104  46 116 120
 116]
```

---

