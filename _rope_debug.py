"""Quick debug script to trace NaN origin in RoPE attention forward pass."""
import sys, math, types
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
from einops import rearrange
from methyldl.modelling.classifiers.dnabert2 import EpigenDnabert2

FOUNDATION_MODEL_PATH = str(ROOT / "foundationalModels/DNABERT-2-117M")

model = EpigenDnabert2(
    foundation_model_huggingface=FOUNDATION_MODEL_PATH,
    max_sequence_length=150,
    num_labels=40,
    num_dmr_labels=2000,
    use_cpg_methylation=True,
    use_m6a_methylation=False,
    use_triton=False,
    positional_encoding="rope",
)

enc = model.model.bert.encoder
# The DNABERT-2-117M directory has hyphens so can't be dotted-imported;
# add it directly to sys.path.
sys.path.insert(0, str(ROOT / "foundationalModels/DNABERT-2-117M"))
from bert_padding import pad_input
from bert_layers import BertUnpadSelfAttention

def _debug_fwd(self, hidden_states, cu_seqlens, max_seqlen_in_batch, indices, attn_mask, bias):
    qkv = self.Wqkv(hidden_states)
    qkv_pad = pad_input(qkv, indices, cu_seqlens.shape[0]-1, max_seqlen_in_batch)
    qkv_r = rearrange(qkv_pad, "b s (t h d) -> b s t h d", t=3, h=self.num_attention_heads)
    q = qkv_r[:,:,0,:,:].permute(0,2,1,3)
    k = qkv_r[:,:,1,:,:].permute(0,2,1,3)
    v = qkv_r[:,:,2,:,:].permute(0,2,1,3)
    print(f"  q pre-rope  NaN={torch.isnan(q).any().item()} max={q.abs().max().item():.4f}")
    
    # check inv_freq
    print(f"  rotary inv_freq NaN={torch.isnan(self.rotary_emb.inv_freq).any().item()}")
    
    q_r, k_r = self.rotary_emb(q, k)
    print(f"  q post-rope NaN={torch.isnan(q_r).any().item()} max={q_r.abs().max().item():.4f}")
    
    # check cos/sin caches
    cos = self.rotary_emb._cos_cached
    sin = self.rotary_emb._sin_cached
    print(f"  cos NaN={torch.isnan(cos).any().item()} max={cos.abs().max().item():.4f}")
    print(f"  sin NaN={torch.isnan(sin).any().item()} max={sin.abs().max().item():.4f}")
    
    k_T = k_r.permute(0,1,3,2)
    scores = torch.matmul(q_r, k_T) / math.sqrt(self.attention_head_size)
    print(f"  scores      NaN={torch.isnan(scores).any().item()} max={scores.abs().max().item():.4f}")
    
    scores_b = scores + bias
    print(f"  bias        NaN={torch.isnan(bias).any().item()} dtype={bias.dtype} shape={tuple(bias.shape)}")
    print(f"  scores+bias NaN={torch.isnan(scores_b).any().item()} max={scores_b.abs().max().item():.4f}")
    
    probs = nn.functional.softmax(scores_b, dim=-1)
    print(f"  probs       NaN={torch.isnan(probs).any().item()} max={probs.max().item():.4f} min={probs.min().item():.6f}")
    
    attn = torch.matmul(probs, v)
    print(f"  attn out    NaN={torch.isnan(attn).any().item()}")
    
    # run the real forward now
    from foundationalModels.DNABERT_2_117M.bert_layers import BertUnpadSelfAttention
    return BertUnpadSelfAttention.forward(self, hidden_states, cu_seqlens, max_seqlen_in_batch, indices, attn_mask, bias)

# Only patch layer 0 for debugging
layer0_attn = enc.layer[0].attention.self
layer0_attn.forward = types.MethodType(_debug_fwd, layer0_attn)

tok = model.tokenizer(["ACGT" * 10], return_tensors="pt", padding=True, truncation=True)
cpg = torch.zeros(1, tok["input_ids"].shape[1], dtype=torch.long)
labels = torch.tensor([0])
dmr_ids = torch.tensor([0])

print("=== Layer 0 debug ===")
out = model.model(
    input_ids=tok["input_ids"],
    attention_mask=tok["attention_mask"],
    cpg_methylation=cpg,
    labels=labels,
    dmr_ids=dmr_ids,
)
print(f"Final logits NaN: {torch.isnan(out.logits).any().item()}")
print(f"Final loss: {out.loss}")
