
import torch 
from torch import nn
import math 

# Lora实现，封装linear，替换到父module里
class LoraLayer(nn.Module):
    def __init__(self,raw_linear,in_features,out_features,r,alpha):
        super().__init__()
        self.r=r 
        self.alpha=alpha
        self.lora_a=nn.Parameter(torch.empty((in_features,r)))
        self.lora_b=nn.Parameter(torch.zeros((r,out_features)))
    
        nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5))

        self.raw_linear=raw_linear
    
    def forward(self,x):    # x:(batch_size,in_features)
        raw_output=self.raw_linear(x)   
        lora_output=x@((self.lora_a@self.lora_b)*self.alpha/self.r)    # matmul(x,matmul(lora_a,lora_b)*alpha/r)
        return raw_output+lora_output

def inject_lora(model,name,layer,LORA_R,LORA_ALPHA=8):
    name_cols=name.split('.')

    # 逐层下探到linear归属的module
    children=name_cols[:-1]
    cur_layer=model 
    for child in children:
        cur_layer=getattr(cur_layer,child)
    
    #print(layer==getattr(cur_layer,name_cols[-1]))
    lora_layer=LoraLayer(layer,layer.in_features,layer.out_features,LORA_R,LORA_ALPHA)
    setattr(cur_layer,name_cols[-1],lora_layer)


class CustomAttention(nn.Module):
    def __init__(self, original_attn, k_lora, v_lora):
        super(CustomAttention, self).__init__()
        self.original_attn = original_attn
        self.k_lora = k_lora
        self.v_lora = v_lora

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, past_key_value=None, use_cache=False, output_attentions=False):
        # 获取原始的 Q, K, V
        mixed_query_layer = self.original_attn.q_proj(hidden_states)
        mixed_key_layer = self.k_lora(hidden_states)
        mixed_value_layer = self.v_lora(hidden_states)

        # 继续使用原始的注意力机制
        return self.original_attn._attn(mixed_query_layer, mixed_key_layer, mixed_value_layer, attention_mask,
                                        head_mask, encoder_hidden_states, encoder_attention_mask, past_key_value,
                                        use_cache, output_attentions)