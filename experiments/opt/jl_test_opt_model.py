"""OPT architecture, gradient, pooling and true-batch regression checks."""

import copy
import io
from contextlib import redirect_stdout
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import OPTConfig, OPTForSequenceClassification

import jl_opt_model as opt
import jl_trainer as trainer
from jl_protocol import selected_state
from lora.lora_finetune import add_lora
from lora.lora import LoraLayer


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        config = OPTConfig(vocab_size=32, hidden_size=32, word_embed_proj_dim=32,
                           ffn_dim=64, num_hidden_layers=12, num_attention_heads=4,
                           max_position_embeddings=64, num_labels=2,
                           pad_token_id=1, eos_token_id=2, use_cache=False,
                           dropout=.3, attention_dropout=.3, activation_dropout=.3)
        config._attn_implementation = 'sdpa'
        self.model = OPTForSequenceClassification(config).requires_grad_(False)
        self.model._jl_micro_batch_size = 64
        add_lora(self.model, range(12), 10, 'opt')
        trainer.configure_dropout(self.model, 'disabled')

    def test_exact_targets_frozen_head_and_sdpa(self):
        model = self.model
        opt._verify_attention(model, 'sdpa')
        expected = {f'model.decoder.layers.{i}.self_attn.{proj}'
                    for i in range(12) for proj in ('k_proj', 'v_proj')}
        self.assertEqual({n for n,m in model.named_modules() if isinstance(m,LoraLayer)}, expected)
        self.assertFalse(model.score.weight.requires_grad)
        self.assertEqual(len(selected_state(model)), 49)
        for n,p in model.named_parameters():
            self.assertEqual(p.requires_grad, n.endswith(('lora_a','lora_b')))
        model.train()
        ids=torch.tensor([[2,4,5,1],[2,6,7,8]])
        with patch('torch.nn.functional.scaled_dot_product_attention',
                   wraps=torch.nn.functional.scaled_dot_product_attention) as call:
            a=model(input_ids=ids,attention_mask=ids.ne(1)).logits
            b=model(input_ids=ids,attention_mask=ids.ne(1)).logits
        self.assertEqual(call.call_count,24)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        a.sum().backward()
        self.assertTrue(any(p.grad is not None for n,p in model.named_parameters() if 'lora_' in n))
        self.assertTrue(all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n))

    def test_padding_and_factor_coordinates(self):
        self.model.eval()
        ids=torch.tensor([[2,4,5,1],[2,6,7,8]])
        mask=ids.ne(1)
        with torch.no_grad():
            hidden=self.model.model(input_ids=ids,attention_mask=mask).last_hidden_state
            expected=self.model.score(hidden)[torch.arange(2),torch.tensor([2,3])]
            actual=self.model(input_ids=ids,attention_mask=mask).logits
        torch.testing.assert_close(actual,expected)
        for layer in range(12):
            for name in opt.factor_param_names(layer,'A'):
                self.assertEqual(opt.model_layer_index(name),layer)
                self.assertIn(name,dict(self.model.named_parameters()))
        self.assertEqual(opt.frozen_for_selection(range(12),{}),([],[],[]))
        self.assertEqual(len(opt.frozen_for_selection([], {})[0]),48)
        with self.assertRaises(ValueError):opt.frozen_for_selection([12],{})


class FakeClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(3,2)
        self._jl_micro_batch_size=64
        self.calls=[]

    def forward(self,input_ids,attention_mask,**kw):
        self.calls.append(len(input_ids))
        return SimpleNamespace(logits=self.projection(input_ids.float()*attention_mask))


class BatchTests(unittest.TestCase):
    def test_actual_batch64_no_microbatch_and_short_final_batch(self):
        model=FakeClassifier()
        dataset={'input_ids':[[2,3,4]]*71,'attention_mask':[[1,1,1]]*71,
                 'label':[i%2 for i in range(71)]}
        with redirect_stdout(io.StringIO()), patch.object(trainer,'_backward_microbatches',
                                                         side_effect=AssertionError('Unexpected splitting')):
            trainer.fit(model,trainset=dataset,epochs=1,lr=.01,args_optimizer='sgd',
                        device='cpu',finetune=None,model='opt',disable_tqdm=True,
                        batch_size=64,shuffle_seed=37)
        self.assertEqual(model.calls,[64,7])
        model.calls.clear()
        with redirect_stdout(io.StringIO()):
            trainer.test(model,dataset,SimpleNamespace(model='opt',out_dim=2,jl_eval_batch_size=64),
                         verbose=False,device='cpu',disable_tqdm=True,threshold=.5)
        self.assertEqual(model.calls,[64,7])

    def test_optional_gradient_accumulation_preserves_weighted_loss(self):
        full=FakeClassifier(); micro=copy.deepcopy(full)
        ids=torch.randn(7,3); mask=torch.ones_like(ids); labels=torch.arange(7)%2
        criterion=nn.CrossEntropyLoss(weight=torch.tensor([1.,2.7]))
        loss=criterion(full(ids,mask).logits,labels);loss.backward()
        measured,_=trainer._backward_microbatches(micro,ids,mask,labels,criterion,'cpu','fp32',4)
        torch.testing.assert_close(measured,loss.detach())
        for a,b in zip(full.parameters(),micro.parameters()):torch.testing.assert_close(a.grad,b.grad)


if __name__=='__main__':unittest.main()
