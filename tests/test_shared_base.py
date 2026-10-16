import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from post_training.shared_base_scenario import adapter_parameters, export_adapter, frozen_digest, isolated_update


def tiny_model():
    base=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2))
    config=LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM')
    model=get_peft_model(base,config,adapter_name='tenant-a')
    model.add_adapter('tenant-b',config)
    return model


def test_updates_are_isolated_and_base_is_frozen(tmp_path):
    model=tiny_model()
    before=frozen_digest(model)
    groups={n:adapter_parameters(model,n) for n in ('tenant-a','tenant-b')}
    assert not {id(p) for p in groups['tenant-a']} & {id(p) for p in groups['tenant-b']}
    optimizers={n:torch.optim.AdamW(p,lr=.01,weight_decay=0) for n,p in groups.items()}
    batch={'input_ids':torch.tensor([[1,2,3,4]]),'attention_mask':torch.ones(1,4,dtype=torch.long),
           'labels':torch.tensor([[-100,-100,3,4]])}
    for name,other in [('tenant-a','tenant-b'),('tenant-b','tenant-a')]:
        saved=[p.detach().clone() for p in groups[other]]
        active_before=[p.detach().clone() for p in groups[name]]
        isolated_update(model,name,groups[name],optimizers[name],batch,'fp32')
        assert all(torch.equal(a,b) for a,b in zip(saved,groups[other]))
        assert any(not torch.equal(a,b) for a,b in zip(active_before,groups[name]))
        assert all(p.grad is None for p in groups[other])
    assert frozen_digest(model)==before
    for name in groups:
        export_adapter(model,name,tmp_path/name)
        assert (tmp_path/name/'adapter_model.safetensors').is_file()
        assert (tmp_path/name/'adapter_config.json').is_file()


def test_completion_mask_ignores_prompt_and_padding():
    import pytest
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast
    from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
    from post_training.shared_base_scenario import tokenize_pairs
    backend=Tokenizer(models.WordLevel({'[UNK]':0,'[EOS]':1,'Question':2,':':3,'Answer':4,'request':5,'label':6},unk_token='[UNK]'))
    backend.pre_tokenizer=pre_tokenizers.Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',eos_token='[EOS]',pad_token='[EOS]')
    rows=tokenize_pairs(tokenizer,[('request','label'),('request request','label')],32)
    collator=DataCollatorForLanguageModeling(pad_token_id=1,completion_only_loss=True)
    batch=collator(rows)
    assert (batch['labels'][:,1:]!=-100).sum().item()==4
    assert torch.all(batch['labels'][batch['attention_mask']==0]==-100)
    assert batch['labels'][0,0].item()==-100
    with pytest.raises(ValueError,match='exceeds'):
        tokenize_pairs(tokenizer,[('request','label')],2)
