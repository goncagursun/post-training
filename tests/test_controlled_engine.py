import torch
from peft import LoraConfig,get_peft_model
from transformers import LlamaConfig,LlamaForCausalLM
from post_training.controlled_engine import isolated_update,adapter_parameters,frozen_digest


def build(shared):
    torch.manual_seed(42)
    model=LlamaForCausalLM(LlamaConfig(vocab_size=32,hidden_size=16,intermediate_size=32,
        num_hidden_layers=1,num_attention_heads=2,num_key_value_heads=2,attention_dropout=0))
    rng=torch.get_rng_state()
    lora=LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],bias='none',task_type='CAUSAL_LM')
    model=get_peft_model(model,lora,adapter_name='tenant-a')
    if shared:
        torch.set_rng_state(rng)
        model.add_adapter('tenant-b',lora)
    return model


def test_same_update_function_preserves_solo_trajectory_when_interleaved():
    solo,shared=build(False),build(True)
    before=frozen_digest(shared)
    groups={name:adapter_parameters(shared,name) for name in ('tenant-a','tenant-b')}
    opts={name:torch.optim.AdamW(p,lr=.01,weight_decay=0) for name,p in groups.items()}
    solo_params=adapter_parameters(solo,'tenant-a')
    solo_opt=torch.optim.AdamW(solo_params,lr=.01,weight_decay=0)
    batch={'input_ids':torch.tensor([[1,2,3,4]]),'attention_mask':torch.ones(1,4,dtype=torch.long),
        'labels':torch.tensor([[-100,-100,3,4]])}
    for _ in range(4):
        isolated_update(solo,'tenant-a',solo_params,solo_opt,batch,'fp32')
        isolated_update(shared,'tenant-a',groups['tenant-a'],opts['tenant-a'],batch,'fp32')
        isolated_update(shared,'tenant-b',groups['tenant-b'],opts['tenant-b'],batch,'fp32')
    assert all(torch.equal(a,b) for a,b in zip(solo_params,groups['tenant-a']))
    assert frozen_digest(shared)==before


def test_weight_gate_rejects_adapter_contamination(tmp_path):
    import runpy
    from pathlib import Path
    from safetensors.torch import save_file
    compare=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/verify_controlled_isolation.py'))['compare_weights']
    left,right=tmp_path/'solo',tmp_path/'interleaved'
    left.mkdir(); right.mkdir()
    save_file({'lora_A.weight':torch.ones(2,2)},str(left/'adapter_model.safetensors'))
    save_file({'lora_A.weight':torch.ones(2,2)},str(right/'adapter_model.safetensors'))
    assert compare(left,right)['exact']
    altered=torch.ones(2,2)
    altered[0,0]+=.01
    save_file({'lora_A.weight':altered},str(right/'adapter_model.safetensors'))
    result=compare(left,right)
    assert not result['passed']
    assert result['max_abs_difference']>.009
