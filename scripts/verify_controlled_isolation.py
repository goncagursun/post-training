"""Compare solo A with shared A using prespecified weight and fixed-logit tolerances."""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from post_training.bitext import load_manifest, intent_question
from post_training.training.sft import format_prompt


def compare_weights(left,right,atol=1e-6,rtol=1e-5):
    a,b=load_file(str(Path(left)/'adapter_model.safetensors')),load_file(str(Path(right)/'adapter_model.safetensors'))
    if a.keys()!=b.keys():raise ValueError('Adapter tensor keys differ')
    details={}
    for key in a:
        x,y=a[key].float(),b[key].float()
        details[key]={'max_abs_difference':float((x-y).abs().max()),
            'exact':torch.equal(x,y),'within_tolerance':torch.allclose(x,y,atol=atol,rtol=rtol)}
    return {'atol':atol,'rtol':rtol,'exact':all(d['exact'] for d in details.values()),
        'passed':all(d['within_tolerance'] for d in details.values()),
        'max_abs_difference':max(d['max_abs_difference'] for d in details.values()),'tensors':details}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    root=Path(args.output_dir)
    solo=json.loads((root/'solo-a/report.json').read_text())['jobs'][0]
    shared=json.loads((root/'shared/report.json').read_text())['jobs'][0]
    initial_equal=solo['initial_adapter_sha256']==shared['initial_adapter_sha256']
    order_equal=solo['order_sha256']==shared['order_sha256']
    weights=compare_weights(solo['adapter_dir'],shared['adapter_dir'])
    if solo['config']!=shared['config']:raise ValueError('Recipes differ')
    config=solo['config']
    _,splits=load_manifest(solo['manifest_path'],solo['tenant_id'])
    prompts=[format_prompt(intent_question(row['instruction'])) for row in splits['validation'][:3]]
    tokenizer=AutoTokenizer.from_pretrained(solo['adapter_dir'],local_files_only=True)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained(config['model_name'],revision=config['model_revision'],
        dtype=torch.float32,attn_implementation='eager').to('cuda')
    model=PeftModel.from_pretrained(model,solo['adapter_dir'],adapter_name='solo',is_trainable=False)
    model.load_adapter(shared['adapter_dir'],adapter_name='shared',is_trainable=False)
    values={}
    for name in ('solo','shared'):
        model.set_adapter(name)
        model.eval()
        outputs=[]
        for prompt in prompts:
            inputs=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to('cuda')
            with torch.inference_mode():outputs.append(model(**inputs).logits[0,-1].float().cpu())
        values[name]=torch.stack(outputs)
    torch.save(values,root/'isolation_fixed_logits.pt')
    logits={'atol':1e-5,'rtol':1e-5,'prompts':prompts,'scope':'full vocabulary logits at final prompt token; FP32 eager inference',
        'exact':torch.equal(values['solo'],values['shared']),
        'max_abs_difference':float((values['solo']-values['shared']).abs().max()),
        'passed':torch.allclose(values['solo'],values['shared'],atol=1e-5,rtol=1e-5)}
    result={'initial_adapter_equal':initial_equal,'batch_order_equal':order_equal,
        'weights':weights,'fixed_input_logits':logits,
        'passed':initial_equal and order_equal and weights['passed'] and logits['passed']}
    (root/'isolation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','initial_adapter_equal','batch_order_equal')}),flush=True)
    if not result['passed']:raise RuntimeError('Solo/shared isolation comparison failed')


if __name__=='__main__':main()
