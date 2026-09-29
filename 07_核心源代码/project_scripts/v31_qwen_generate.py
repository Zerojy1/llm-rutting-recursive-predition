from __future__ import annotations
import json, os, re
from pathlib import Path
import numpy as np
import torch


def qwen_model_load_plan(device='cuda:0'):
    """Return a dependency-free load plan used by the runtime model loader."""
    cuda=str(device).startswith('cuda')
    return {'use_4bit':cuda,'dtype':'float16' if cuda else 'float32','device':str(device)}

def parse_increment_json(text:str,expected_len:int):
    """Parse the structured increment payload from Qwen3.5 output.

    Qwen3.5 thinking-mode templates may emit <think>...</think> before the final
    JSON. The scientific payload is the JSON object; reasoning wrappers and
    markdown fences are ignored, while the required key/length remain strict.
    """
    clean=str(text).strip().replace('```json','').replace('```','').strip()
    clean=re.sub(r'<think>.*?</think>', '', clean, flags=re.DOTALL|re.IGNORECASE)
    clean=clean.replace('<think>','').replace('</think>','').strip()
    decoder=json.JSONDecoder(); obj=None
    for m in re.finditer(r'\{', clean):
        try:
            candidate,_=decoder.raw_decode(clean[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate,dict) and 'rut_increment_mm' in candidate:
            obj=candidate; break
    if obj is None:
        raise ValueError('no valid rut_increment_mm JSON object found')
    vals=obj.get('rut_increment_mm')
    # In masked Stage06 replacement, a block can contain exactly one withheld
    # state. Qwen3.5 may then serialize the scientifically equivalent one-step
    # payload as a JSON scalar instead of a one-element array. Normalize only
    # this unambiguous case; multi-step payloads must still be arrays.
    if not isinstance(vals,list):
        if expected_len == 1 and not isinstance(vals,bool) and isinstance(vals,(int,float)) and np.isfinite(float(vals)):
            vals=[float(vals)]
        else:
            raise ValueError(f'expected {expected_len} increments as a JSON array; got {type(vals).__name__}')
    # The V3.1 Qwen SFT target is a six-step trajectory. Masked replacement can
    # request shorter prefixes (1-5 steps), so a valid six-step output may supply
    # the requested prefix. Qwen can also occasionally append exactly one extra
    # tail forecast despite an exact-horizon instruction; because the list is in
    # temporal order, dropping only that unrequested seventh tail value preserves
    # the six trained/requested steps. Missing steps and larger arbitrary length
    # mismatches remain hard failures.
    if len(vals)==expected_len:
        use_vals=vals
    elif 0 < expected_len < 6 and len(vals)==6:
        use_vals=vals[:expected_len]
    elif 0 < expected_len <= 6 and len(vals)==7:
        use_vals=vals[:expected_len]
    else:
        raise ValueError(f'expected {expected_len} increments (or the trained six-step horizon / one-extra-tail repair), got {len(vals)}')
    arr=np.asarray(use_vals,dtype=float)
    if not np.isfinite(arr).all(): raise ValueError('non-finite increment')
    return arr.tolist()


def apply_generation_template(processor,messages):
    """Build a direct-response Qwen3.5 prompt with thinking disabled."""
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

class QwenTrajectoryGenerator:
    def __init__(self,model_dir,adapter_dir=None,device='cuda:0',max_new_tokens=160):
        from transformers import AutoProcessor
        self.processor=AutoProcessor.from_pretrained(str(model_dir),local_files_only=True)
        try:
            from transformers import AutoModelForMultimodalLM as AutoModel
        except ImportError:
            from transformers import AutoModelForCausalLM as AutoModel
        plan=qwen_model_load_plan(device)
        dtype=torch.float16 if plan['dtype']=='float16' else torch.float32
        kwargs={'dtype':dtype,'local_files_only':True}
        if plan['use_4bit']:
            from transformers import BitsAndBytesConfig
            kwargs['quantization_config']=BitsAndBytesConfig(
                load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16)
            kwargs['device_map']={'':device}
            kwargs['low_cpu_mem_usage']=True
        self.model=AutoModel.from_pretrained(str(model_dir),**kwargs)
        if not plan['use_4bit']:
            self.model=self.model.to(device)
        if adapter_dir:
            from peft import PeftModel
            self.model=PeftModel.from_pretrained(self.model,str(adapter_dir))
        self.model.eval(); self.device=torch.device(device); self.max_new_tokens=max_new_tokens
        tok=getattr(self.processor,'tokenizer',self.processor)
        if getattr(tok,'pad_token_id',None) is None and getattr(tok,'eos_token_id',None) is not None:
            tok.pad_token_id=tok.eos_token_id
        if hasattr(self.model,'generation_config') and getattr(tok,'pad_token_id',None) is not None:
            self.model.generation_config.pad_token_id=tok.pad_token_id
    def __call__(self,condition:dict,temperature=.7,do_sample=True):
        future_len=len(condition['future_scenario'])
        system=(
            'You are a pavement deterioration trajectory generator. Return strict JSON only. '
            f'Return exactly {future_len} numeric values under the key rut_increment_mm, in the same temporal order as future_scenario, with no extra values. '
            'Generate observed rutting increments in mm. Local negative increments are allowed when plausible.'
        )
        request=dict(condition)
        request['output']=f'exactly {future_len} observed rutting increments in mm, one per future_scenario item'
        user=json.dumps(request,ensure_ascii=False)
        msgs=[{'role':'system','content':[{'type':'text','text':system}]},{'role':'user','content':[{'type':'text','text':user}]}]
        prompt=apply_generation_template(self.processor,msgs)
        tok=getattr(self.processor,'tokenizer',self.processor); inp=tok(prompt,return_tensors='pt').to(self.device)
        gen_kwargs={'max_new_tokens':self.max_new_tokens,'do_sample':do_sample}
        if do_sample: gen_kwargs['temperature']=temperature
        if getattr(tok,'pad_token_id',None) is not None: gen_kwargs['pad_token_id']=tok.pad_token_id
        if getattr(tok,'eos_token_id',None) is not None: gen_kwargs['eos_token_id']=tok.eos_token_id
        with torch.no_grad(): out=self.model.generate(**inp,**gen_kwargs)
        text=tok.decode(out[0,inp['input_ids'].shape[1]:],skip_special_tokens=True)
        return parse_increment_json(text,future_len)

def consensus_generate(generator,condition,k=5,temperature=.7):
    draws=[]; errors=[]
    for _ in range(k):
        try: draws.append(generator(condition,temperature=temperature,do_sample=True))
        except Exception as e: errors.append(str(e))
    if not draws: raise RuntimeError('all Qwen generations failed: '+('; '.join(errors[:3])))
    a=np.asarray(draws,float); med=np.median(a,axis=0); mad=np.median(np.abs(a-med),axis=0)
    return {'increments':med.tolist(),'mad':mad.tolist(),'n_valid':len(draws),'draws':draws,'errors':errors}
