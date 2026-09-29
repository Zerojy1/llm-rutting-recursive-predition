from __future__ import annotations
"""V3.1 chronological Qwen3.5-9B QLoRA selector + refit.

Selection: 2016-2017 train, 2018 development. The best epoch is selected only by 2018
assistant-token loss. Refit: reinitialize the base model and train on 2016-2018 for exactly
the selected number of epochs. 2019-2021 targets are never used by this module.
"""
import gc, importlib.util, json, math, os, platform, random, shutil, time
from pathlib import Path
import numpy as np
import torch

MAX_LENGTH=2048; MAX_EPOCHS=8; PATIENCE=2; GRAD_ACCUM=8; LR=1e-4; WD=0.01; WARMUP=.10
LORA_R=8; LORA_ALPHA=16; LORA_DROPOUT=.05; SEED=20260805
os.environ.setdefault('HF_HUB_OFFLINE','1'); os.environ.setdefault('TRANSFORMERS_OFFLINE','1'); os.environ.setdefault('TOKENIZERS_PARALLELISM','false')


def read_jsonl(path):
    records=[]
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line); roles=[m.get('role') for m in r.get('messages',[])]
            if roles!=['system','user','assistant']: raise ValueError(f'bad message roles: {roles}')
            records.append(r)
    return records


def assert_target_years(records,min_year=None,max_year=None):
    for r in records:
        md=r.get('metadata',{}); lo=int(md['target_year_min']); hi=int(md['target_year_max'])
        if min_year is not None and lo<min_year: raise ValueError(f'target-year leakage: {md}')
        if max_year is not None and hi>max_year: raise ValueError(f'target-year leakage: {md}')


def _convert_messages(messages):
    return [{'role':m['role'],'content':[{'type':'text','text':m['content']}]} for m in messages]


def _tokenizer(processor):
    tok=getattr(processor,'tokenizer',processor)
    if getattr(tok,'pad_token_id',None) is None: tok.pad_token=tok.eos_token
    return tok


def tokenize_record(processor,record,max_length=MAX_LENGTH):
    tok=_tokenizer(processor); msgs=_convert_messages(record['messages'])
    prompt=processor.apply_chat_template(msgs[:2],tokenize=False,add_generation_prompt=True)
    full=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=False)
    p=tok(prompt,add_special_tokens=False,return_tensors='pt'); a=tok(full,add_special_tokens=False,return_tensors='pt')
    ids=a['input_ids'].squeeze(0); mask=a['attention_mask'].squeeze(0)
    if len(ids)>max_length: raise ValueError(f"{record.get('metadata',{}).get('window_id')} token length {len(ids)}>{max_length}")
    labels=ids.clone(); labels[:p['input_ids'].shape[1]]=-100; labels[mask==0]=-100
    if int((labels!=-100).sum())==0: raise ValueError('no assistant supervision tokens')
    return {'input_ids':ids,'attention_mask':mask,'labels':labels,'supervised':int((labels!=-100).sum())}


def summarize_tokenized(data):
    """Return a compact, JSON-safe summary of tokenized training examples."""
    lengths=[int(len(x['input_ids'])) for x in data]
    supervised=[int(x.get('supervised',0)) for x in data]
    if not lengths:
        return {
            'n':0,'tokens_min':0,'tokens_max':0,'tokens_p50':0.0,
            'supervised_total':0,
        }
    return {
        'n':len(lengths),
        'tokens_min':min(lengths),
        'tokens_max':max(lengths),
        'tokens_p50':float(np.median(lengths)),
        'supervised_total':int(sum(supervised)),
    }


def fast_kernel_status():
    """Report optional fast-kernel availability without importing heavy packages."""
    return {
        'fla':importlib.util.find_spec('fla') is not None,
        'causal_conv1d':importlib.util.find_spec('causal_conv1d') is not None,
        'platform':platform.platform(),
    }


def _load_model(model_dir,processor_only=False):
    from transformers import AutoProcessor, BitsAndBytesConfig
    processor=AutoProcessor.from_pretrained(str(model_dir),local_files_only=True)
    if processor_only: return processor,None
    try:
        from transformers import AutoModelForMultimodalLM as AutoModel
    except ImportError:
        from transformers import AutoModelForCausalLM as AutoModel
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
    model=AutoModel.from_pretrained(str(model_dir),quantization_config=q,torch_dtype=torch.float16,device_map={'':0},low_cpu_mem_usage=True,local_files_only=True)
    model.config.use_cache=False
    from peft import LoraConfig,TaskType,get_peft_model,prepare_model_for_kbit_training
    model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
    model=get_peft_model(model,LoraConfig(task_type=TaskType.CAUSAL_LM,r=LORA_R,lora_alpha=LORA_ALPHA,lora_dropout=LORA_DROPOUT,target_modules='all-linear',bias='none'))
    return processor,model


def _batch(item,device):
    return {k:item[k].unsqueeze(0).to(device) for k in ('input_ids','attention_mask','labels')}

@torch.no_grad()
def eval_loss(model,data,device):
    model.eval(); num=0.; den=0
    for x in data:
        b=_batch(x,device); out=model(**b); n=x['supervised']; num+=float(out.loss.detach().cpu())*n; den+=n
    model.train(); return num/max(den,1)


def train_phase(model_dir,train_jsonl,eval_jsonl,out_dir,max_epochs=MAX_EPOCHS,patience=PATIENCE,fixed_epochs=None):
    if not torch.cuda.is_available(): raise RuntimeError('V3.1 QLoRA requires CUDA GPU + bitsandbytes')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    tr=read_jsonl(train_jsonl); ev=[] if eval_jsonl is None else read_jsonl(eval_jsonl)
    processor,model=_load_model(model_dir); td=[tokenize_record(processor,r) for r in tr]; ed=[tokenize_record(processor,r) for r in ev]
    device=torch.device('cuda:0'); opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=LR,weight_decay=WD)
    epochs=int(fixed_epochs or max_epochs); steps_epoch=math.ceil(len(td)/GRAD_ACCUM); total=max(1,steps_epoch*epochs)
    from transformers import get_linear_schedule_with_warmup
    sched=get_linear_schedule_with_warmup(opt,max(1,round(total*WARMUP)),total)
    history=[]; best=float('inf'); best_epoch=epochs if not ed else 0; wait=0; best_dir=out/'best_adapter'
    for epoch in range(1,epochs+1):
        order=np.random.default_rng(SEED+epoch).permutation(len(td)); opt.zero_grad(set_to_none=True); losses=[]
        for pos,ii in enumerate(order,1):
            b=_batch(td[int(ii)],device); loss=model(**b).loss/GRAD_ACCUM; loss.backward(); losses.append(float(loss.detach().cpu())*GRAD_ACCUM)
            if pos%GRAD_ACCUM==0 or pos==len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        score=eval_loss(model,ed,device) if ed else float(np.mean(losses))
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'eval_loss':score})
        if not ed: continue
        if score<best:
            best=score; best_epoch=epoch; wait=0
            if best_dir.exists(): shutil.rmtree(best_dir)
            model.save_pretrained(best_dir); processor.save_pretrained(best_dir)
        else:
            wait+=1
            if wait>=patience: break
    if fixed_epochs is not None:
        best_epoch=epochs; best_dir=out/'refit_adapter'; model.save_pretrained(best_dir); processor.save_pretrained(best_dir)
    (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    return {'best_epoch':best_epoch,'best_eval_loss':best if ed else None,'adapter_dir':str(best_dir),'history':history}


def run_v31_qwen_training(project_root,model_dir=None):
    root=Path(project_root); data=root/'data'/'v31_qwen'; report=root/'reports'/'v31'/'qwen_training'; models=root/'models'/'v31_qwen'
    model_dir=Path(model_dir or os.environ.get('QWEN_BASE_MODEL_DIR',r'K:\ollama\Qwen3.5-9B-hf'))
    tr=data/'qwen_train_2016_2017.jsonl'; dev=data/'qwen_dev_2018.jsonl'; refit=data/'qwen_refit_2016_2018.jsonl'
    assert_target_years(read_jsonl(tr),max_year=2017); assert_target_years(read_jsonl(dev),min_year=2018,max_year=2018); assert_target_years(read_jsonl(refit),max_year=2018)
    sel=train_phase(model_dir,tr,dev,models/'selection')
    # unload to ensure refit starts from base, not selected adapter
    gc.collect(); torch.cuda.empty_cache()
    ref=train_phase(model_dir,refit,None,models/'refit',fixed_epochs=sel['best_epoch'])
    report.mkdir(parents=True,exist_ok=True); manifest={'selection':sel,'refit':ref,'model_dir':str(model_dir),'no_2019_targets_used':True}
    (report/'training_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest

if __name__=='__main__':
    here=Path(__file__).resolve().parents[1]; run_v31_qwen_training(here)
