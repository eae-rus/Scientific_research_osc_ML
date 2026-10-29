"""Real-data integration test: uninterrupted vs interrupted/resumed GPU run."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import torch

ROOT=Path(__file__).resolve().parents[2]


def main():
    root=Path(tempfile.mkdtemp(prefix="hierarchical_resume_",dir=ROOT/"tmp"))
    common=[sys.executable,"-X","utf8",str(ROOT/"scripts/phase5_experiments/run_phase5_hierarchical.py"),
            "--samples-per-epoch","6","--validation-samples","3","--epochs","1",
            "--workers","2","--target-tokens","240"]
    def run(name,*args):
        command=common+["--output",str(root/name),*args]
        completed=subprocess.run(command,cwd=ROOT,text=True,encoding="utf-8",capture_output=True)
        (root/(name+('_resume' if '--resume' in args else '')+'.log')).write_text(completed.stdout+completed.stderr,encoding="utf-8")
        if completed.returncode: raise RuntimeError(completed.stdout+completed.stderr)
    run("continuous")
    run("resumed","--max-steps","1")
    run("resumed","--resume")
    a=torch.load(root/"continuous/latest_checkpoint.pt",map_location="cpu",weights_only=False)
    b=torch.load(root/"resumed/latest_checkpoint.pt",map_location="cpu",weights_only=False)
    delta=max((a["model"][k]-b["model"][k]).abs().max().item() for k in a["model"])
    read=lambda name:[json.loads(line) for line in (root/name/"training_log.jsonl").read_text().splitlines()]
    ar,br=read("continuous"),read("resumed")
    windows_equal=[r["windows"] for r in ar if r["kind"]=="train"]==[r["windows"] for r in br if r["kind"]=="train"]
    loss_error=max(abs(x["loss"]-y["loss"]) for x,y in zip(ar,br) if x["kind"]=="train")
    state_equal=(a["epoch"],a["cursor"],a["step"],a["best"])==(b["epoch"],b["cursor"],b["step"],b["best"])
    result={"max_parameter_error":delta,"max_loss_error":loss_error,"windows_equal":windows_equal,
            "progress_and_best_equal":state_equal,"validation_continuous":ar[-1],"validation_resumed":br[-1],
            "model_data_sha256":a["contract"]["model_data_sha256"],
            "microbatches_per_step":[len(r["windows"]) for r in ar if r["kind"]=="train"],
            "artifacts":str(root),"device":"cuda" if torch.cuda.is_available() else "cpu"}
    destination=ROOT/"reports/phase5/hierarchical_v1_resume_check.json"
    destination.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)
    assert windows_equal and state_equal and delta<=1e-6 and loss_error<=1e-6
    # These checkpoints are exclusively temporary diagnostics created above.
    # Keep logs/configs/report, not hundreds of MB of duplicate test weights.
    for name in ("continuous","resumed"):
        for checkpoint_path in (root/name).glob("*.pt"):
            checkpoint_path.unlink()


if __name__=="__main__": main()
