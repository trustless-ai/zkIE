#!/usr/bin/env python3
"""Witness/weight-free matrix product acceptance in a verifier-only process."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def run(args,cwd,success=True):
    p=subprocess.run([str(a) for a in args],cwd=cwd,capture_output=True,text=True,timeout=30)
    assert p.returncode==(0 if success else 1),(p.returncode,p.stderr)
    return json.loads(p.stdout) if success else None


def main():
    if len(sys.argv)!=3:
        raise SystemExit("usage: accept_linear.py PROVER_LINEAR_BINARY VERIFIER_ONLY_LINEAR_BINARY")
    prover,verifier=[Path(p).resolve(strict=True) for p in sys.argv[1:]]
    with tempfile.TemporaryDirectory(prefix="zkie-linear-") as temporary:
        root=Path(temporary);producer=root/"producer";consumer=root/"consumer"
        producer.mkdir();consumer.mkdir()
        m,k,n=4,16,8
        weights=[i%17-8 for i in range(k*n)]
        inputs=[i%13-6 for i in range(m*k)]
        expected=[sum(inputs[i*k+t]*weights[t*n+j] for t in range(k)) for i in range(m) for j in range(n)]
        request={"m":m,"input":inputs,"context":[9]*32}
        (producer/"weights.json").write_text(json.dumps({"k":k,"n":n,"weights":weights}))
        (producer/"request.json").write_text(json.dumps(request))
        prepared=run([prover,"prepare","weights.json","prepared"],producer)
        proving=run([prover,"prove","weights.json","request.json","bundle"],producer)
        assert prepared["model_id"]==proving["model_id"]
        assert proving["output"]==expected
        for name in ["key.bin","statement.json","proof.bin"]:
            shutil.copyfile(producer/"bundle"/name,consumer/name)
        key=(consumer/"key.bin").read_bytes()
        expected_id=hashlib.sha256(b"zkie/committed-matmul/v2\0"+key).hexdigest()
        assert expected_id==prepared["model_id"] and len(key)<256
        (consumer/"request.json").write_text(json.dumps(request))
        shutil.copy2(verifier,consumer/"verify")
        shutil.rmtree(producer)
        args=[consumer/"verify","verify","key.bin",expected_id,"request.json","statement.json","proof.bin"]
        verified=run(args,consumer)
        assert verified["output"]==expected
        statement=json.loads((consumer/"statement.json").read_text())
        statement["output"][0]+=1
        (consumer/"statement.json").write_text(json.dumps(statement));run(args,consumer,False)
        assert not (consumer/"weights.json").exists()
        print(json.dumps({"acceptance":"passed","weight_files_present":False,
                          "prover":proving,"verifier":verified},indent=2))


if __name__=="__main__":main()
