#!/usr/bin/env python3
"""Bounded full-architecture acceptance; no real weights or download required."""
import json
import pathlib
import shutil
import struct
import subprocess
import sys
import tempfile


def call(argv, cwd, ok=True):
    result = subprocess.run([str(x) for x in argv], cwd=cwd, capture_output=True, text=True)
    if (result.returncode == 0) != ok:
        raise AssertionError(f"unexpected exit {result.returncode}: {result.stderr[:2000]}")
    return json.loads(result.stdout) if ok else None


def nearest(a, d):
    return (a + d // 2) // d


def mm(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def norm(x, c):
    result = []
    for row in x:
        mean = nearest(sum(row[:c['real_width']]), c['real_width'])
        centered = [v - mean for v in row]
        var = nearest(sum(v*v for v in centered[:c['real_width']]), c['real_width'])
        raw = nearest(var, 1 << 18)
        fine, coarse = 1 << 20, 1 << 22
        end = fine + coarse * 256
        idx = max(raw, 0) if raw < fine else fine + (raw-fine)//256 if raw < end else fine+coarse+(raw-end)//65536
        # The fixture's endorsed rstd table contains only this entry.
        assert idx == 0
        result.append([nearest(v * 65536 * 65536, 1 << 32) for v in centered])
    return result


def reference(flat, c):
    x = [flat[i:i+c['width']] for i in range(0, len(flat), c['width'])]
    exp = [1, 2, 3, 4, 5]
    gelu = list(range(-2, 6))
    for _ in range(c['layers']):
        h = norm(x, c)
        accumulated = [[0]*c['width'] for _ in x]
        for head in range(c['heads']):
            start, end = head*c['head_width'], (head+1)*c['head_width']
            v = [row[start:end] for row in h]
            qk = [[nearest(y*23170, 65536) for y in row] for row in v]
            score = [[nearest(y, 65536) for y in row] for row in mm(qk, list(map(list, zip(*qk))))]
            masked = [[y - ((1 << 30) if j > i else 0) for j, y in enumerate(row)] for i, row in enumerate(score)]
            e = [[exp[min(max(y-max(row)+c['exp_offset'], 0), len(exp)-1)] for y in row] for row in masked]
            probs = [[nearest(y*65536, sum(row)) for y in row] for row in e]
            attn = [[nearest(y, 65536) for y in row] for row in mm(probs, v)]
            for acc, row in zip(accumulated, attn):
                acc[start:end] = row
        residual = [[a+b for a, b in zip(row, attn)] for row, attn in zip(x, accumulated)]
        h = norm(residual, c)
        # Identity Q16 fc and projection matrices, zero biases.
        act = [[gelu[min(max(y+c['gelu_offset'], 0), len(gelu)-1)] for y in row] for row in h]
        x = [[a+b for a, b in zip(row, activation)] for row, activation in zip(residual, act)]
    return [y*65536 for row in norm(x, c) for y in row]


def main():
    prover, verifier = (pathlib.Path(x).resolve() for x in sys.argv[1:])
    c = dict(sequence=2, width=4, real_width=3, hidden=4, heads=2,
             head_width=2, layers=2, vocabulary=4, gelu_offset=2, exp_offset=2)
    request = dict(input=[2, -2, 1, 0, 4, -4, 2, 0], context=[19]*32)
    with tempfile.TemporaryDirectory(prefix='zkie-transformer-') as tmp:
        root = pathlib.Path(tmp)
        producer, consumer = root/'producer', root/'consumer'
        producer.mkdir(); consumer.mkdir()
        weights = producer/'weights'; weights.mkdir()
        def put(name, values):
            (weights/f'{name}_i32.bin').write_bytes(struct.pack('<'+'i'*len(values), *values))
        identity = [65536 if i//4 == i%4 else 0 for i in range(16)]
        put('exp_table', [1,2,3,4,5]); put('gelu_table', list(range(-2,6))); put('rsqrt_table', [65536])
        for layer in range(c['layers']):
            for name in ['q','k','v','o_proj','fc','proj']:
                put(f'L{layer}_{name}_w', identity)
                put(f'L{layer}_{name}_b', [0]*4)
            for name in ['ln1','ln2']:
                put(f'L{layer}_{name}_w', [65536]*4); put(f'L{layer}_{name}_b', [0]*4)
        put('ln_f_w', [65536]*4); put('ln_f_b', [0]*4); put('lm_head_w', identity)
        config = producer/'config.json'; config.write_text(json.dumps(c))
        req = producer/'request.json'; req.write_text(json.dumps(request))
        inspection = call([prover, 'inspect', config, weights], producer)
        trusted = call([prover,'prepare',config,weights,producer/'trusted'], producer)
        result = call([prover,'prove',config,weights,req,producer/'proof'], producer)
        assert result['output'] == reference(request['input'], c), (result['output'], reference(request['input'], c))
        for name in ['statement.json','proof.bin']:
            shutil.copyfile(producer/'proof'/name, consumer/name)
        shutil.copyfile(producer/'trusted'/'key.bin', consumer/'key.bin')
        shutil.copyfile(req, consumer/'request.json')
        binary = consumer/'verify'; shutil.copyfile(verifier,binary); binary.chmod(0o755)
        # Reject malformed quantized files instead of ignoring a partial i32.
        (weights/'exp_table_i32.bin').write_bytes(b'\x01')
        call([prover, 'inspect', config, weights], producer, False)
        # Verification has no access to the producer's weights or proving binary.
        shutil.rmtree(producer)
        command = [binary,'verify',consumer/'key.bin',trusted['model_id'],consumer/'request.json',consumer/'statement.json',consumer/'proof.bin']
        accepted = call(command, consumer)
        assert accepted['output'] == result['output']
        attacks = []
        statement_path = consumer/'statement.json'; statement = json.loads(statement_path.read_text())
        for field in ['output','input','context']:
            modified = json.loads(json.dumps(statement)); modified[field][0] += 1
            statement_path.write_text(json.dumps(modified)); call(command,consumer,False); attacks.append(field)
        statement_path.write_text(json.dumps(statement))
        proof_path=consumer/'proof.bin'; proof=proof_path.read_bytes()
        for name, data in [('truncated',proof[:-1]),('trailing',proof+b'x'),('corrupt',proof[:len(proof)//2]+bytes([proof[len(proof)//2]^1])+proof[len(proof)//2+1:])]:
            proof_path.write_bytes(data); call(command,consumer,False); attacks.append(name)
        proof_path.write_bytes(proof)
        wrong_id=command.copy(); wrong_id[3]='00'*32;call(wrong_id,consumer,False);attacks.append('untrusted-key-id')
        key_path=consumer/'key.bin';key=key_path.read_bytes();key_path.write_bytes(key[:-1]);call(command,consumer,False);attacks.append('truncated-key')
        print(json.dumps(dict(acceptance='passed', fixture='two-layer/two-head integer Transformer; synthetic weights',
             independent_reference=True, weight_files_present=False, inspection=inspection,
             prover=result, verifier=accepted, rejected_attacks=attacks),indent=2))


if __name__ == '__main__':
    main()
