#![cfg(feature = "prover")]

use zkie_verified::{evaluate, prove, verify, Error, Model, ModelId, Statement, VERSION};

fn reference(model: &Model, input: [i32; 2]) -> [i32; 2] {
    // Deliberately different integer widths and rounding algorithm from the
    // executor's floor((raw+8)/16). Truncation plus signed-remainder correction.
    let mut v = input.map(i128::from);
    for layer in 0..2 {
        v = std::array::from_fn(|j| {
            let raw = v[0] * i128::from(model.weights[layer][j][0])
                + v[1] * i128::from(model.weights[layer][j][1]);
            let mut q = raw / 16;
            if raw % 16 >= 8 {
                q += 1;
            }
            if raw % 16 < -8 {
                q -= 1;
            }
            q += i128::from(model.biases[layer][j]);
            if layer == 0 {
                q.max(0)
            } else {
                q
            }
        });
    }
    v.map(|x| i32::try_from(x).unwrap())
}

fn fixture() -> Model {
    serde_json::from_str(include_str!("../fixtures/model.json")).unwrap()
}

#[test]
fn every_input_pair_matches_independent_reference_for_three_models() {
    let models = [
        fixture(),
        Model {
            version: VERSION,
            fractional_bits: 4,
            weights: [[[8, 8], [-8, -8]], [[8, 8], [-8, -8]]],
            biases: [[16, -16], [16, -16]],
        },
        Model {
            version: VERSION,
            fractional_bits: 4,
            weights: [[[0, 0]; 2]; 2],
            biases: [[-16, 16], [16, -16]],
        },
    ];
    for m in models {
        for x in -128..=127 {
            for y in -128..=127 {
                let actual = evaluate(&m, [x, y]).unwrap();
                assert_eq!(actual, reference(&m, [x, y]), "input={x},{y} model={m:?}");
            }
        }
    }
}

#[test]
fn honest_boundary_and_negative_tie_proofs_accept() {
    let m = fixture();
    for input in [
        [-128, -128],
        [127, 127],
        [-128, 127],
        [127, -128],
        [-1, 0],
        [0, -1],
        [0, 0],
        [1, 0],
    ] {
        let (statement, proof) = prove(&m, input, [3; 32]).unwrap();
        assert_eq!(statement.output, reference(&m, input));
        verify(&m, m.id().unwrap(), &statement, &proof).unwrap();
    }
}

#[test]
fn domains_and_strict_model_encoding_are_enforced() {
    let m = fixture();
    assert!(evaluate(&m, [-129, 0]).is_err());
    assert!(evaluate(&m, [0, 128]).is_err());
    for (version, bits, weight, bias) in [
        (0, 4, 8, 1),
        (VERSION, 3, 8, 1),
        (VERSION, 4, -9, 1),
        (VERSION, 4, 9, 1),
        (VERSION, 4, 8, -17),
        (VERSION, 4, 8, 17),
    ] {
        let mut bad = m.clone();
        bad.version = version;
        bad.fractional_bits = bits;
        bad.weights[0][0][0] = weight;
        bad.biases[0][0] = bias;
        assert_eq!(bad.validate(), Err(Error::InvalidModel));
        assert!(prove(&bad, [0, 0], [0; 32]).is_err());
    }
    let id = m.id().unwrap();
    assert_eq!(ModelId::from_hex(&id.to_hex()).unwrap(), id);
    assert!(ModelId::from_hex("é").is_err());
    assert!(ModelId::from_hex(&"g".repeat(64)).is_err());
    let compact = serde_json::to_string(&m).unwrap();
    let reparsed: Model = serde_json::from_str(&compact).unwrap();
    assert_eq!(reparsed.id().unwrap(), id);
    let unknown = compact.replacen('{', "{\"unexpected\":0,", 1);
    assert!(serde_json::from_str::<Model>(&unknown).is_err());
    let duplicate = compact.replacen('{', "{\"version\":1,", 1);
    assert!(serde_json::from_str::<Model>(&duplicate).is_err());
    for output in [[-257, 0], [0, 256]] {
        assert_eq!(
            Statement {
                version: VERSION,
                model_id: id,
                context: [0; 32],
                input: [0, 0],
                output
            }
            .validate(),
            Err(Error::InvalidStatement)
        );
    }
}
