use crate::{
    air, backend, model::signed, Error, Model, Statement, MAGIC, MAX_PROOF_BYTES, VERSION,
};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

/// Exact integer reference executor for this bounded profile. Rounding ties
/// toward +infinity, including negative values. No floating-point arithmetic.
pub fn evaluate(model: &Model, input: [i32; 2]) -> Result<[i32; 2], Error> {
    validate_input(model, input)?;
    let mut values = input;
    for layer in 0..2 {
        values = dense(model, values, layer)?.map(|(q, _)| if layer == 0 { q.max(0) } else { q });
    }
    Ok(values)
}

fn validate_input(model: &Model, input: [i32; 2]) -> Result<(), Error> {
    model.validate()?;
    if input.iter().any(|x| !(-128..=127).contains(x)) {
        return Err(Error::InvalidStatement);
    }
    Ok(())
}

fn dense(model: &Model, values: [i32; 2], layer: usize) -> Result<[(i32, i32); 2], Error> {
    let mut result = [(0, 0); 2];
    for (j, slot) in result.iter_mut().enumerate() {
        let raw: i64 = (0..2)
            .map(|i| i64::from(values[i]) * i64::from(model.weights[layer][j][i]))
            .sum();
        let q = i32::try_from((raw + 8).div_euclid(16) + i64::from(model.biases[layer][j]))
            .map_err(|_| Error::InvalidStatement)?;
        if !(-256..=255).contains(&q) {
            return Err(Error::InvalidStatement);
        }
        *slot = (q, (raw + 8).rem_euclid(16) as i32);
    }
    Ok(result)
}

fn execute(model: &Model, input: [i32; 2]) -> Result<([i32; 2], Vec<backend::F>), Error> {
    validate_input(model, input)?;
    let mut row = vec![backend::F::ZERO; air::WIDTH];
    let mut write_bits = |start: usize, len: usize, value: i32| {
        for i in 0..len {
            row[start + i] = backend::F::from_u32(((value as u32) >> i) & 1);
        }
    };
    for (i, &x) in input.iter().enumerate() {
        write_bits(i * 8, 8, x + 128);
    }
    let mut values = input;
    let mut hidden = [0; 2];
    for layer in 0..2 {
        let mut next = [0; 2];
        for (j, &(q, rem)) in dense(model, values, layer)?.iter().enumerate() {
            write_bits(air::Q_START + (layer * 2 + j) * 9, 9, q + 256);
            write_bits(air::REM_START + (layer * 2 + j) * 4, 4, rem);
            next[j] = if layer == 0 { q.max(0) } else { q };
        }
        values = next;
        if layer == 0 {
            hidden = values;
        }
    }
    for i in 0..2 {
        row[air::H_START + i] = signed(hidden[i]);
    }
    Ok((values, row))
}

fn make_trace(row: Vec<backend::F>) -> RowMajorMatrix<backend::F> {
    RowMajorMatrix::new(row.repeat(1 << air::TRACE_LOG_ROWS), air::WIDTH)
}

/// Return a public statement and portable proof. Only this optional module
/// executes the network or materializes the private trace.
pub fn prove(
    model: &Model,
    input: [i32; 2],
    context: [u8; 32],
) -> Result<(Statement, Vec<u8>), Error> {
    let (output, row) = execute(model, input)?;
    let statement = Statement {
        version: VERSION,
        model_id: model.id()?,
        context,
        input,
        output,
    };
    let proof = p3_uni_stark::prove(
        &backend::config(),
        &air::InferenceAir,
        make_trace(row),
        &model.public_values(&statement),
    );
    let mut bytes = MAGIC.to_vec();
    bytes.extend(postcard::to_allocvec(&proof).map_err(|_| Error::ProofEncoding)?);
    if bytes.len() > MAX_PROOF_BYTES {
        return Err(Error::ProofTooLarge);
    }
    Ok((statement, bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verify;

    fn model() -> Model {
        Model {
            version: VERSION,
            fractional_bits: 4,
            weights: [[[8, -4], [-8, 8]], [[8, -8], [4, 8]]],
            biases: [[1, -2], [3, -4]],
        }
    }

    #[test]
    fn portable_honest_proof_and_statement_binding() {
        let m = model();
        let (statement, proof) = prove(&m, [17, -9], [7; 32]).unwrap();
        assert_eq!(statement.output, evaluate(&m, statement.input).unwrap());
        verify(&m, m.id().unwrap(), &statement, &proof).unwrap();
        let mut changed = statement.clone();
        changed.input[0] += 1;
        assert!(verify(&m, m.id().unwrap(), &changed, &proof).is_err());
        changed = statement.clone();
        changed.output[0] += 1;
        assert!(verify(&m, m.id().unwrap(), &changed, &proof).is_err());
        changed = statement.clone();
        changed.context[0] ^= 1;
        assert!(verify(&m, m.id().unwrap(), &changed, &proof).is_err());
        changed = statement.clone();
        changed.version += 1;
        assert!(verify(&m, m.id().unwrap(), &changed, &proof).is_err());
        let mut other = m.clone();
        other.weights[0][0][0] -= 1;
        assert!(verify(&other, m.id().unwrap(), &statement, &proof).is_err());
        changed = statement.clone();
        changed.model_id = other.id().unwrap();
        // Even when the attacker supplies a new manifest and matching id, old
        // proof bytes must not verify for that new model/statement.
        assert!(verify(&other, other.id().unwrap(), &changed, &proof).is_err());
    }

    #[test]
    fn rejects_dishonest_witnesses_even_when_the_prover_is_modified() {
        let m = model();
        let (output, row) = execute(&m, [17, -9]).unwrap();
        let statement = Statement {
            version: VERSION,
            model_id: m.id().unwrap(),
            context: [0; 32],
            input: [17, -9],
            output,
        };
        // An attacker may change both the claimed output and the corresponding
        // witness bits. The rounding/matmul relation must still reject it.
        let mut false_statement = statement.clone();
        false_statement.output[0] += 1;
        let mut dishonest_row = row.clone();
        let encoded_q = (false_statement.output[0] + 256) as u32;
        for bit in 0..9 {
            dishonest_row[air::Q_START + 18 + bit] = backend::F::from_u32((encoded_q >> bit) & 1);
        }
        let forged = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            p3_uni_stark::prove(
                &backend::config(),
                &air::InferenceAir,
                make_trace(dishonest_row),
                &m.public_values(&false_statement),
            )
        }));
        if let Ok(proof) = forged {
            let mut bytes = MAGIC.to_vec();
            bytes.extend(postcard::to_allocvec(&proof).unwrap());
            assert!(verify(&m, m.id().unwrap(), &false_statement, &bytes).is_err());
        }
        // Corrupt input, matrix/rounding output, remainder, ReLU linkage and
        // booleanity, then deliberately run the backend prover anyway.
        for (col, value) in [
            (0, 2),
            (air::Q_START, 2),
            (air::REM_START, 2),
            (air::H_START, 99),
            (air::Q_START, 1),
            (air::REM_START, 1),
        ] {
            let mut bad = row.clone();
            bad[col] = if value == 1 {
                backend::F::ONE - bad[col]
            } else {
                backend::F::from_u32(value)
            };
            let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                p3_uni_stark::prove(
                    &backend::config(),
                    &air::InferenceAir,
                    make_trace(bad),
                    &m.public_values(&statement),
                )
            }));
            if let Ok(forged) = attempt {
                let mut bytes = MAGIC.to_vec();
                bytes.extend(postcard::to_allocvec(&forged).unwrap());
                assert!(verify(&m, m.id().unwrap(), &statement, &bytes).is_err());
            }
            // Debug backend assertions may abort proof construction; release
            // tests additionally exercise actual rejection of generated proofs.
        }
    }
}
