use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;
use std::sync::Arc;

use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner};
use halo2_proofs::dev::MockProver;
use halo2_proofs::plonk::{Circuit, ConstraintSystem, ErrorFront};
use zkie_compiler::circuit_binding::to_assembler_program;
use zkie_compiler::dag::Shard;
use zkie_compiler::graph_compiler::{CompiledInstruction, CompiledProgram, Register};
use zkie_compiler::onnx_parser::WeightTensor;
use zkie_core::assembler::{AssemblerChip, AssemblerConfig, AssemblerProgram};
use zkie_core::chips::layer_norm::RsqrtDomain;
use zkie_core::field_convert::Fr;
use zkie_core::fixed_point::I18;
use zkie_core::isa::{EltwiseOp, Instruction};
use zkie_core::program_circuit::{AssemblerCircuit, AssemblerCircuitParams};
use zkie_prover::{
    BackendError, BoundaryShapeManifest, CpuWitnessInputs, Digest32, ModelVisibility,
    ProofFlavorId, RunIdentity, ShardIdentity, WitnessArtifact, WitnessBackend, WitnessJob,
    ZkieIsaCpuWitnessBackend, MAX_CPU_WITNESS_ARTIFACT_BYTES, MAX_CPU_WITNESS_DOMAIN_POINTS,
    MAX_CPU_WITNESS_INPUT_BYTES, MAX_CPU_WITNESS_VALUES_PER_TENSOR,
};

fn raw(values: &[i64]) -> Vec<I18> {
    values.iter().copied().map(I18::from_raw).collect()
}

fn tensor(data: &[f32]) -> WeightTensor {
    WeightTensor {
        shape: vec![data.len()],
        data: data.to_vec(),
    }
}

fn linear_program() -> CompiledProgram {
    CompiledProgram {
        instructions: vec![
            CompiledInstruction {
                instruction: Instruction::DotGeneral {
                    m: 2,
                    n: 2,
                    k: 2,
                    batch_dims: vec![],
                    trans_a: false,
                    trans_b: false,
                },
                inputs: vec![
                    Register::GraphInput("x".into()),
                    Register::Weight("w".into()),
                ],
                output_name: "dot".into(),
            },
            CompiledInstruction {
                instruction: Instruction::Eltwise { op: EltwiseOp::Add },
                inputs: vec![Register::Virtual(0), Register::Weight("bias".into())],
                output_name: "added".into(),
            },
            CompiledInstruction {
                instruction: Instruction::Eltwise { op: EltwiseOp::Mul },
                inputs: vec![Register::Virtual(1), Register::Weight("scale".into())],
                output_name: "scaled".into(),
            },
        ],
        weights: HashMap::from([
            ("w".into(), tensor(&[1.0, 0.5, -0.5, 1.0])),
            ("bias".into(), tensor(&[0.125, -0.25])),
            ("scale".into(), tensor(&[0.5])),
        ]),
        graph_inputs: vec!["x".into()],
        graph_outputs: vec![("scaled".into(), Register::Virtual(2))],
    }
}

fn whole_shard(program: &CompiledProgram) -> Shard {
    Shard {
        id: 7,
        name: "fixture".into(),
        range: 0..program.instructions.len(),
        inputs: vec![],
        outputs: vec![Register::Virtual(program.instructions.len() - 1)],
    }
}

fn backend(
    program: CompiledProgram,
    shard: Shard,
    domains: HashMap<(usize, u64), RsqrtDomain>,
) -> ZkieIsaCpuWitnessBackend {
    let shapes = test_boundary_shapes(&program, &shard);
    ZkieIsaCpuWitnessBackend::new(Arc::new(program), shard, domains, run_identity(), shapes)
        .unwrap()
}

fn test_boundary_shapes(program: &CompiledProgram, shard: &Shard) -> BoundaryShapeManifest {
    fn output_len(program: &CompiledProgram, index: usize) -> usize {
        match &program.instructions[index].instruction {
            Instruction::DotGeneral { m, n, .. } => m * n,
            Instruction::RmsNorm { dim, .. } => *dim,
            Instruction::Reduce { .. } => 1,
            Instruction::Softmax { axis_dim } => *axis_dim,
            Instruction::Eltwise { .. } => program.instructions[index]
                .inputs
                .iter()
                .find_map(|register| match register {
                    Register::Weight(name) => Some(program.weights[name].data.len()),
                    Register::Virtual(source) if *source < index => {
                        Some(output_len(program, *source))
                    }
                    _ => None,
                })
                .unwrap_or(1),
            _ => 1,
        }
    }
    let mut graph = BTreeMap::new();
    let mut virtuals = BTreeMap::new();
    for (index, instruction) in program.instructions[shard.range.clone()].iter().enumerate() {
        let global = shard.range.start + index;
        for (position, register) in instruction.inputs.iter().enumerate() {
            let len = match register {
                Register::GraphInput(_) => match &instruction.instruction {
                    Instruction::DotGeneral {
                        m, n, k, trans_a, ..
                    } => {
                        if position == 0 {
                            if *trans_a {
                                k * m
                            } else {
                                m * k
                            }
                        } else {
                            k * n
                        }
                    }
                    Instruction::RmsNorm { dim, .. } => *dim,
                    Instruction::Reduce { .. } => 1,
                    Instruction::Softmax { axis_dim } => *axis_dim,
                    Instruction::Eltwise { .. } => instruction
                        .inputs
                        .iter()
                        .find_map(|other| match other {
                            Register::Weight(name) => Some(program.weights[name].data.len()),
                            Register::Virtual(source) if *source < global => {
                                Some(output_len(program, *source))
                            }
                            _ => None,
                        })
                        .unwrap_or(1),
                    _ => 1,
                },
                Register::Virtual(source) if *source < shard.range.start => {
                    output_len(program, *source)
                }
                _ => continue,
            };
            match register {
                Register::GraphInput(name) => {
                    graph.insert(name.clone(), len);
                }
                Register::Virtual(source) => {
                    virtuals.insert(*source, len);
                }
                Register::Weight(_) => {}
            }
        }
    }
    BoundaryShapeManifest::new(graph, virtuals).unwrap()
}

#[derive(Clone)]
struct CaptureAssemblerCircuit {
    params: AssemblerCircuitParams,
    program: AssemblerProgram,
    captured: Rc<RefCell<Option<Vec<Vec<I18>>>>>,
}

impl Circuit<Fr> for CaptureAssemblerCircuit {
    type Config = AssemblerConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = AssemblerCircuitParams;

    fn without_witnesses(&self) -> Self {
        self.clone()
    }

    fn params(&self) -> Self::Params {
        self.params.clone()
    }

    fn configure_with_params(
        meta: &mut ConstraintSystem<Fr>,
        params: Self::Params,
    ) -> Self::Config {
        AssemblerChip::configure_with_rms_norm_domains(
            meta,
            &params.instructions,
            &params.rms_norm_domains,
        )
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        Self::configure_with_params(meta, AssemblerCircuitParams::default())
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), ErrorFront> {
        let chip = AssemblerChip::construct(config);
        chip.load_rms_norm_tables(layouter.namespace(|| "tables"))?;
        let values = chip
            .assign(layouter.namespace(|| "program"), &self.program)
            .map_err(|_| ErrorFront::Synthesis)?;
        *self.captured.borrow_mut() = Some(values);
        Ok(())
    }
}

fn assembler_virtuals(
    program: &CompiledProgram,
    inputs: &HashMap<String, Vec<I18>>,
    domains: HashMap<(usize, u64), RsqrtDomain>,
) -> Vec<Vec<I18>> {
    let assembler = to_assembler_program(program, inputs).unwrap();
    let base = AssemblerCircuit::new(assembler.clone(), domains);
    let captured = Rc::new(RefCell::new(None));
    let circuit = CaptureAssemblerCircuit {
        params: base.params().clone(),
        program: assembler,
        captured: Rc::clone(&captured),
    };
    MockProver::run(13, &circuit, vec![])
        .unwrap()
        .assert_satisfied();
    let values = captured.borrow().clone().unwrap();
    values
}

#[test]
fn dot_add_and_mul_match_exact_literals_and_every_assembler_virtual() {
    let program = linear_program();
    let graph_inputs = HashMap::from([(
        "x".into(),
        raw(&[
            500_000_000_000_000_000,
            -250_000_000_000_000_000,
            1_000_000_000_000_000_000,
            500_000_000_000_000_000,
        ]),
    )]);
    let got = backend(program.clone(), whole_shard(&program), HashMap::new())
        .execute(&CpuWitnessInputs::new(
            graph_inputs.clone(),
            BTreeMap::new(),
        ))
        .unwrap();

    let expected = [
        vec![
            625_000_000_000_000_000,
            0,
            750_000_000_000_000_000,
            1_000_000_000_000_000_000,
        ],
        vec![
            750_000_000_000_000_000,
            -250_000_000_000_000_000,
            875_000_000_000_000_000,
            750_000_000_000_000_000,
        ],
        vec![
            375_000_000_000_000_000,
            -125_000_000_000_000_000,
            437_500_000_000_000_000,
            375_000_000_000_000_000,
        ],
    ];
    let circuit = assembler_virtuals(&program, &graph_inputs, HashMap::new());
    for (index, expected_raw) in expected.iter().enumerate() {
        assert_eq!(
            got.virtual_value(index)
                .unwrap()
                .iter()
                .map(I18::raw)
                .collect::<Vec<_>>(),
            *expected_raw
        );
        assert_eq!(got.virtual_value(index).unwrap(), circuit[index].as_slice());
    }
    assert_eq!(got.output_value(2), got.virtual_value(2));
}

fn rms_program(xs: &[f64], weights: &[f32]) -> CompiledProgram {
    CompiledProgram {
        instructions: vec![CompiledInstruction {
            instruction: Instruction::RmsNorm {
                dim: xs.len(),
                epsilon_milli: 0,
            },
            inputs: vec![
                Register::GraphInput("x".into()),
                Register::Weight("weight".into()),
            ],
            output_name: "y".into(),
        }],
        weights: HashMap::from([("weight".into(), tensor(weights))]),
        graph_inputs: vec!["x".into()],
        graph_outputs: vec![("y".into(), Register::Virtual(0))],
    }
}

#[test]
#[allow(clippy::excessive_precision)] // Preserve the exact recorded FinText f64 fixture.
fn fintext_rms_norm_matches_exact_raw_fixture_and_assembler() {
    let xs = [
        -5.05007207393646240e-02,
        -1.55950617790222168,
        -2.67525017261505127e-02,
        -1.26703572273254395,
        4.33678776025772095e-01,
        -1.34162211418151855,
        -1.43549692630767822,
        8.94961804151535034e-02,
    ];
    let weights = [
        2.30960012413561344e-03,
        -1.45695207174867392e-03,
        3.29286698251962662e-03,
        -3.56230302713811398e-03,
        7.55330431275069714e-04,
        2.02001072466373444e-03,
        2.22328794188797474e-03,
        2.46914196759462357e-03,
    ];
    let program = rms_program(&xs, &weights.map(|v| v as f32));
    let graph_inputs = HashMap::from([(
        "x".into(),
        xs.iter().map(|v| I18::from_f64(*v).unwrap()).collect(),
    )]);
    let target = 1_012_174_153_815_396_446;
    let domains = HashMap::from([((8, 0), RsqrtDomain::RawAnchors(vec![target]))]);
    let got = backend(program.clone(), whole_shard(&program), domains.clone())
        .execute(&CpuWitnessInputs::new(
            graph_inputs.clone(),
            BTreeMap::new(),
        ))
        .unwrap();
    assert_eq!(
        got.virtual_value(0)
            .unwrap()
            .iter()
            .map(I18::raw)
            .collect::<Vec<_>>(),
        vec![
            -115_932_913_157_290,
            2_258_420_166_854_342,
            -87_561_051_159_643,
            4_486_339_112_312_736,
            325_594_851_639_641,
            -2_693_743_637_890_860,
            -3_172_271_561_176_149,
            219_645_818_643_126
        ]
    );
    assert_eq!(
        got.virtual_value(0).unwrap(),
        assembler_virtuals(&program, &graph_inputs, domains)[0].as_slice()
    );
}

#[test]
fn softmax_is_explicitly_unsupported() {
    let mut program = linear_program();
    program.instructions = vec![CompiledInstruction {
        instruction: Instruction::Softmax { axis_dim: 2 },
        inputs: vec![Register::GraphInput("x".into())],
        output_name: "y".into(),
    }];
    let err = backend(program.clone(), whole_shard(&program), HashMap::new())
        .execute(&CpuWitnessInputs::new(
            HashMap::from([("x".into(), raw(&[0, 0]))]),
            BTreeMap::new(),
        ))
        .unwrap_err();
    assert!(matches!(err, BackendError::UnsupportedInstruction { .. }));
}

#[test]
fn rejects_missing_forward_shape_and_overflow_with_typed_errors() {
    let program = linear_program();
    let missing = backend(program.clone(), whole_shard(&program), HashMap::new())
        .execute(&CpuWitnessInputs::new(HashMap::new(), BTreeMap::new()))
        .unwrap_err();
    assert!(matches!(missing, BackendError::MissingRegister { .. }));

    let mut forward_program = linear_program();
    forward_program.instructions[0].inputs[0] = Register::Virtual(1);
    let forward = ZkieIsaCpuWitnessBackend::new(
        Arc::new(forward_program.clone()),
        whole_shard(&forward_program),
        HashMap::new(),
        run_identity(),
        BoundaryShapeManifest::new(BTreeMap::new(), BTreeMap::new()).unwrap(),
    )
    .err()
    .unwrap();
    assert!(matches!(forward, BackendError::ForwardRegister { .. }));

    let shape = backend(program.clone(), whole_shard(&program), HashMap::new())
        .execute(&CpuWitnessInputs::new(
            HashMap::from([("x".into(), raw(&[0]))]),
            BTreeMap::new(),
        ))
        .unwrap_err();
    assert!(matches!(shape, BackendError::InvalidJob { .. }));

    let mut overflow_program = linear_program();
    overflow_program.instructions.truncate(1);
    let overflow_shard = whole_shard(&overflow_program);
    overflow_program
        .weights
        .insert("w".into(), tensor(&[2.0, 0.0, 0.0, 2.0]));
    let overflow = backend(overflow_program, overflow_shard, HashMap::new())
        .execute(&CpuWitnessInputs::new(
            HashMap::from([("x".into(), raw(&[i64::MAX, 0, 0, i64::MAX]))]),
            BTreeMap::new(),
        ))
        .unwrap_err();
    assert!(matches!(overflow, BackendError::ArithmeticOverflow { .. }));
}

#[test]
fn empty_eltwise_operand_returns_shape_error_instead_of_panicking() {
    let mut program = linear_program();
    program.instructions = vec![CompiledInstruction {
        instruction: Instruction::Eltwise { op: EltwiseOp::Add },
        inputs: vec![
            Register::GraphInput("empty".into()),
            Register::GraphInput("one".into()),
        ],
        output_name: "y".into(),
    }];
    program.graph_inputs = vec!["empty".into(), "one".into()];
    let err = backend(program.clone(), whole_shard(&program), HashMap::new())
        .execute(&CpuWitnessInputs::new(
            HashMap::from([
                ("empty".into(), vec![]),
                ("one".into(), raw(&[1_000_000_000_000_000_000])),
            ]),
            BTreeMap::new(),
        ))
        .unwrap_err();
    assert!(matches!(err, BackendError::InvalidJob { .. }));
}

#[test]
fn executes_only_declared_range_and_uses_declared_cross_shard_input() {
    let program = linear_program();
    let shard = Shard {
        id: 8,
        name: "tail".into(),
        range: 1..3,
        inputs: vec![Register::Virtual(0)],
        outputs: vec![Register::Virtual(2)],
    };
    let got = backend(program, shard, HashMap::new())
        .execute(&CpuWitnessInputs::new(
            HashMap::new(),
            BTreeMap::from([(
                0,
                raw(&[
                    625_000_000_000_000_000,
                    0,
                    750_000_000_000_000_000,
                    1_000_000_000_000_000_000,
                ]),
            )]),
        ))
        .unwrap();
    assert!(got.virtual_value(0).is_none());
    assert_eq!(
        got.virtual_value(2).unwrap()[0].raw(),
        375_000_000_000_000_000
    );
}

fn run_identity() -> RunIdentity {
    RunIdentity {
        model_graph_digest: Digest32::new([1; 32]),
        weights_digest: Digest32::new([2; 32]),
        compiler_digest: Digest32::new([3; 32]),
        isa_digest: Digest32::new([4; 32]),
        quantization_digest: Digest32::new([5; 32]),
        partition_plan_digest: Digest32::new([6; 32]),
        aggregation_plan_digest: Digest32::new([7; 32]),
        proof_flavor: ProofFlavorId::parse("halo2-kzg-bn256-shplonk-v1").unwrap(),
        model_visibility: ModelVisibility::PublicModel,
        aggregation_fan_in: 2,
        public_input_schema_version: 1,
    }
}

#[test]
fn witness_backend_rejects_wrong_shard_before_reading_and_writes_versioned_raw_json() {
    let program = linear_program();
    let cpu = backend(program.clone(), whole_shard(&program), HashMap::new());
    let digest = cpu.circuit_digest();
    let wrong = WitnessJob::new(
        run_identity(),
        ShardIdentity::new(99, "wrong".into(), digest).unwrap(),
        "/path/that/must/not/be/read".into(),
        digest,
    )
    .unwrap();
    assert!(matches!(
        cpu.estimate_resources(wrong),
        Err(BackendError::InvalidJob { .. })
    ));

    let nonce = format!(
        "{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    );
    let input_path = std::env::temp_dir().join(format!("zkie-witness-input-{nonce}.json"));
    let output_path = std::env::temp_dir().join(format!("zkie-witness-output-{nonce}.json"));
    std::fs::write(
        &input_path,
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "graph_inputs": {"x": [500_000_000_000_000_000_i64, -250_000_000_000_000_000_i64, 1_000_000_000_000_000_000_i64, 500_000_000_000_000_000_i64]},
            "virtual_inputs": {}
        }))
        .unwrap(),
    )
    .unwrap();
    let job = WitnessJob::new(
        run_identity(),
        cpu.shard_identity().clone(),
        input_path.clone(),
        digest,
    )
    .unwrap();
    let artifact = cpu.generate(job, output_path.clone()).unwrap();
    let wire: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&output_path).unwrap()).unwrap();
    assert_eq!(wire["schema_version"], 1);
    let virtual_two = wire["virtuals"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["index"] == 2)
        .unwrap();
    assert_eq!(virtual_two["raw_values"][0], 375_000_000_000_000_000_i64);
    assert_eq!(artifact.path(), &output_path);
    std::fs::remove_file(input_path).unwrap();
    std::fs::remove_file(output_path).unwrap();
}

#[test]
fn circuit_digest_is_derived_from_program_weight_shard_and_domain() {
    let program = linear_program();
    let base = backend(program.clone(), whole_shard(&program), HashMap::new());

    let mut changed_weight = program.clone();
    changed_weight
        .weights
        .insert("scale".into(), tensor(&[0.25]));
    let weight_backend = backend(
        changed_weight.clone(),
        whole_shard(&changed_weight),
        HashMap::new(),
    );
    assert_ne!(base.circuit_digest(), weight_backend.circuit_digest());

    let mut changed_shard = whole_shard(&program);
    changed_shard.name = "different".into();
    let shard_backend = backend(program.clone(), changed_shard, HashMap::new());
    assert_ne!(base.circuit_digest(), shard_backend.circuit_digest());

    let mut changed_instruction = program.clone();
    if let Instruction::DotGeneral { trans_b, .. } =
        &mut changed_instruction.instructions[0].instruction
    {
        *trans_b = true;
    }
    let instruction_backend = backend(
        changed_instruction.clone(),
        whole_shard(&changed_instruction),
        HashMap::new(),
    );
    assert_ne!(base.circuit_digest(), instruction_backend.circuit_digest());

    let shortened_shard = Shard {
        id: 7,
        name: "fixture".into(),
        range: 0..2,
        inputs: vec![],
        outputs: vec![Register::Virtual(1)],
    };
    let shortened = backend(program.clone(), shortened_shard, HashMap::new());
    assert_ne!(base.circuit_digest(), shortened.circuit_digest());

    let mut no_outputs = whole_shard(&program);
    no_outputs.outputs.clear();
    let boundary = backend(program.clone(), no_outputs, HashMap::new());
    assert_ne!(base.circuit_digest(), boundary.circuit_digest());

    let rms = rms_program(&[1.0, -1.0], &[1.0, 1.0]);
    let first = backend(
        rms.clone(),
        whole_shard(&rms),
        HashMap::from([(
            (2, 0),
            RsqrtDomain::RawAnchors(vec![1_000_000_000_000_000_000]),
        )]),
    );
    let second = backend(
        rms.clone(),
        whole_shard(&rms),
        HashMap::from([(
            (2, 0),
            RsqrtDomain::RawAnchors(vec![1_000_000_000_000_000_001]),
        )]),
    );
    assert_ne!(first.circuit_digest(), second.circuit_digest());
}

#[test]
fn rejects_unsupported_or_crossed_run_identity_before_reading() {
    let program = linear_program();
    let shard = whole_shard(&program);
    let mut private = run_identity();
    private.model_visibility = ModelVisibility::PrivateModel;
    assert!(matches!(
        ZkieIsaCpuWitnessBackend::new(
            Arc::new(program.clone()),
            shard.clone(),
            HashMap::new(),
            private,
            BoundaryShapeManifest::new(BTreeMap::new(), BTreeMap::new()).unwrap()
        ),
        Err(BackendError::InvalidJob { .. })
    ));
    let mut wrong_flavor = run_identity();
    wrong_flavor.proof_flavor = ProofFlavorId::parse("unsupported-proof").unwrap();
    assert!(matches!(
        ZkieIsaCpuWitnessBackend::new(
            Arc::new(program.clone()),
            shard,
            HashMap::new(),
            wrong_flavor,
            BoundaryShapeManifest::new(BTreeMap::new(), BTreeMap::new()).unwrap()
        ),
        Err(BackendError::InvalidJob { .. })
    ));

    let cpu = backend(program, whole_shard(&linear_program()), HashMap::new());
    let mut crossed = run_identity();
    crossed.compiler_digest = Digest32::new([99; 32]);
    let job = WitnessJob::new(
        crossed,
        cpu.shard_identity().clone(),
        "/path/that/must/not/be/read".into(),
        cpu.circuit_digest(),
    )
    .unwrap();
    assert!(matches!(
        cpu.estimate_resources(job),
        Err(BackendError::InvalidJob { .. })
    ));
}

#[test]
fn generate_load_round_trip_reconstructs_and_validates_typed_artifact() {
    let program = linear_program();
    let cpu = backend(program.clone(), whole_shard(&program), HashMap::new());
    let input_path = unique_temp("roundtrip-input");
    let output_path = unique_temp("roundtrip-output");
    write_linear_input(&input_path);
    let job = WitnessJob::new(
        run_identity(),
        cpu.shard_identity().clone(),
        input_path.clone(),
        cpu.circuit_digest(),
    )
    .unwrap();
    let generic = cpu.generate(job, output_path.clone()).unwrap();
    let typed = cpu.load_artifact(&generic).unwrap();
    assert_eq!(typed.circuit_digest(), cpu.circuit_digest());
    assert_eq!(typed.run_identity(), &run_identity());
    assert_eq!(typed.shard_identity(), cpu.shard_identity());
    assert_eq!(typed.virtuals().len(), 3);
    assert_eq!(
        typed.output_value(2).unwrap()[0].raw(),
        375_000_000_000_000_000
    );
    assert!(typed
        .input_value(&Register::GraphInput("x".into()))
        .is_some());
    std::fs::remove_file(input_path).unwrap();
    std::fs::remove_file(output_path).unwrap();
}

#[test]
fn decoder_rejects_schema_identity_duplicate_and_missing_register_tampering() {
    let program = linear_program();
    let cpu = backend(program.clone(), whole_shard(&program), HashMap::new());
    let input_path = unique_temp("tamper-input");
    let output_path = unique_temp("tamper-output");
    write_linear_input(&input_path);
    let job = WitnessJob::new(
        run_identity(),
        cpu.shard_identity().clone(),
        input_path.clone(),
        cpu.circuit_digest(),
    )
    .unwrap();
    let generic = cpu.generate(job, output_path.clone()).unwrap();
    let original: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&output_path).unwrap()).unwrap();

    let mut cases = Vec::new();
    let mut schema = original.clone();
    schema["schema_version"] = 999.into();
    cases.push(schema);
    let mut run = original.clone();
    run["run_identity_digest"] = serde_json::to_value(Digest32::new([77; 32])).unwrap();
    cases.push(run);
    let mut crossed_run = original.clone();
    let mut alternate = run_identity();
    alternate.compiler_digest = Digest32::new([78; 32]);
    crossed_run["run_identity"] = serde_json::to_value(&alternate).unwrap();
    crossed_run["run_identity_digest"] =
        serde_json::to_value(alternate.canonical_digest()).unwrap();
    cases.push(crossed_run);
    let mut duplicate = original.clone();
    let repeated = duplicate["inputs"][0].clone();
    duplicate["inputs"].as_array_mut().unwrap().push(repeated);
    cases.push(duplicate);
    let mut duplicate_virtual = original.clone();
    let repeated = duplicate_virtual["virtuals"][0].clone();
    duplicate_virtual["virtuals"]
        .as_array_mut()
        .unwrap()
        .push(repeated);
    cases.push(duplicate_virtual);
    let mut missing = original;
    missing["virtuals"]
        .as_array_mut()
        .unwrap()
        .retain(|entry| entry["index"] != 1);
    cases.push(missing);

    for (index, value) in cases.into_iter().enumerate() {
        let path = unique_temp(&format!("tampered-{index}"));
        let bytes = serde_json::to_vec(&value).unwrap();
        std::fs::write(&path, &bytes).unwrap();
        let artifact = WitnessArtifact::new(
            path.clone(),
            Digest32::new(*blake3::hash(&bytes).as_bytes()),
            cpu.shard_identity().clone(),
            cpu.circuit_digest(),
        )
        .unwrap();
        assert!(cpu.load_artifact(&artifact).is_err());
        std::fs::remove_file(path).unwrap();
    }
    std::fs::remove_file(input_path).unwrap();
    std::fs::remove_file(output_path).unwrap();
    assert_eq!(generic.shard(), cpu.shard_identity());
}

#[test]
fn rejects_extraneous_inputs_and_zero_dot_dimensions() {
    let program = linear_program();
    let cpu = backend(program.clone(), whole_shard(&program), HashMap::new());
    let err = cpu
        .execute(&CpuWitnessInputs::new(
            HashMap::from([
                ("x".into(), raw(&[0, 0, 0, 0])),
                ("unused".into(), raw(&[0])),
            ]),
            BTreeMap::from([(88, raw(&[0]))]),
        ))
        .unwrap_err();
    assert!(matches!(err, BackendError::InvalidJob { .. }));

    for (m, n, k) in [(0, 1, 1), (1, 0, 1), (1, 1, 0)] {
        let mut zero = linear_program();
        zero.instructions.truncate(1);
        zero.instructions[0].instruction = Instruction::DotGeneral {
            m,
            n,
            k,
            batch_dims: vec![],
            trans_a: false,
            trans_b: false,
        };
        let shard = whole_shard(&zero);
        assert!(matches!(
            ZkieIsaCpuWitnessBackend::new(
                Arc::new(zero),
                shard,
                HashMap::new(),
                run_identity(),
                BoundaryShapeManifest::new(BTreeMap::new(), BTreeMap::new()).unwrap(),
            ),
            Err(BackendError::ShapeMismatch { .. })
        ));
    }
}

#[test]
fn enforces_domain_input_artifact_and_resource_limits() {
    let rms = rms_program(&[1.0, -1.0], &[1.0, 1.0]);
    assert!(matches!(
        ZkieIsaCpuWitnessBackend::new(
            Arc::new(rms.clone()),
            whole_shard(&rms),
            HashMap::from([(
                (2, 0),
                RsqrtDomain::Range {
                    min: 0.1,
                    max: 1.0,
                    n: MAX_CPU_WITNESS_DOMAIN_POINTS + 1,
                }
            )]),
            run_identity(),
            BoundaryShapeManifest::new(BTreeMap::new(), BTreeMap::new()).unwrap()
        ),
        Err(BackendError::Resource { .. })
    ));
    assert!(matches!(
        ZkieIsaCpuWitnessBackend::new(
            Arc::new(rms.clone()),
            whole_shard(&rms),
            HashMap::from([(
                (2, 0),
                RsqrtDomain::Range {
                    min: 0.1,
                    max: 0.100_000_000_000_000_1,
                    n: 10_000,
                }
            )]),
            run_identity(),
            BoundaryShapeManifest::new(BTreeMap::new(), BTreeMap::new()).unwrap()
        ),
        Err(BackendError::InvalidJob { .. })
    ));

    let program = linear_program();
    let cpu = backend(program.clone(), whole_shard(&program), HashMap::new());
    let request = cpu
        .estimate_resources(
            WitnessJob::new(
                run_identity(),
                cpu.shard_identity().clone(),
                "unused".into(),
                cpu.circuit_digest(),
            )
            .unwrap(),
        )
        .unwrap();
    assert!(
        request.ram_bytes >= (MAX_CPU_WITNESS_INPUT_BYTES + MAX_CPU_WITNESS_ARTIFACT_BYTES) as u64
    );

    let input_path = unique_temp("oversized-input");
    std::fs::write(&input_path, vec![b' '; MAX_CPU_WITNESS_INPUT_BYTES + 1]).unwrap();
    let output_path = unique_temp("oversized-output");
    let job = WitnessJob::new(
        run_identity(),
        cpu.shard_identity().clone(),
        input_path.clone(),
        cpu.circuit_digest(),
    )
    .unwrap();
    assert!(matches!(
        cpu.generate(job, output_path),
        Err(BackendError::Resource { .. })
    ));
    std::fs::remove_file(input_path).unwrap();

    let huge = CpuWitnessInputs::new(
        HashMap::from([(
            "x".into(),
            raw(&vec![0; MAX_CPU_WITNESS_VALUES_PER_TENSOR + 1]),
        )]),
        BTreeMap::new(),
    );
    assert!(matches!(
        cpu.execute(&huge),
        Err(BackendError::Resource { .. }) | Err(BackendError::InvalidJob { .. })
    ));

    let mut oversized_dot = linear_program();
    oversized_dot.instructions.truncate(1);
    oversized_dot.instructions[0].instruction = Instruction::DotGeneral {
        m: 257,
        n: 257,
        k: 1,
        batch_dims: vec![],
        trans_a: false,
        trans_b: false,
    };
    let shard = whole_shard(&oversized_dot);
    assert!(matches!(
        ZkieIsaCpuWitnessBackend::new(
            Arc::new(oversized_dot),
            shard,
            HashMap::new(),
            run_identity(),
            BoundaryShapeManifest::new(BTreeMap::new(), BTreeMap::new()).unwrap()
        ),
        Err(BackendError::Resource { .. })
    ));

    let tight = zkie_prover::CpuWitnessLimits {
        artifact_bytes: 64,
        ..Default::default()
    };
    let limited = ZkieIsaCpuWitnessBackend::with_limits(
        Arc::new(program.clone()),
        whole_shard(&program),
        HashMap::new(),
        run_identity(),
        tight,
        test_boundary_shapes(&program, &whole_shard(&program)),
    )
    .unwrap();
    let limited_input = unique_temp("limited-input");
    let limited_output = unique_temp("limited-output");
    write_linear_input(&limited_input);
    let job = WitnessJob::new(
        run_identity(),
        limited.shard_identity().clone(),
        limited_input.clone(),
        limited.circuit_digest(),
    )
    .unwrap();
    assert!(matches!(
        limited.generate(job, limited_output.clone()),
        Err(BackendError::Resource { .. })
    ));
    assert!(!limited_output.exists());
    let retry = backend(program.clone(), whole_shard(&program), HashMap::new());
    let retry_job = WitnessJob::new(
        run_identity(),
        retry.shard_identity().clone(),
        limited_input.clone(),
        retry.circuit_digest(),
    )
    .unwrap();
    let written = retry.generate(retry_job, limited_output.clone()).unwrap();
    retry.load_artifact(&written).unwrap();
    std::fs::remove_file(limited_output).unwrap();
    std::fs::remove_file(limited_input).unwrap();
}

#[test]
fn artifact_record_limit_is_identical_for_construction_generation_and_loading() {
    let program = linear_program();
    // Wire records are 4 boundary inputs + 3 virtuals + 1 declared output.
    let too_tight = zkie_prover::CpuWitnessLimits {
        tensor_count: 7,
        ..Default::default()
    };
    assert!(matches!(
        ZkieIsaCpuWitnessBackend::with_limits(
            Arc::new(program.clone()),
            whole_shard(&program),
            HashMap::new(),
            run_identity(),
            too_tight,
            test_boundary_shapes(&program, &whole_shard(&program)),
        ),
        Err(BackendError::Resource { .. })
    ));

    let exact = zkie_prover::CpuWitnessLimits {
        tensor_count: 8,
        ..Default::default()
    };
    let cpu = ZkieIsaCpuWitnessBackend::with_limits(
        Arc::new(program.clone()),
        whole_shard(&program),
        HashMap::new(),
        run_identity(),
        exact,
        test_boundary_shapes(&program, &whole_shard(&program)),
    )
    .unwrap();
    let input = unique_temp("record-boundary-input");
    let output = unique_temp("record-boundary-output");
    write_linear_input(&input);
    let job = WitnessJob::new(
        run_identity(),
        cpu.shard_identity().clone(),
        input.clone(),
        cpu.circuit_digest(),
    )
    .unwrap();
    let artifact = cpu.generate(job, output.clone()).unwrap();
    cpu.load_artifact(&artifact).unwrap();
    std::fs::remove_file(input).unwrap();
    std::fs::remove_file(output).unwrap();

    let long = scalar_chain(1_025);
    assert!(matches!(
        ZkieIsaCpuWitnessBackend::new(
            Arc::new(long.clone()),
            whole_shard(&long),
            HashMap::new(),
            run_identity(),
            BoundaryShapeManifest::new(BTreeMap::new(), BTreeMap::new()).unwrap(),
        ),
        Err(BackendError::Resource { .. })
    ));

    let operand_tight = zkie_prover::CpuWitnessLimits {
        operand_references: 5,
        ..Default::default()
    };
    assert!(matches!(
        ZkieIsaCpuWitnessBackend::with_limits(
            Arc::new(program.clone()),
            whole_shard(&program),
            HashMap::new(),
            run_identity(),
            operand_tight,
            test_boundary_shapes(&program, &whole_shard(&program)),
        ),
        Err(BackendError::Resource { .. })
    ));
}

fn scalar_chain(instruction_count: usize) -> CompiledProgram {
    let instructions = (0..instruction_count)
        .map(|index| CompiledInstruction {
            instruction: Instruction::Eltwise { op: EltwiseOp::Add },
            inputs: vec![
                if index == 0 {
                    Register::GraphInput("x".into())
                } else {
                    Register::Virtual(index - 1)
                },
                Register::Weight("zero".into()),
            ],
            output_name: format!("v{index}"),
        })
        .collect();
    CompiledProgram {
        instructions,
        weights: HashMap::from([("zero".into(), tensor(&[0.0]))]),
        graph_inputs: vec!["x".into()],
        graph_outputs: vec![(
            "y".into(),
            Register::Virtual(instruction_count.saturating_sub(1)),
        )],
    }
}

#[test]
fn generate_never_overwrites_existing_file_or_follows_output_symlink() {
    let program = linear_program();
    let cpu = backend(program.clone(), whole_shard(&program), HashMap::new());
    let input_path = unique_temp("exclusive-input");
    write_linear_input(&input_path);
    let existing = unique_temp("existing-output");
    std::fs::write(&existing, b"keep").unwrap();
    let make_job = || {
        WitnessJob::new(
            run_identity(),
            cpu.shard_identity().clone(),
            input_path.clone(),
            cpu.circuit_digest(),
        )
        .unwrap()
    };
    assert!(matches!(
        cpu.generate(make_job(), existing.clone()),
        Err(BackendError::Io { .. })
    ));
    assert_eq!(std::fs::read(&existing).unwrap(), b"keep");
    let temp_prefix = format!(
        ".{}.zkie-tmp-",
        existing.file_name().unwrap().to_string_lossy()
    );
    assert!(!std::fs::read_dir(existing.parent().unwrap())
        .unwrap()
        .filter_map(Result::ok)
        .any(|entry| entry
            .file_name()
            .to_string_lossy()
            .starts_with(&temp_prefix)));
    std::fs::remove_file(&existing).unwrap();
    let retried = cpu.generate(make_job(), existing.clone()).unwrap();
    cpu.load_artifact(&retried).unwrap();
    std::fs::remove_file(&existing).unwrap();

    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let target = unique_temp("symlink-target");
        let link = unique_temp("symlink-output");
        std::fs::write(&target, b"keep-target").unwrap();
        symlink(&target, &link).unwrap();
        assert!(matches!(
            cpu.generate(make_job(), link.clone()),
            Err(BackendError::Io { .. })
        ));
        assert_eq!(std::fs::read(&target).unwrap(), b"keep-target");
        std::fs::remove_file(link).unwrap();
        std::fs::remove_file(target).unwrap();
    }
    std::fs::remove_file(input_path).unwrap();
}

#[cfg(unix)]
#[test]
fn generated_artifact_is_private_to_its_owner() {
    use std::os::unix::fs::PermissionsExt;

    let program = linear_program();
    let cpu = backend(program.clone(), whole_shard(&program), HashMap::new());
    let input = unique_temp("private-mode-input");
    let output = unique_temp("private-mode-output");
    write_linear_input(&input);
    let job = WitnessJob::new(
        run_identity(),
        cpu.shard_identity().clone(),
        input.clone(),
        cpu.circuit_digest(),
    )
    .unwrap();

    cpu.generate(job, output.clone()).unwrap();

    let mode = std::fs::metadata(&output).unwrap().permissions().mode();
    assert_eq!(mode & 0o077, 0);
    std::fs::remove_file(input).unwrap();
    std::fs::remove_file(output).unwrap();
}

#[cfg(unix)]
#[test]
fn bounded_input_reader_rejects_fifo_without_blocking() {
    let program = linear_program();
    let cpu = backend(program.clone(), whole_shard(&program), HashMap::new());
    let fifo = unique_temp("input-fifo");
    let status = std::process::Command::new("mkfifo")
        .arg(&fifo)
        .status()
        .unwrap();
    assert!(status.success());
    let output = unique_temp("fifo-output");
    let job = WitnessJob::new(
        run_identity(),
        cpu.shard_identity().clone(),
        fifo.clone(),
        cpu.circuit_digest(),
    )
    .unwrap();
    assert!(matches!(
        cpu.generate(job, output.clone()),
        Err(BackendError::Io { .. })
    ));
    assert!(!output.exists());
    std::fs::remove_file(fifo).unwrap();
}

fn unique_temp(label: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "zkie-{label}-{}-{}",
        std::process::id(),
        std::thread::current().name().unwrap_or("test")
    ))
}

fn write_linear_input(path: &std::path::Path) {
    std::fs::write(
        path,
        serde_json::to_vec(&serde_json::json!({
            "schema_version": 1,
            "graph_inputs": {"x": [500_000_000_000_000_000_i64, -250_000_000_000_000_000_i64, 1_000_000_000_000_000_000_i64, 500_000_000_000_000_000_i64]},
            "virtual_inputs": {}
        }))
        .unwrap(),
    )
    .unwrap();
}
