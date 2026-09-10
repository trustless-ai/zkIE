use std::collections::HashMap;

use zkie_compiler::circuit_binding::to_assembler_program;
use zkie_compiler::dag::{
    InstructionCostModel, InstructionEstimate, PartitionError, PartitionPlanner, PartitionRequest,
    ShardEstimate,
};
use zkie_compiler::graph_compiler::{CompiledInstruction, CompiledProgram, Register};
use zkie_compiler::shard_binding::{
    bind_shard_program, BoundaryLayout, BoundaryTensorSpec, ShardBindingError,
};
use zkie_core::assembler::{AssemblerProgram, RegisterRef};
use zkie_core::fixed_point::{requantize_mul, I18};
use zkie_core::isa::{EltwiseOp, Instruction};
use zkie_types::Digest32;

#[derive(Clone)]
struct UnitCost;

impl InstructionCostModel for UnitCost {
    fn identity(&self) -> &str {
        "unit"
    }
    fn version(&self) -> u32 {
        1
    }
    fn estimate_instruction(
        &self,
        _: &CompiledInstruction,
    ) -> Result<InstructionEstimate, PartitionError> {
        InstructionEstimate::new(1, 1)
    }
    fn estimate_shard(
        &self,
        instructions: &[CompiledInstruction],
    ) -> Result<ShardEstimate, PartitionError> {
        ShardEstimate::new(instructions.len() as u64, 1)
    }
}

fn model_digest() -> Digest32 {
    Digest32::new([9; 32])
}

fn program() -> CompiledProgram {
    CompiledProgram {
        instructions: vec![
            CompiledInstruction {
                instruction: Instruction::Eltwise { op: EltwiseOp::Add },
                inputs: vec![
                    Register::GraphInput("x".into()),
                    Register::Weight("one".into()),
                ],
                output_name: "broadcast".into(),
            },
            CompiledInstruction {
                instruction: Instruction::Eltwise { op: EltwiseOp::Mul },
                inputs: vec![Register::Virtual(0), Register::Weight("two".into())],
                output_name: "middle".into(),
            },
            CompiledInstruction {
                instruction: Instruction::Eltwise { op: EltwiseOp::Add },
                inputs: vec![Register::Virtual(1), Register::Virtual(0)],
                output_name: "y".into(),
            },
        ],
        weights: HashMap::from([
            (
                "one".into(),
                zkie_compiler::onnx_parser::WeightTensor {
                    shape: vec![2],
                    data: vec![1.0, 1.0],
                },
            ),
            (
                "two".into(),
                zkie_compiler::onnx_parser::WeightTensor {
                    shape: vec![1],
                    data: vec![2.0],
                },
            ),
        ]),
        graph_inputs: vec!["x".into()],
        graph_outputs: vec![("y".into(), Register::Virtual(2))],
    }
}

fn make_plan(program: &CompiledProgram) -> zkie_compiler::dag::PartitionPlan {
    let request = PartitionRequest::new(1, 20, model_digest(), "compiler-v1", UnitCost)
        .unwrap()
        .with_boundary_layout_digest(layout().digest());
    PartitionPlanner::plan(program, &request).unwrap()
}

fn layout() -> BoundaryLayout {
    BoundaryLayout::new(vec![
        BoundaryTensorSpec::flat(Register::GraphInput("x".into()), 2).unwrap(),
        BoundaryTensorSpec::flat(Register::Virtual(0), 2).unwrap(),
        BoundaryTensorSpec::flat(Register::Virtual(1), 2).unwrap(),
        BoundaryTensorSpec::flat(Register::Virtual(2), 2).unwrap(),
    ])
    .unwrap()
}

fn raw(values: &[i64]) -> Vec<I18> {
    values.iter().copied().map(I18::from_raw).collect()
}

#[test]
fn sequential_and_broadcast_inputs_remap_to_stable_local_indices() {
    let program = program();
    let plan = make_plan(&program);
    let graph = HashMap::from([("x".into(), raw(&[3, 4]))]);
    let shard0 = bind_shard_program(
        &program,
        &plan,
        0,
        model_digest(),
        &layout(),
        &graph,
        &HashMap::new(),
    )
    .unwrap();
    assert_eq!(
        shard0.assembler_program().instructions[0].inputs,
        vec![RegisterRef::Input(0), RegisterRef::Weight(0)]
    );
    assert_eq!(
        shard0.output_boundaries()[0].register(),
        &Register::Virtual(0)
    );
    assert_eq!(shard0.output_boundaries()[0].edge_ids().len(), 2);

    let incoming = HashMap::from([(Register::Virtual(0), raw(&[4, 5]))]);
    let shard1 = bind_shard_program(
        &program,
        &plan,
        1,
        model_digest(),
        &layout(),
        &graph,
        &incoming,
    )
    .unwrap();
    assert_eq!(
        shard1.assembler_program().instructions[0].inputs,
        vec![RegisterRef::Input(0), RegisterRef::Weight(0)]
    );
    assert_eq!(shard1.global_to_local().get(&1), Some(&0));

    let reversed = HashMap::from([
        (Register::Virtual(1), raw(&[8, 10])),
        (Register::Virtual(0), raw(&[4, 5])),
    ]);
    let shard2 = bind_shard_program(
        &program,
        &plan,
        2,
        model_digest(),
        &layout(),
        &graph,
        &reversed,
    )
    .unwrap();
    assert_eq!(
        shard2.assembler_program().instructions[0].inputs,
        vec![RegisterRef::Input(1), RegisterRef::Input(0)]
    );
    assert_eq!(
        shard2.input_boundaries()[0].register(),
        &Register::Virtual(0)
    );
    assert_eq!(
        shard2.input_boundaries()[1].register(),
        &Register::Virtual(1)
    );
    assert_eq!(
        shard2.output_boundaries()[0].register(),
        &Register::Virtual(2)
    );
}

#[test]
fn rejects_missing_extra_shape_and_plan_mismatches() {
    let program = program();
    let plan = make_plan(&program);
    let graph = HashMap::from([("x".into(), raw(&[3, 4]))]);
    assert!(matches!(
        bind_shard_program(
            &program,
            &plan,
            1,
            model_digest(),
            &layout(),
            &graph,
            &HashMap::new()
        ),
        Err(ShardBindingError::MissingBoundary { .. })
    ));
    let extra = HashMap::from([
        (Register::Virtual(0), raw(&[4, 5])),
        (Register::Virtual(2), raw(&[1, 2])),
    ]);
    assert!(matches!(
        bind_shard_program(
            &program,
            &plan,
            1,
            model_digest(),
            &layout(),
            &graph,
            &extra
        ),
        Err(ShardBindingError::ExtraBoundary { .. })
    ));
    assert!(matches!(
        bind_shard_program(
            &program,
            &plan,
            0,
            Digest32::new([8; 32]),
            &layout(),
            &graph,
            &HashMap::new()
        ),
        Err(ShardBindingError::PlanProgramMismatch)
    ));

    assert!(matches!(
        bind_shard_program(
            &program,
            &plan,
            0,
            model_digest(),
            &layout(),
            &HashMap::new(),
            &HashMap::new()
        ),
        Err(ShardBindingError::MissingGraphInput { .. })
    ));
    let wrong_len = HashMap::from([("x".into(), raw(&[3]))]);
    assert!(matches!(
        bind_shard_program(
            &program,
            &plan,
            0,
            model_digest(),
            &layout(),
            &wrong_len,
            &HashMap::new()
        ),
        Err(ShardBindingError::BoundaryLengthMismatch { .. })
    ));

    let mut forward = program.clone();
    forward.instructions[0].inputs[0] = Register::Virtual(1);
    assert!(matches!(
        bind_shard_program(
            &forward,
            &plan,
            0,
            model_digest(),
            &layout(),
            &graph,
            &HashMap::new()
        ),
        Err(ShardBindingError::LaterOrOutsideReference { .. })
    ));
    let mut different_program = program.clone();
    different_program.instructions[0].instruction = Instruction::Eltwise { op: EltwiseOp::Mul };
    assert!(matches!(
        bind_shard_program(
            &different_program,
            &plan,
            0,
            model_digest(),
            &layout(),
            &graph,
            &HashMap::new(),
        ),
        Err(ShardBindingError::PlanProgramMismatch)
    ));
    let mut bad_output = program.clone();
    bad_output.graph_outputs = vec![("bad".into(), Register::Virtual(9))];
    let bad_output_plan = make_plan(&bad_output);
    assert!(matches!(
        bind_shard_program(
            &bad_output,
            &bad_output_plan,
            0,
            model_digest(),
            &layout(),
            &graph,
            &HashMap::new(),
        ),
        Err(ShardBindingError::OutputOutsideProgram { .. })
    ));
    let duplicate = BoundaryLayout::new(vec![
        BoundaryTensorSpec::flat(Register::Virtual(0), 2).unwrap(),
        BoundaryTensorSpec::flat(Register::Virtual(0), 2).unwrap(),
    ]);
    assert!(matches!(
        duplicate,
        Err(ShardBindingError::DuplicateBoundary)
    ));
    let substituted_layout = BoundaryLayout::new(vec![
        BoundaryTensorSpec::flat(Register::GraphInput("x".into()), 2).unwrap(),
        BoundaryTensorSpec::flat(Register::Virtual(0), 3).unwrap(),
        BoundaryTensorSpec::flat(Register::Virtual(1), 2).unwrap(),
        BoundaryTensorSpec::flat(Register::Virtual(2), 2).unwrap(),
    ])
    .unwrap();
    let substituted_incoming = HashMap::from([(Register::Virtual(0), raw(&[4, 5, 6]))]);
    assert!(matches!(
        bind_shard_program(
            &program,
            &plan,
            1,
            model_digest(),
            &substituted_layout,
            &graph,
            &substituted_incoming,
        ),
        Err(ShardBindingError::BoundaryLayoutMismatch)
    ));
}

fn execute(program: &AssemblerProgram) -> Vec<Vec<I18>> {
    let mut virtuals = Vec::<Vec<I18>>::new();
    for instruction in &program.instructions {
        let operands = instruction
            .inputs
            .iter()
            .map(|reference| match reference {
                RegisterRef::Input(index) => &program.input_values[*index],
                RegisterRef::Weight(index) => &program.weight_values[*index],
                RegisterRef::Virtual(index) => &virtuals[*index],
            })
            .collect::<Vec<_>>();
        let len = operands.iter().map(|values| values.len()).max().unwrap();
        let values = (0..len)
            .map(|index| {
                let left = operands[0][index % operands[0].len()];
                let right = operands[1][index % operands[1].len()];
                match instruction.instruction {
                    Instruction::Eltwise { op: EltwiseOp::Add } => {
                        I18::from_raw(left.raw().checked_add(right.raw()).unwrap())
                    }
                    Instruction::Eltwise { op: EltwiseOp::Mul } => {
                        requantize_mul(left, right).unwrap().0
                    }
                    _ => panic!("fixture uses only add/mul"),
                }
            })
            .collect();
        virtuals.push(values);
    }
    virtuals
}

#[test]
fn sharded_execution_matches_the_unsplit_program_exactly() {
    let program = program();
    let plan = make_plan(&program);
    let graph = HashMap::from([("x".into(), raw(&[3, 4]))]);
    let unsplit = execute(&to_assembler_program(&program, &graph).unwrap());
    let mut incoming = HashMap::new();
    let shard0 = bind_shard_program(
        &program,
        &plan,
        0,
        model_digest(),
        &layout(),
        &graph,
        &incoming,
    )
    .unwrap();
    let out0 = execute(shard0.assembler_program()).pop().unwrap();
    incoming.insert(Register::Virtual(0), out0);
    let shard1 = bind_shard_program(
        &program,
        &plan,
        1,
        model_digest(),
        &layout(),
        &graph,
        &incoming,
    )
    .unwrap();
    let out1 = execute(shard1.assembler_program()).pop().unwrap();
    incoming.insert(Register::Virtual(1), out1);
    let shard2 = bind_shard_program(
        &program,
        &plan,
        2,
        model_digest(),
        &layout(),
        &graph,
        &incoming,
    )
    .unwrap();
    let final_sharded = execute(shard2.assembler_program()).pop().unwrap();
    assert_eq!(final_sharded, unsplit[2]);
    assert_eq!(
        shard2.output_boundaries()[0].graph_output_names(),
        &["y".to_string()]
    );
}
