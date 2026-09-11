use std::cell::Cell;
use std::collections::HashMap;

use zkie_compiler::dag::{
    BoundaryHint, InstructionCostModel, InstructionEstimate, PartitionError, PartitionPlanner,
    PartitionRequest, ShardEstimate,
};
use zkie_compiler::graph_compiler::{CompiledInstruction, CompiledProgram, Register};
use zkie_core::isa::{EltwiseOp, Instruction};
use zkie_types::Digest32;

#[derive(Clone)]
struct FixedCost {
    bytes: u64,
    k: u32,
    identity: &'static str,
    version: u32,
}

impl InstructionCostModel for FixedCost {
    fn identity(&self) -> &str {
        self.identity
    }

    fn version(&self) -> u32 {
        self.version
    }

    fn estimate_instruction(
        &self,
        _instruction: &CompiledInstruction,
    ) -> Result<InstructionEstimate, PartitionError> {
        InstructionEstimate::new(self.bytes, self.k)
    }

    fn estimate_shard(
        &self,
        instructions: &[CompiledInstruction],
    ) -> Result<ShardEstimate, PartitionError> {
        let count = u64::try_from(instructions.len()).unwrap();
        ShardEstimate::new(
            self.bytes
                .checked_mul(count)
                .ok_or(PartitionError::EstimateOverflow)?,
            self.k,
        )
    }
}

fn instruction(inputs: Vec<Register>, name: &str) -> CompiledInstruction {
    CompiledInstruction {
        instruction: Instruction::Eltwise { op: EltwiseOp::Add },
        inputs,
        output_name: name.into(),
    }
}

fn program(instructions: Vec<CompiledInstruction>) -> CompiledProgram {
    CompiledProgram {
        instructions,
        weights: HashMap::new(),
        graph_inputs: vec!["x".into()],
        graph_outputs: vec![],
    }
}

fn request(bytes: u64, cost: FixedCost) -> PartitionRequest<FixedCost> {
    PartitionRequest::new(bytes, 20, Digest32::new([7; 32]), "compiler-v1", cost).unwrap()
}

#[test]
fn planner_cuts_before_exceeding_target_and_is_deterministic() {
    let program = program(vec![
        instruction(vec![Register::GraphInput("x".into())], "a"),
        instruction(vec![Register::Virtual(0)], "b"),
        instruction(vec![Register::Virtual(1)], "c"),
        instruction(vec![Register::Virtual(2)], "d"),
    ]);
    let request = request(
        300,
        FixedCost {
            bytes: 100,
            k: 10,
            identity: "fixed-100",
            version: 1,
        },
    );
    let first = PartitionPlanner::plan(&program, &request).unwrap();
    let second = PartitionPlanner::plan(&program, &request).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        first
            .shards()
            .iter()
            .map(|shard| shard.range())
            .collect::<Vec<_>>(),
        vec![0..3, 3..4]
    );
    assert_eq!(first.digest(), second.digest());
}

#[test]
fn hint_priority_then_latest_boundary_is_deterministic() {
    let program = program(
        (0..5)
            .map(|i| instruction(vec![], &format!("v{i}")))
            .collect(),
    );
    let request = request(
        350,
        FixedCost {
            bytes: 100,
            k: 10,
            identity: "fixed",
            version: 1,
        },
    )
    .with_boundary_hints(vec![
        BoundaryHint::new(1, 9, "early").unwrap(),
        BoundaryHint::new(2, 9, "later").unwrap(),
        BoundaryHint::new(3, 8, "lower-priority").unwrap(),
    ])
    .unwrap();
    let plan = PartitionPlanner::plan(&program, &request).unwrap();
    assert_eq!(plan.shards()[0].range(), 0..2);
}

#[test]
fn rejects_unsplittable_instruction_zero_budget_and_forward_virtual() {
    let one = program(vec![instruction(vec![], "too-large")]);
    let too_small = request(
        99,
        FixedCost {
            bytes: 100,
            k: 10,
            identity: "fixed",
            version: 1,
        },
    );
    assert!(matches!(
        PartitionPlanner::plan(&one, &too_small),
        Err(PartitionError::UnsplittableInstruction { instruction: 0, .. })
    ));
    assert!(matches!(
        PartitionRequest::new(
            0,
            20,
            Digest32::new([1; 32]),
            "compiler",
            too_small.cost_model().clone()
        ),
        Err(PartitionError::ZeroBudget)
    ));
    assert_eq!(
        PartitionPlanner::plan(&program(vec![]), &too_small),
        Err(PartitionError::EmptyProgram)
    );
    let forward = program(vec![
        instruction(vec![Register::Virtual(1)], "bad"),
        instruction(vec![], "later"),
    ]);
    assert!(matches!(
        PartitionPlanner::plan(&forward, &request(100, too_small.cost_model().clone())),
        Err(PartitionError::InvalidVirtualReference { .. })
    ));
}

#[test]
fn broadcast_dependency_has_one_stable_edge_per_consumer_and_full_coverage() {
    let program = program(vec![
        instruction(vec![], "broadcast"),
        instruction(vec![Register::Virtual(0)], "left"),
        instruction(vec![Register::Virtual(0)], "right"),
    ]);
    let plan = PartitionPlanner::plan(
        &program,
        &request(
            100,
            FixedCost {
                bytes: 100,
                k: 10,
                identity: "fixed",
                version: 1,
            },
        ),
    )
    .unwrap();
    assert_eq!(
        plan.shards().iter().map(|s| s.range().len()).sum::<usize>(),
        3
    );
    assert_eq!(plan.dag().edges.len(), 2);
    assert_ne!(plan.dag().edges[0].id, plan.dag().edges[1].id);
    assert!(plan.dag().edges.iter().all(|edge| edge.producer == 0));
}

#[test]
fn estimate_overflow_and_invalid_hints_are_typed_errors() {
    #[derive(Clone)]
    struct OverflowCost;
    impl InstructionCostModel for OverflowCost {
        fn identity(&self) -> &str {
            "overflow"
        }
        fn version(&self) -> u32 {
            1
        }
        fn estimate_instruction(
            &self,
            _: &CompiledInstruction,
        ) -> Result<InstructionEstimate, PartitionError> {
            Err(PartitionError::EstimateOverflow)
        }
        fn estimate_shard(
            &self,
            _: &[CompiledInstruction],
        ) -> Result<ShardEstimate, PartitionError> {
            Err(PartitionError::EstimateOverflow)
        }
    }
    let one = program(vec![instruction(vec![], "one")]);
    let overflow =
        PartitionRequest::new(100, 20, Digest32::new([1; 32]), "compiler", OverflowCost).unwrap();
    assert_eq!(
        PartitionPlanner::plan(&one, &overflow),
        Err(PartitionError::EstimateOverflow)
    );

    let huge = FixedCost {
        bytes: u64::MAX,
        k: 1,
        identity: "huge",
        version: 1,
    };
    let two = program(vec![instruction(vec![], "a"), instruction(vec![], "b")]);
    let huge_request =
        PartitionRequest::new(u64::MAX, 20, Digest32::new([1; 32]), "compiler", huge).unwrap();
    assert_eq!(
        PartitionPlanner::plan(&two, &huge_request),
        Err(PartitionError::EstimateOverflow)
    );

    let duplicate = request(
        100,
        FixedCost {
            bytes: 1,
            k: 1,
            identity: "fixed",
            version: 1,
        },
    )
    .with_boundary_hints(vec![
        BoundaryHint::new(1, 1, "a").unwrap(),
        BoundaryHint::new(1, 2, "b").unwrap(),
    ]);
    assert!(matches!(duplicate, Err(PartitionError::InvalidHint { .. })));
}

#[test]
fn digest_binds_ranges_edges_estimates_model_compiler_and_cost_model_identity() {
    let chain = program(vec![
        instruction(vec![Register::GraphInput("x".into())], "a"),
        instruction(vec![Register::Virtual(0)], "b"),
        instruction(vec![Register::Virtual(1)], "c"),
    ]);
    let fixed = FixedCost {
        bytes: 100,
        k: 10,
        identity: "fixed",
        version: 1,
    };
    let base_request = |budget, model, compiler: &str, cost: FixedCost| {
        PartitionRequest::new(budget, 20, Digest32::new([model; 32]), compiler, cost).unwrap()
    };
    let base = PartitionPlanner::plan(
        &chain,
        &base_request(1_000, 7, "compiler-v1", fixed.clone()),
    )
    .unwrap();
    let range = PartitionPlanner::plan(&chain, &base_request(200, 7, "compiler-v1", fixed.clone()))
        .unwrap();
    let estimate = PartitionPlanner::plan(
        &chain,
        &base_request(
            1_000,
            7,
            "compiler-v1",
            FixedCost {
                bytes: 101,
                ..fixed.clone()
            },
        ),
    )
    .unwrap();
    let model = PartitionPlanner::plan(
        &chain,
        &base_request(1_000, 8, "compiler-v1", fixed.clone()),
    )
    .unwrap();
    let compiler = PartitionPlanner::plan(
        &chain,
        &base_request(1_000, 7, "compiler-v2", fixed.clone()),
    )
    .unwrap();
    let identity = PartitionPlanner::plan(
        &chain,
        &base_request(
            1_000,
            7,
            "compiler-v1",
            FixedCost {
                identity: "other",
                ..fixed.clone()
            },
        ),
    )
    .unwrap();
    let version = PartitionPlanner::plan(
        &chain,
        &base_request(
            1_000,
            7,
            "compiler-v1",
            FixedCost {
                version: 2,
                ..fixed.clone()
            },
        ),
    )
    .unwrap();
    let no_edge_program = program(vec![
        instruction(vec![Register::GraphInput("x".into())], "a"),
        instruction(vec![Register::GraphInput("x".into())], "b"),
        instruction(vec![Register::GraphInput("x".into())], "c"),
    ]);
    let chain_edges =
        PartitionPlanner::plan(&chain, &base_request(100, 7, "compiler-v1", fixed.clone()))
            .unwrap();
    let edge = PartitionPlanner::plan(
        &no_edge_program,
        &base_request(100, 7, "compiler-v1", fixed),
    )
    .unwrap();
    assert_ne!(chain_edges.digest(), edge.digest());
    for changed in [range, estimate, model, compiler, identity, version] {
        assert_ne!(base.digest(), changed.digest());
    }
}

#[test]
fn accepted_shard_estimate_is_not_recomputed() {
    struct StatefulCost(Cell<u32>);
    impl InstructionCostModel for StatefulCost {
        fn identity(&self) -> &str {
            "stateful-test"
        }
        fn version(&self) -> u32 {
            1
        }
        fn estimate_instruction(
            &self,
            _: &CompiledInstruction,
        ) -> Result<InstructionEstimate, PartitionError> {
            InstructionEstimate::new(10, 1)
        }
        fn estimate_shard(
            &self,
            _: &[CompiledInstruction],
        ) -> Result<ShardEstimate, PartitionError> {
            let call = self.0.get();
            self.0.set(call + 1);
            ShardEstimate::new(if call == 0 { 10 } else { 1_000 }, 1)
        }
    }
    let request = PartitionRequest::new(
        100,
        20,
        Digest32::new([1; 32]),
        "compiler",
        StatefulCost(Cell::new(0)),
    )
    .unwrap();
    let plan =
        PartitionPlanner::plan(&program(vec![instruction(vec![], "one")]), &request).unwrap();
    assert_eq!(request.cost_model().0.get(), 1);
    assert_eq!(plan.shards()[0].estimate().ram_bytes(), 10);
}

#[test]
fn stable_edge_ids_are_canonically_sorted_beyond_single_digit_shards() {
    let instructions = (0..12)
        .map(|index| {
            instruction(
                if index == 0 {
                    vec![]
                } else {
                    vec![Register::Virtual(0)]
                },
                &format!("v{index}"),
            )
        })
        .collect();
    let plan = PartitionPlanner::plan(
        &program(instructions),
        &request(
            1,
            FixedCost {
                bytes: 1,
                k: 1,
                identity: "unit",
                version: 1,
            },
        ),
    )
    .unwrap();
    assert!(plan
        .dag()
        .edges
        .windows(2)
        .all(|pair| pair[0].id < pair[1].id));
}
