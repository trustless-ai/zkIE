use super::*;

impl RunDb {
    #[allow(dead_code)]
    pub(crate) fn fail_verification(
        &mut self,
        job: &JobId,
        source_attempt: &AttemptId,
        failure: FailureRecord,
    ) -> Result<JobState, DbError> {
        let run_id = self.run_id.clone();
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        require_verification_source(&transaction, &run_id, job, source_attempt)?;
        let event = JobEvent::VerificationFailed(failure);
        let state = apply_event_tx(
            &transaction,
            &run_id,
            job,
            Some(source_attempt),
            &event,
            None,
        )?;
        transaction.execute(
            "UPDATE jobs SET terminal_failure=1 WHERE run_id=?1 AND job_id=?2",
            params![run_id, job.as_str()],
        )?;
        block_descendants_tx(&transaction, &run_id, job)?;
        transaction.commit()?;
        Ok(state)
    }

    pub fn record_resource_peak(
        &mut self,
        key: &ReservationKey,
        peak_bytes: u64,
    ) -> Result<(), DbError> {
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        record_resource_peak_tx(&transaction, key, peak_bytes)?;
        transaction.commit()?;
        Ok(())
    }

    pub fn max_observed_peak_bytes(&self, key: &ReservationKey) -> Result<Option<u64>, DbError> {
        let raw = self
            .connection
            .query_row(
                "SELECT max_observed_peak_bytes FROM resource_history
                 WHERE circuit_digest=?1 AND circuit_k=?2 AND proof_flavor=?3
                   AND execution_backend=?4 AND hardware_profile=?5",
                params![
                    key.circuit_digest.to_string(),
                    i64::from(key.k),
                    key.proof_flavor.as_str(),
                    key.execution_backend.as_str(),
                    key.hardware_profile.to_string(),
                ],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        raw.map(exact_peak_from_sql).transpose()
    }

    pub fn handle_attempt_failure(
        &mut self,
        job: &JobId,
        attempt: &AttemptId,
        failure: FailureRecord,
        resource_observation: Option<(&ReservationKey, u64)>,
    ) -> Result<RetryDecision, DbError> {
        let is_resource = failure.kind() == FailureKind::ResourceExceeded;
        if is_resource != resource_observation.is_some()
            || !matches!(
                failure.kind(),
                FailureKind::Execution | FailureKind::ResourceExceeded
            )
        {
            return Err(DbError::InvalidFailureObservation);
        }
        let run_id = self.run_id.clone();
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        require_unique_open_attempt(&transaction, &run_id, job, attempt)?;
        if let Some((key, peak)) = resource_observation {
            // Persist the observation before any reservation or retry decision is derived.
            record_resource_peak_tx(&transaction, key, peak)?;
        }
        let failure_event = if is_resource {
            JobEvent::ResourceExceeded(failure)
        } else {
            JobEvent::ExecutionFailed(failure)
        };
        apply_event_tx(
            &transaction,
            &run_id,
            job,
            Some(attempt),
            &failure_event,
            None,
        )?;

        let counter_column = if is_resource {
            "resource_requeues"
        } else {
            "execution_retries"
        };
        let retry_limit = if is_resource { 3_i64 } else { 2_i64 };
        let retries: i64 = transaction.query_row(
            &format!("SELECT {counter_column} FROM jobs WHERE run_id=?1 AND job_id=?2"),
            params![run_id, job.as_str()],
            |row| row.get(0),
        )?;
        let decision = if retries < retry_limit {
            transaction.execute(
                &format!("UPDATE jobs SET {counter_column}={counter_column}+1 WHERE run_id=?1 AND job_id=?2"),
                params![run_id, job.as_str()],
            )?;
            let retry_event = if is_resource {
                JobEvent::RetryResourceExceeded
            } else {
                JobEvent::RetryExecution
            };
            apply_event_tx(
                &transaction,
                &run_id,
                job,
                Some(attempt),
                &retry_event,
                None,
            )?;
            RetryDecision::Requeued
        } else {
            transaction.execute(
                "UPDATE jobs SET terminal_failure=1 WHERE run_id=?1 AND job_id=?2",
                params![run_id, job.as_str()],
            )?;
            block_descendants_tx(&transaction, &run_id, job)?;
            RetryDecision::Terminal
        };
        transaction.commit()?;
        Ok(decision)
    }

    pub fn apply_event(&mut self, _job: &JobId, event: JobEvent) -> Result<JobState, DbError> {
        match event {
            JobEvent::StartWitnessing
            | JobEvent::StartPreparing
            | JobEvent::StartProving
            | JobEvent::StartVerification => Err(DbError::AttemptStartRequiresBegin),
            JobEvent::WitnessCompleted
            | JobEvent::PreparationCompleted
            | JobEvent::ProofProduced
            | JobEvent::ExecutionFailed(_)
            | JobEvent::ResourceExceeded(_)
            | JobEvent::Interrupted(_) => Err(DbError::AttemptEventRequiresAttempt),
            JobEvent::DependenciesSatisfied
            | JobEvent::DependencyBlocked { .. }
            | JobEvent::VerificationSucceeded { .. }
            | JobEvent::VerificationFailed(_)
            | JobEvent::RetryExecution
            | JobEvent::RetryResourceExceeded
            | JobEvent::RecoveryRequeue => Err(DbError::ControlledEventRequired),
        }
    }

    pub fn apply_attempt_event(
        &mut self,
        job: &JobId,
        attempt: &AttemptId,
        event: JobEvent,
    ) -> Result<JobState, DbError> {
        if !matches!(
            event,
            JobEvent::WitnessCompleted
                | JobEvent::PreparationCompleted
                | JobEvent::ProofProduced
                | JobEvent::Interrupted(_)
        ) {
            return Err(DbError::ControlledEventRequired);
        }
        let run_id = self.run_id.clone();
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        require_unique_open_attempt(&transaction, &run_id, job, attempt)?;
        let state = apply_event_tx(&transaction, &run_id, job, Some(attempt), &event, None)?;
        transaction.commit()?;
        Ok(state)
    }

    pub fn begin_attempt(&mut self, job: &JobId, event: JobEvent) -> Result<AttemptId, DbError> {
        if !event.starts_attempt() {
            return Err(DbError::NotAttemptStart);
        }
        let run_id = self.run_id.clone();
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let (kind_raw, count): (String, i64) = transaction
            .query_row(
                "SELECT kind,attempt_count FROM jobs WHERE run_id=?1 AND job_id=?2",
                params![run_id, job.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?
            .ok_or_else(|| DbError::JobNotFound(job.clone()))?;
        let kind = parse_kind(&kind_raw)?;
        if !start_matches_kind(kind, &event) {
            return Err(DbError::WrongStartForJobKind {
                kind,
                event: event.name(),
            });
        }
        let open_attempts: i64 = transaction.query_row(
            "SELECT COUNT(*) FROM attempts WHERE run_id=?1 AND job_id=?2 AND finished_at_unix_ms IS NULL",
            params![run_id, job.as_str()],
            |row| row.get(0),
        )?;
        if open_attempts != 0 {
            return Err(DbError::OpenAttemptConflict { job: job.clone() });
        }
        let count = u32::try_from(count).map_err(|_| DbError::UnknownEnum {
            column: "attempt_count",
            value: count.to_string(),
        })?;
        let next_count = count
            .checked_add(1)
            .ok_or_else(|| DbError::AttemptCountExhausted { job: job.clone() })?;
        let attempt = AttemptId::new(format!("{}:attempt:{next_count}", job.as_str()))?;
        let next = current_state(&transaction, &run_id, job)?.transition(event.clone())?;
        let now = unix_ms()?;
        transaction.execute(
            "INSERT INTO attempts(attempt_id,run_id,job_id,state,started_at_unix_ms) VALUES(?1,?2,?3,?4,?5)",
            params![attempt.as_str(), run_id, job.as_str(), next.as_str(), now],
        )?;
        apply_event_tx(
            &transaction,
            &run_id,
            job,
            Some(&attempt),
            &event,
            Some(next_count),
        )?;
        transaction.commit()?;
        Ok(attempt)
    }

    pub(crate) fn required_predecessors_verified(&self, job: &JobId) -> Result<bool, DbError> {
        self.job(job)?;
        let all_verified: i64 = self.connection.query_row(
            "SELECT NOT EXISTS(
                SELECT 1
                FROM dependencies d
                JOIN jobs p ON p.job_id=d.predecessor_job_id
                WHERE d.run_id=?1 AND d.successor_job_id=?2 AND p.state<>'verified'
            )",
            params![self.run_id, job.as_str()],
            |row| row.get(0),
        )?;
        Ok(all_verified == 1)
    }

    pub fn refresh_readiness(&mut self) -> Result<usize, DbError> {
        let run_id = self.run_id.clone();
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let candidates = {
            let mut statement = transaction.prepare(
                "SELECT job_id,state FROM jobs WHERE run_id=?1 AND state IN ('pending','requeued') ORDER BY job_id",
            )?;
            let rows = statement.query_map([&run_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            rows.collect::<Result<Vec<_>, _>>()?
        };
        let mut ready = 0;
        for (raw_id, raw_state) in candidates {
            let job = JobId::new(raw_id)?;
            parse_state(&raw_state)?;
            let predecessors = predecessor_states(&transaction, &run_id, &job)?;
            if predecessors
                .iter()
                .all(|(_, state, _)| *state == JobState::Verified)
            {
                apply_event_tx(
                    &transaction,
                    &run_id,
                    &job,
                    None,
                    &JobEvent::DependenciesSatisfied,
                    None,
                )?;
                ready += 1;
            } else if let Some((predecessor, _, _)) =
                predecessors.iter().find(|(_, state, terminal)| {
                    *terminal || matches!(state, JobState::VerificationFailed | JobState::Blocked)
                })
            {
                apply_event_tx(
                    &transaction,
                    &run_id,
                    &job,
                    None,
                    &JobEvent::DependencyBlocked {
                        predecessor: predecessor.clone(),
                    },
                    None,
                )?;
            }
        }
        transaction.commit()?;
        Ok(ready)
    }

    pub fn recover_interrupted(&mut self) -> Result<usize, DbError> {
        let run_id = self.run_id.clone();
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        let active = {
            let mut statement = transaction.prepare(
                "SELECT a.attempt_id,a.job_id,j.state FROM attempts a JOIN jobs j ON j.job_id=a.job_id WHERE a.run_id=?1 AND a.finished_at_unix_ms IS NULL ORDER BY a.attempt_id",
            )?;
            let rows = statement.query_map([&run_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?;
            rows.collect::<Result<Vec<_>, _>>()?
        };
        let mut recovered = 0;
        for (attempt_raw, job_raw, state_raw) in active {
            let state = parse_state(&state_raw)?;
            if !state.is_active() {
                continue;
            }
            let attempt = AttemptId::new(attempt_raw)?;
            let job = JobId::new(job_raw)?;
            let interruption = FailureRecord::new(
                FailureKind::Interrupted,
                FailureStage::Scheduler,
                FailureCode::SchedulerRestart,
            );
            apply_event_tx(
                &transaction,
                &run_id,
                &job,
                Some(&attempt),
                &JobEvent::Interrupted(interruption),
                None,
            )?;
            apply_event_tx(
                &transaction,
                &run_id,
                &job,
                Some(&attempt),
                &JobEvent::RecoveryRequeue,
                None,
            )?;
            transaction.execute(
                "UPDATE attempts SET state='interrupted',finished_at_unix_ms=?1 WHERE attempt_id=?2 AND finished_at_unix_ms IS NULL",
                params![unix_ms()?, attempt.as_str()],
            )?;
            recovered += 1;
        }
        transaction.commit()?;
        Ok(recovered)
    }

    pub(crate) fn require_job_in_run(&self, job: &JobId) -> Result<(), DbError> {
        let present: bool = self.connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM jobs WHERE run_id=?1 AND job_id=?2)",
            params![self.run_id, job.as_str()],
            |row| row.get(0),
        )?;
        if present {
            Ok(())
        } else {
            Err(DbError::JobNotFound(job.clone()))
        }
    }

    /// Every job of the run, ordered by logical id so status output is stable.
    pub fn jobs(&self) -> Result<Vec<JobRecord>, DbError> {
        let ids = {
            let mut statement = self
                .connection
                .prepare("SELECT job_id FROM jobs WHERE run_id=?1 ORDER BY logical_job_id")?;
            let rows = statement.query_map([&self.run_id], |row| row.get::<_, String>(0))?;
            rows.collect::<Result<Vec<_>, _>>()?
        };
        ids.into_iter()
            .map(|raw| self.job(&JobId::new(raw)?))
            .collect()
    }
}
