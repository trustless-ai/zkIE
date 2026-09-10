use super::*;

impl RunDb {
    /// Atomically records a durable artifact and the successful independent verification.
    ///
    /// The crate-internal attestation binds the independently verified proof identity and source
    /// attempt to the exact durable object; callers outside the trusted verifier cannot construct it.
    #[allow(dead_code)]
    pub(crate) fn commit_verified_artifact(
        &mut self,
        verified: &VerifiedArtifact,
    ) -> Result<JobState, DbError> {
        let (attested_run, job, source_attempt, identity, published) = verified.binding();
        if attested_run != self.run_id {
            return Err(DbError::VerifiedArtifactRunMismatch);
        }
        let verifier = identity.verifier_id();
        let artifact_id = published
            .artifact_id()
            .ok_or(DbError::ArtifactPublicationRequired)?;
        let metadata = match published.metadata() {
            crate::ObjectMetadata::Artifact(metadata) => metadata,
            crate::ObjectMetadata::Key(_) => return Err(DbError::ArtifactPublicationRequired),
        };
        let audit = identity.audit();
        let (arity_tag, arity) = audit.aggregation_arity();
        let identity_digest = identity.digest();
        let attestation_digest = verified.attestation_digest();
        published.ensure_verified()?;
        published.ensure_current_locator()?;
        let run_id = self.run_id.clone();
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        // Full proof hashing happens before taking SQLite's write lock. Within the transaction only
        // inode/locator identity is checked again before the attested row and state transition land.
        published.ensure_current_locator()?;
        published.ensure_pinned_file_identity()?;
        require_verification_source(&transaction, &run_id, job, source_attempt)?;
        let (logical_job_id, persisted_kind): (String, String) = transaction.query_row(
            "SELECT logical_job_id,kind FROM jobs WHERE run_id=?1 AND job_id=?2",
            params![run_id, job.as_str()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        if persisted_kind != audit.job_kind() {
            return Err(DbError::VerifiedIdentityJobKindMismatch);
        }
        let ordinal: i64 = transaction.query_row(
            "SELECT COUNT(*) + 1 FROM verification_records WHERE run_id=?1 AND job_id=?2",
            params![run_id, job.as_str()],
            |row| row.get(0),
        )?;
        let verification =
            VerificationRecordId::new(format!("{}:verification:{ordinal}", job.as_str()))?;
        let now = unix_ms()?;
        transaction.execute(
            "INSERT INTO verification_records(verification_record_id,run_id,job_id,attempt_id,verifier,created_at_unix_ms) VALUES(?1,?2,?3,?4,?5,?6)",
            params![verification.as_str(), run_id, job.as_str(), source_attempt.as_str(), verifier, now],
        )?;
        transaction.execute(
            "INSERT INTO artifacts(
                artifact_id,run_id,job_id,logical_job_id,attempt_id,object_digest,content_digest,
                store_identity_digest,identity_digest,attestation_digest,artifact_role,job_kind,
                circuit_k,aggregation_arity_tag,aggregation_arity,artifact_manifest_digest,
                run_identity_digest,public_statement_digest,circuit_digest,verifying_key_digest,
                srs_source_digest,shard_identity_digest,witness_artifact_digest,proof_flavor,
                execution_backend,created_at_unix_ms
             ) VALUES(
                ?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,
                ?19,?20,?21,?22,?23,?24,?25,?26
             )",
            params![
                artifact_id,
                run_id,
                job.as_str(),
                logical_job_id,
                source_attempt.as_str(),
                published.digest().to_string(),
                metadata.content_digest().to_string(),
                published.store_identity_digest().to_string(),
                identity_digest.to_string(),
                attestation_digest.to_string(),
                audit.role(),
                audit.job_kind(),
                i64::from(audit.k()),
                arity_tag,
                arity.map(i64::from),
                audit.artifact_manifest_digest().to_string(),
                audit.run_identity_digest().to_string(),
                audit.public_statement_digest().to_string(),
                audit.circuit_digest().to_string(),
                audit.verifying_key_digest().to_string(),
                audit.srs_source_digest().to_string(),
                audit.shard_identity_digest().to_string(),
                audit.witness_artifact_digest().to_string(),
                audit.proof_flavor(),
                audit.execution_backend(),
                now,
            ],
        )?;
        let event = JobEvent::VerificationSucceeded {
            verification_record: Some(verification),
        };
        let state = apply_event_tx(
            &transaction,
            &run_id,
            job,
            Some(source_attempt),
            &event,
            None,
        )?;
        transaction.commit()?;
        Ok(state)
    }

    #[allow(dead_code)]
    pub(crate) fn commit_verified_key(
        &mut self,
        verified: &VerifiedKey,
    ) -> Result<JobState, DbError> {
        let (attested_run, job, source_attempt, identity, published) = verified.binding();
        if attested_run != self.run_id {
            return Err(DbError::VerifiedArtifactRunMismatch);
        }
        let metadata = identity.metadata();
        let identity_digest = identity.digest();
        let attestation_digest = verified.attestation_digest();
        published.ensure_verified()?;
        published.ensure_current_locator()?;
        let run_id = self.run_id.clone();
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        published.ensure_current_locator()?;
        published.ensure_pinned_file_identity()?;
        require_verification_source(&transaction, &run_id, job, source_attempt)?;
        let (logical_job_id, persisted_kind): (String, String) = transaction.query_row(
            "SELECT logical_job_id,kind FROM jobs WHERE run_id=?1 AND job_id=?2",
            params![run_id, job.as_str()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        if persisted_kind != identity.job_kind() {
            return Err(DbError::VerifiedIdentityJobKindMismatch);
        }
        let ordinal: i64 = transaction.query_row(
            "SELECT COUNT(*) + 1 FROM verification_records WHERE run_id=?1 AND job_id=?2",
            params![run_id, job.as_str()],
            |row| row.get(0),
        )?;
        let verification =
            VerificationRecordId::new(format!("{}:verification:{ordinal}", job.as_str()))?;
        let now = unix_ms()?;
        transaction.execute(
            "INSERT INTO verification_records(verification_record_id,run_id,job_id,attempt_id,verifier,created_at_unix_ms) VALUES(?1,?2,?3,?4,?5,?6)",
            params![verification.as_str(), run_id, job.as_str(), source_attempt.as_str(), metadata.proof_flavor().as_str(), now],
        )?;
        transaction.execute(
            "INSERT INTO key_artifacts(key_id,run_id,job_id,logical_job_id,attempt_id,object_digest,content_digest,content_size,store_identity_digest,identity_digest,attestation_digest,key_role,job_kind,srs_source_digest,proof_flavor,circuit_digest,circuit_k,aggregation_arity,created_at_unix_ms) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19)",
            params![
                identity_digest.to_string(), run_id, job.as_str(), logical_job_id,
                source_attempt.as_str(), published.digest().to_string(),
                metadata.content_digest().to_string(), i64::try_from(metadata.size()).map_err(|_| StoreError::InvalidMetadata)?, published.store_identity_digest().to_string(),
                identity_digest.to_string(), attestation_digest.to_string(), identity.role(),
                identity.job_kind(), metadata.srs_source_digest().to_string(),
                metadata.proof_flavor().as_str(), metadata.circuit_digest().to_string(),
                i64::from(metadata.k()), i64::from(metadata.aggregation_arity()), now,
            ],
        )?;
        let state = apply_event_tx(
            &transaction,
            &run_id,
            job,
            Some(source_attempt),
            &JobEvent::VerificationSucceeded {
                verification_record: Some(verification),
            },
            None,
        )?;
        transaction.commit()?;
        Ok(state)
    }

    // The crate-internal verifier worker introduced by the scheduler consumes this entry point.
    #[allow(dead_code)]
    pub(crate) fn complete_verification(
        &mut self,
        job: &JobId,
        attempt: &AttemptId,
        verifier: &str,
    ) -> Result<JobState, DbError> {
        validate_text_id(verifier)?;
        if self.job(job)?.kind != JobKind::LeafVerification {
            return Err(DbError::ControlledEventRequired);
        }
        let run_id = self.run_id.clone();
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)?;
        require_verification_source(&transaction, &run_id, job, attempt)?;
        let ordinal: i64 = transaction.query_row(
            "SELECT COUNT(*) + 1 FROM verification_records WHERE run_id=?1 AND job_id=?2",
            params![run_id, job.as_str()],
            |row| row.get(0),
        )?;
        let id = VerificationRecordId::new(format!("{}:verification:{ordinal}", job.as_str()))?;
        transaction.execute(
            "INSERT INTO verification_records(verification_record_id,run_id,job_id,attempt_id,verifier,created_at_unix_ms) VALUES(?1,?2,?3,?4,?5,?6)",
            params![id.as_str(), run_id, job.as_str(), attempt.as_str(), verifier, unix_ms()?],
        )?;
        let event = JobEvent::VerificationSucceeded {
            verification_record: Some(id),
        };
        let state = apply_event_tx(&transaction, &run_id, job, Some(attempt), &event, None)?;
        transaction.commit()?;
        Ok(state)
    }

    // The crate-internal verifier worker introduced by the scheduler consumes this entry point.
}
