use super::*;

impl RunDb {
    pub fn open(path: impl AsRef<Path>, run_id: &str) -> Result<Self, DbError> {
        validate_text_id(run_id)?;
        let mut connection = Connection::open(path)?;
        let has_schema_version: bool = connection.query_row(
            "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version')",
            [],
            |row| row.get(0),
        )?;
        let persisted_version = if has_schema_version {
            connection.query_row("SELECT MAX(version) FROM schema_version", [], |row| {
                row.get::<_, Option<i64>>(0)
            })?
        } else {
            None
        };
        if let Some(version) = persisted_version {
            if version < 1 || version > i64::from(SCHEMA_VERSION) {
                return Err(DbError::UnknownEnum {
                    column: "schema_version",
                    value: version.to_string(),
                });
            }
        }
        if persisted_version == Some(1) {
            let legacy_artifacts: i64 =
                connection.query_row("SELECT COUNT(*) FROM artifacts", [], |row| row.get(0))?;
            let legacy_verified: i64 = connection.query_row(
                "SELECT COUNT(*) FROM jobs WHERE state='verified'",
                [],
                |row| row.get(0),
            )?;
            let legacy_success_records: i64 =
                connection.query_row("SELECT COUNT(*) FROM verification_records", [], |row| {
                    row.get(0)
                })?;
            if legacy_artifacts != 0 || legacy_verified != 0 || legacy_success_records != 0 {
                return Err(DbError::LegacyArtifactsRequireReverification);
            }
        }

        connection.pragma_update(None, "journal_mode", "WAL")?;
        connection.pragma_update(None, "foreign_keys", true)?;
        connection.pragma_update(None, "synchronous", "FULL")?;
        let transaction = connection.transaction_with_behavior(TransactionBehavior::Immediate)?;
        transaction.execute_batch(SCHEMA_V1)?;
        let now = unix_ms()?;
        if persisted_version.is_none() {
            transaction.execute(
                "INSERT INTO schema_version(version, applied_at_unix_ms) VALUES(1, ?1)",
                params![now],
            )?;
        }
        if persisted_version.is_none() || persisted_version == Some(1) {
            transaction.execute_batch(MIGRATE_V1_TO_V2)?;
            transaction.execute(
                "INSERT INTO schema_version(version, applied_at_unix_ms) VALUES(2, ?1)",
                params![now],
            )?;
        }
        if persisted_version.is_none()
            || persisted_version == Some(1)
            || persisted_version == Some(2)
        {
            transaction.execute_batch(MIGRATE_V2_TO_V3)?;
            transaction.execute(
                "INSERT INTO schema_version(version, applied_at_unix_ms) VALUES(3, ?1)",
                params![now],
            )?;
        }
        transaction.execute(
            "INSERT OR IGNORE INTO runs(run_id, created_at_unix_ms) VALUES(?1, ?2)",
            params![run_id, now],
        )?;
        transaction.commit()?;
        Ok(Self {
            connection,
            run_id: run_id.to_owned(),
        })
    }

    pub fn schema_version(&self) -> Result<u32, DbError> {
        let value: i64 =
            self.connection
                .query_row("SELECT MAX(version) FROM schema_version", [], |row| {
                    row.get(0)
                })?;
        u32::try_from(value).map_err(|_| DbError::UnknownEnum {
            column: "schema_version",
            value: value.to_string(),
        })
    }

    pub fn sqlite_settings(&self) -> Result<SqliteSettings, DbError> {
        let journal_mode: String =
            self.connection
                .pragma_query_value(None, "journal_mode", |row| row.get(0))?;
        let foreign_keys: i64 =
            self.connection
                .pragma_query_value(None, "foreign_keys", |row| row.get(0))?;
        let synchronous: i64 = self
            .connection
            .pragma_query_value(None, "synchronous", |row| row.get(0))?;
        Ok(SqliteSettings {
            journal_mode: journal_mode.to_ascii_lowercase(),
            foreign_keys: foreign_keys == 1,
            synchronous: match synchronous {
                2 => "FULL".into(),
                other => other.to_string(),
            },
        })
    }

    pub fn insert_job(&mut self, logical_job_id: &str, kind: JobKind) -> Result<JobId, DbError> {
        validate_text_id(logical_job_id)?;
        let id = JobId::new(format!("{}:{logical_job_id}", self.run_id))?;
        let now = unix_ms()?;
        self.connection.execute(
            "INSERT INTO jobs(job_id, run_id, logical_job_id, kind, state, created_at_unix_ms, updated_at_unix_ms) VALUES(?1,?2,?3,?4,?5,?6,?6)",
            params![id.as_str(), self.run_id, logical_job_id, kind.as_str(), JobState::Pending.as_str(), now],
        )?;
        Ok(id)
    }

    pub fn add_dependency(
        &mut self,
        predecessor: &JobId,
        successor: &JobId,
    ) -> Result<(), DbError> {
        self.require_job_in_run(predecessor)?;
        self.require_job_in_run(successor)?;
        self.connection.execute(
            "INSERT INTO dependencies(run_id, predecessor_job_id, successor_job_id) VALUES(?1,?2,?3)",
            params![self.run_id, predecessor.as_str(), successor.as_str()],
        )?;
        Ok(())
    }

    pub fn job(&self, id: &JobId) -> Result<JobRecord, DbError> {
        let raw = self
            .connection
            .query_row(
                "SELECT logical_job_id,kind,state,failure_kind,failure_stage,failure_code,failure_summary,attempt_count,execution_retries,resource_requeues,terminal_failure,blocking_predecessor_id FROM jobs WHERE run_id=?1 AND job_id=?2",
                params![self.run_id, id.as_str()],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, Option<String>>(3)?,
                        row.get::<_, Option<String>>(4)?,
                        row.get::<_, Option<String>>(5)?,
                        row.get::<_, Option<String>>(6)?,
                        row.get::<_, i64>(7)?,
                        row.get::<_, i64>(8)?,
                        row.get::<_, i64>(9)?,
                        row.get::<_, i64>(10)?,
                        row.get::<_, Option<String>>(11)?,
                    ))
                },
            )
            .optional()?
            .ok_or_else(|| DbError::JobNotFound(id.clone()))?;
        Ok(JobRecord {
            id: id.clone(),
            logical_job_id: raw.0,
            kind: parse_kind(&raw.1)?,
            state: parse_state(&raw.2)?,
            failure_kind: raw.3.as_deref().map(parse_failure_kind).transpose()?,
            failure_stage: raw.4.as_deref().map(parse_failure_stage).transpose()?,
            failure_code: raw.5.as_deref().map(parse_failure_code).transpose()?,
            failure_summary: raw.6,
            attempt_count: u32::try_from(raw.7).map_err(|_| DbError::UnknownEnum {
                column: "attempt_count",
                value: raw.7.to_string(),
            })?,
            execution_retries: parse_u32("execution_retries", raw.8)?,
            resource_requeues: parse_u32("resource_requeues", raw.9)?,
            terminal_failure: match raw.10 {
                0 => false,
                1 => true,
                value => {
                    return Err(DbError::UnknownEnum {
                        column: "terminal_failure",
                        value: value.to_string(),
                    })
                }
            },
            blocking_predecessor_id: raw.11.map(JobId::new).transpose()?,
        })
    }

    pub fn attempt(&self, id: &AttemptId) -> Result<AttemptRecord, DbError> {
        let raw = self
            .connection
            .query_row(
                "SELECT job_id,state,started_at_unix_ms,finished_at_unix_ms FROM attempts WHERE run_id=?1 AND attempt_id=?2",
                params![self.run_id, id.as_str()],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get(2)?, row.get(3)?)),
            )
            .optional()?
            .ok_or_else(|| DbError::AttemptNotFound(id.clone()))?;
        Ok(AttemptRecord {
            id: id.clone(),
            job_id: JobId::new(raw.0)?,
            state: parse_state(&raw.1)?,
            started_at_unix_ms: raw.2,
            finished_at_unix_ms: raw.3,
        })
    }

    pub fn events(&self, id: &JobId) -> Result<Vec<StateEventRecord>, DbError> {
        self.require_job_in_run(id)?;
        let mut statement = self.connection.prepare(
            "SELECT from_state,to_state,event_type,created_at_unix_ms FROM state_events WHERE run_id=?1 AND job_id=?2 ORDER BY event_id",
        )?;
        let rows = statement.query_map(params![self.run_id, id.as_str()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })?;
        rows.map(|row| {
            let (from, to, event_type, created_at_unix_ms) = row?;
            PersistedEventType::try_from(event_type.as_str()).map_err(|_| {
                DbError::UnknownEnum {
                    column: "event_type",
                    value: event_type.clone(),
                }
            })?;
            Ok(StateEventRecord {
                from_state: parse_state(&from)?,
                to_state: parse_state(&to)?,
                event_type,
                created_at_unix_ms,
            })
        })
        .collect()
    }

    #[allow(dead_code)]
    pub(crate) fn artifact_for_job(&self, job: &JobId) -> Result<Option<ArtifactRecord>, DbError> {
        self.require_job_in_run(job)?;
        let raw = self
            .connection
            .query_row(
                "SELECT artifact_id,object_digest,content_digest,store_identity_digest,identity_digest,attestation_digest FROM artifacts WHERE run_id=?1 AND job_id=?2",
                params![self.run_id, job.as_str()],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                    ))
                },
            )
            .optional()?;
        raw.map(
            |(artifact_id, object, content, store, identity, attestation)| {
                Ok(ArtifactRecord {
                    artifact_id,
                    object_digest: parse_digest("artifact_object_digest", object)?,
                    content_digest: parse_digest("artifact_content_digest", content)?,
                    store_identity_digest: parse_digest("artifact_store_identity_digest", store)?,
                    identity_digest: parse_digest("artifact_identity_digest", identity)?,
                    attestation_digest: parse_digest("artifact_attestation_digest", attestation)?,
                })
            },
        )
        .transpose()
    }

    #[allow(dead_code)]
    pub(crate) fn reopen_artifact_for_job(
        &self,
        store: &ArtifactStore,
        job: &JobId,
    ) -> Result<Option<File>, DbError> {
        let Some(record) = self.artifact_for_job(job)? else {
            return Ok(None);
        };
        let identity = self.persisted_proof_identity(job)?;
        if identity.digest() != record.identity_digest {
            return Err(StoreError::ProofIdentityMismatch.into());
        }
        let (file, metadata) =
            store.reopen_persisted(record.store_identity_digest, record.object_digest)?;
        let artifact = match metadata {
            crate::ObjectMetadata::Artifact(value) => value,
            crate::ObjectMetadata::Key(_) => return Err(StoreError::WrongStoreKind.into()),
        };
        if artifact.artifact_id() != record.artifact_id
            || artifact.proof_identity_digest() != Some(record.identity_digest)
            || artifact.content_digest() != record.content_digest
        {
            return Err(StoreError::ProofIdentityMismatch.into());
        }
        let (run_id, job_id, attempt_id): (String, String, String) = self.connection.query_row(
            "SELECT run_id,job_id,attempt_id FROM artifacts WHERE run_id=?1 AND job_id=?2",
            params![self.run_id, job.as_str()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        let attempt = AttemptId::new(attempt_id)?;
        if crate::proof_attestation_digest(
            record.identity_digest,
            record.object_digest,
            &run_id,
            &JobId::new(job_id)?,
            &attempt,
        ) != record.attestation_digest
        {
            return Err(StoreError::ProofIdentityMismatch.into());
        }
        Ok(Some(file))
    }

    fn persisted_proof_identity(&self, job: &JobId) -> Result<TrustedProofIdentity, DbError> {
        type RawProofIdentity = (
            String,
            String,
            i64,
            i64,
            Option<i64>,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
        );
        let raw: RawProofIdentity = self.connection.query_row(
            "SELECT artifact_role,job_kind,circuit_k,aggregation_arity_tag,aggregation_arity,
                    run_id,job_id,attempt_id,content_digest,artifact_manifest_digest,
                    run_identity_digest,public_statement_digest,circuit_digest,verifying_key_digest,
                    srs_source_digest,shard_identity_digest,witness_artifact_digest,
                    proof_flavor,execution_backend,identity_digest
             FROM artifacts WHERE run_id=?1 AND job_id=?2",
            params![self.run_id, job.as_str()],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                    row.get(8)?,
                    row.get(9)?,
                    row.get(10)?,
                    row.get(11)?,
                    row.get(12)?,
                    row.get(13)?,
                    row.get(14)?,
                    row.get(15)?,
                    row.get(16)?,
                    row.get(17)?,
                    row.get(18)?,
                    row.get(19)?,
                ))
            },
        )?;
        let role = match raw.0.as_str() {
            "witness" => ArtifactRole::Witness,
            "prepared-material" => ArtifactRole::PreparedMaterial,
            "leaf-proof" => ArtifactRole::LeafProof,
            "native-verified-manifest" => ArtifactRole::NativeVerifiedManifest,
            value => {
                return Err(DbError::UnknownEnum {
                    column: "artifact_role",
                    value: value.to_owned(),
                })
            }
        };
        let kind = JobKind::try_from(raw.1.as_str()).map_err(|_| DbError::UnknownEnum {
            column: "artifact_job_kind",
            value: raw.1.clone(),
        })?;
        let k = parse_u32("artifact_circuit_k", raw.2)?;
        let arity = match (raw.3, raw.4) {
            (0, None) => AggregationArity::NotApplicable,
            (1, Some(value)) => {
                AggregationArity::Actual(parse_u32("artifact_aggregation_arity", value)?)
            }
            _ => {
                return Err(DbError::UnknownEnum {
                    column: "artifact_aggregation_arity_tag",
                    value: raw.3.to_string(),
                })
            }
        };
        let run_id = raw.5;
        let persisted_job = JobId::new(raw.6)?;
        let attempt = AttemptId::new(raw.7)?;
        let identity = TrustedProofIdentity::new(
            role,
            kind,
            k,
            arity,
            &run_id,
            &persisted_job,
            &attempt,
            parse_digest("artifact_content_digest", raw.8)?,
            parse_digest("artifact_manifest_digest", raw.9)?,
            parse_digest("artifact_run_identity_digest", raw.10)?,
            parse_digest("artifact_public_statement_digest", raw.11)?,
            parse_digest("artifact_circuit_digest", raw.12)?,
            parse_digest("artifact_verifying_key_digest", raw.13)?,
            parse_digest("artifact_srs_source_digest", raw.14)?,
            parse_digest("artifact_shard_identity_digest", raw.15)?,
            parse_digest("artifact_witness_artifact_digest", raw.16)?,
            zkie_types::ProofFlavorId::parse(raw.17).map_err(|_| StoreError::InvalidMetadata)?,
            zkie_types::ExecutionBackendId::parse(raw.18)
                .map_err(|_| StoreError::InvalidMetadata)?,
        )?;
        if identity.digest() != parse_digest("artifact_identity_digest", raw.19)? {
            return Err(StoreError::ProofIdentityMismatch.into());
        }
        Ok(identity)
    }

    #[allow(dead_code)]
    pub(crate) fn key_for_job(&self, job: &JobId) -> Result<Option<KeyArtifactRecord>, DbError> {
        self.require_job_in_run(job)?;
        let raw = self.connection.query_row(
            "SELECT key_id,object_digest,content_digest,store_identity_digest,identity_digest,attestation_digest FROM key_artifacts WHERE run_id=?1 AND job_id=?2",
            params![self.run_id, job.as_str()],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?, row.get::<_, String>(3)?, row.get::<_, String>(4)?, row.get::<_, String>(5)?)),
        ).optional()?;
        raw.map(|(key_id, object, content, store, identity, attestation)| {
            Ok(KeyArtifactRecord {
                key_id,
                object_digest: parse_digest("key_object_digest", object)?,
                content_digest: parse_digest("key_content_digest", content)?,
                store_identity_digest: parse_digest("key_store_identity_digest", store)?,
                identity_digest: parse_digest("key_identity_digest", identity)?,
                attestation_digest: parse_digest("key_attestation_digest", attestation)?,
            })
        })
        .transpose()
    }

    #[allow(dead_code)]
    pub(crate) fn reopen_key_for_job(
        &self,
        store: &KeyStore,
        job: &JobId,
    ) -> Result<Option<File>, DbError> {
        let Some(record) = self.key_for_job(job)? else {
            return Ok(None);
        };
        let identity = self.persisted_key_identity(job)?;
        if identity.digest() != record.identity_digest {
            return Err(StoreError::KeyIdentityMismatch.into());
        }
        let (file, metadata) =
            store.reopen_persisted(record.store_identity_digest, record.object_digest)?;
        let key = match metadata {
            crate::ObjectMetadata::Key(value) => value,
            crate::ObjectMetadata::Artifact(_) => return Err(StoreError::WrongStoreKind.into()),
        };
        if key.key_identity_digest() != Some(record.identity_digest)
            || key.content_digest() != record.content_digest
        {
            return Err(StoreError::KeyIdentityMismatch.into());
        }
        let (run_id, job_id, attempt_id): (String, String, String) = self.connection.query_row(
            "SELECT run_id,job_id,attempt_id FROM key_artifacts WHERE run_id=?1 AND job_id=?2",
            params![self.run_id, job.as_str()],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        if crate::key_attestation_digest(
            record.identity_digest,
            record.object_digest,
            &run_id,
            &JobId::new(job_id)?,
            &AttemptId::new(attempt_id)?,
        ) != record.attestation_digest
        {
            return Err(StoreError::KeyIdentityMismatch.into());
        }
        Ok(Some(file))
    }

    fn persisted_key_identity(&self, job: &JobId) -> Result<TrustedKeyIdentity, DbError> {
        type RawKeyIdentity = (
            String,
            String,
            String,
            String,
            String,
            String,
            i64,
            String,
            String,
            String,
            i64,
            i64,
            String,
        );
        let raw: RawKeyIdentity = self.connection.query_row(
            "SELECT key_role,job_kind,run_id,job_id,attempt_id,content_digest,content_size,
                    srs_source_digest,proof_flavor,circuit_digest,circuit_k,aggregation_arity,
                    identity_digest
             FROM key_artifacts WHERE run_id=?1 AND job_id=?2",
            params![self.run_id, job.as_str()],
            |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                    row.get(8)?,
                    row.get(9)?,
                    row.get(10)?,
                    row.get(11)?,
                    row.get(12)?,
                ))
            },
        )?;
        if raw.0 != "proving-and-verifying-key" {
            return Err(DbError::UnknownEnum {
                column: "key_role",
                value: raw.0,
            });
        }
        let kind = JobKind::try_from(raw.1.as_str()).map_err(|_| DbError::UnknownEnum {
            column: "key_job_kind",
            value: raw.1.clone(),
        })?;
        let run_id = raw.2;
        let persisted_job = JobId::new(raw.3)?;
        let attempt = AttemptId::new(raw.4)?;
        let metadata = KeyMetadata::new(
            parse_digest("key_content_digest", raw.5)?,
            u64::try_from(raw.6).map_err(|_| DbError::UnknownEnum {
                column: "key_content_size",
                value: raw.6.to_string(),
            })?,
            parse_digest("key_srs_source_digest", raw.7)?,
            zkie_types::ProofFlavorId::parse(raw.8).map_err(|_| StoreError::InvalidMetadata)?,
            parse_digest("key_circuit_digest", raw.9)?,
            parse_u32("key_circuit_k", raw.10)?,
            parse_u32("key_aggregation_arity", raw.11)?,
        )?;
        let identity = TrustedKeyIdentity::new(
            KeyRole::ProvingAndVerifyingKey,
            kind,
            &run_id,
            &persisted_job,
            &attempt,
            metadata,
        )?;
        if identity.digest() != parse_digest("key_identity_digest", raw.12)? {
            return Err(StoreError::KeyIdentityMismatch.into());
        }
        Ok(identity)
    }
}
