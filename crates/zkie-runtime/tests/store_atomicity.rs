use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use zkie_runtime::{
    ArtifactMetadata, ArtifactStore, ContentStore, KeyMetadata, KeyStore, ObjectMetadata,
    StoreError,
};
use zkie_types::{Digest32, ProofFlavorId};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        let nonce = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "zkie-runtime-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn digest(bytes: &[u8]) -> Digest32 {
    Digest32::new(*blake3::hash(bytes).as_bytes())
}

fn artifact_metadata(id: &str, bytes: &[u8]) -> ObjectMetadata {
    ObjectMetadata::Artifact(ArtifactMetadata::new(id, digest(bytes), bytes.len() as u64).unwrap())
}

fn publish_artifact(
    store: &ArtifactStore,
    id: &str,
    bytes: &[u8],
) -> zkie_runtime::PublishedObject {
    let mut staged = store.stage(&artifact_metadata(id, bytes)).unwrap();
    staged.write_all(bytes).unwrap();
    let validated = store.validate(staged).unwrap();
    store.publish(validated).unwrap()
}

#[test]
fn publishes_and_reopens_only_digest_named_content() {
    let dir = TestDir::new("publish");
    let store = ArtifactStore::open(dir.path()).unwrap();
    let bytes = b"verified proof bytes";

    let published = publish_artifact(&store, "leaf-proof", bytes);

    assert_ne!(published.digest(), digest(bytes));
    assert!(dir
        .path()
        .join("objects")
        .join(published.digest().to_string())
        .is_file());
    let mut reopened = store.open_verified(published.digest()).unwrap();
    let mut actual = Vec::new();
    reopened.read_to_end(&mut actual).unwrap();
    assert_eq!(actual, bytes);
}

#[test]
fn validation_rejects_hash_and_size_mismatches() {
    let dir = TestDir::new("hash-mismatch");
    let store = ArtifactStore::open(dir.path()).unwrap();
    let expected = b"expected";
    let mut staged = store.stage(&artifact_metadata("proof", expected)).unwrap();
    staged.write_all(b"tampered").unwrap();

    assert!(matches!(
        store.validate(staged),
        Err(StoreError::ContentMismatch)
    ));
    assert!(fs::read_dir(dir.path().join("staging"))
        .unwrap()
        .next()
        .is_none());
    assert!(fs::read_dir(dir.path().join("objects"))
        .unwrap()
        .next()
        .is_none());
}

#[test]
fn validation_rereads_and_rejects_staged_metadata_tampering() {
    let dir = TestDir::new("metadata-mismatch");
    let store = ArtifactStore::open(dir.path()).unwrap();
    let bytes = b"proof";
    let mut staged = store.stage(&artifact_metadata("proof", bytes)).unwrap();
    staged.write_all(bytes).unwrap();
    let path = fs::read_dir(dir.path().join("staging"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let mut file = OpenOptions::new().write(true).open(path).unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();
    file.write_all(b"BADMAGIC").unwrap();
    file.sync_all().unwrap();

    assert!(matches!(
        store.validate(staged),
        Err(StoreError::MetadataMismatch)
    ));
}

#[test]
fn identical_publication_is_idempotent_and_metadata_changes_the_object_digest() {
    let dir = TestDir::new("duplicates");
    let store = ArtifactStore::open(dir.path()).unwrap();
    let bytes = b"same proof";
    let first = publish_artifact(&store, "leaf-a", bytes);
    let duplicate = publish_artifact(&store, "leaf-a", bytes);
    assert_eq!(duplicate.digest(), first.digest());

    let second = publish_artifact(&store, "leaf-b", bytes);
    assert_ne!(second.digest(), first.digest());
    assert!(dir
        .path()
        .join("objects")
        .join(first.digest().to_string())
        .is_file());
    assert!(dir
        .path()
        .join("objects")
        .join(second.digest().to_string())
        .is_file());
}

#[test]
fn staged_writer_rejects_payload_larger_than_declared_size_immediately() {
    let dir = TestDir::new("bounded-stage");
    let store = ArtifactStore::open(dir.path()).unwrap();
    let mut staged = store.stage(&artifact_metadata("proof", b"1234")).unwrap();

    assert!(staged.write_all(b"12345").is_err());
}

#[test]
fn publication_uses_the_pinned_objects_directory_after_path_replacement() {
    let dir = TestDir::new("parent-swap");
    let store = ArtifactStore::open(dir.path()).unwrap();
    let bytes = b"proof";
    let mut staged = store.stage(&artifact_metadata("proof", bytes)).unwrap();
    staged.write_all(bytes).unwrap();
    let validated = store.validate(staged).unwrap();
    let pinned_objects = dir.path().join("objects-pinned");
    fs::rename(dir.path().join("objects"), &pinned_objects).unwrap();
    fs::create_dir(dir.path().join("objects")).unwrap();

    let published = store.publish(validated).unwrap();

    assert!(pinned_objects
        .join(published.digest().to_string())
        .is_file());
    assert!(!dir
        .path()
        .join("objects")
        .join(published.digest().to_string())
        .exists());
}

#[test]
fn swapping_the_named_stage_after_validation_cannot_publish_attacker_bytes() {
    let dir = TestDir::new("stage-swap");
    let store = ArtifactStore::open(dir.path()).unwrap();
    let bytes = b"proof";
    let mut staged = store.stage(&artifact_metadata("proof", bytes)).unwrap();
    staged.write_all(bytes).unwrap();
    let validated = store.validate(staged).unwrap();
    let named_stage = fs::read_dir(dir.path().join("staging"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let displaced = dir.path().join("displaced-stage");
    fs::rename(&named_stage, displaced).unwrap();
    fs::write(&named_stage, b"attacker bytes").unwrap();

    assert!(matches!(
        store.publish(validated),
        Err(StoreError::FileIdentityChanged)
            | Err(StoreError::MetadataMismatch)
            | Err(StoreError::ContentMismatch)
    ));
    assert!(fs::read_dir(dir.path().join("objects"))
        .unwrap()
        .next()
        .is_none());
}

#[test]
fn a_writer_left_open_cannot_change_the_validated_object_that_gets_published() {
    let dir = TestDir::new("leftover-writer");
    let store = ArtifactStore::open(dir.path()).unwrap();
    let bytes = b"proof";
    let mut staged = store.stage(&artifact_metadata("proof", bytes)).unwrap();
    staged.write_all(bytes).unwrap();
    let stage_path = fs::read_dir(dir.path().join("staging"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let mut leftover = OpenOptions::new().write(true).open(stage_path).unwrap();
    let validated = store.validate(staged).unwrap();
    leftover.seek(SeekFrom::End(-1)).unwrap();
    leftover.write_all(b"X").unwrap();
    leftover.sync_all().unwrap();

    assert!(store.publish(validated).is_err());
    assert!(fs::read_dir(dir.path().join("objects"))
        .unwrap()
        .next()
        .is_none());
}

#[test]
fn a_writer_retained_through_publish_cannot_mutate_the_published_object() {
    let dir = TestDir::new("post-publish-writer");
    let store = ArtifactStore::open(dir.path()).unwrap();
    let bytes = b"proof";
    let mut staged = store.stage(&artifact_metadata("proof", bytes)).unwrap();
    staged.write_all(bytes).unwrap();
    let stage_path = fs::read_dir(dir.path().join("staging"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let mut retained_writer = OpenOptions::new().write(true).open(stage_path).unwrap();
    let published = store.publish(store.validate(staged).unwrap()).unwrap();

    retained_writer.seek(SeekFrom::End(-1)).unwrap();
    retained_writer.write_all(b"X").unwrap();
    retained_writer.sync_all().unwrap();

    let mut reopened = store.open_verified(published.digest()).unwrap();
    let mut actual = Vec::new();
    reopened.read_to_end(&mut actual).unwrap();
    assert_eq!(actual, bytes);
}

#[test]
#[cfg(unix)]
fn corrupt_existing_digest_object_is_never_overwritten_by_retry() {
    let dir = TestDir::new("existing-conflict");
    let store = ArtifactStore::open(dir.path()).unwrap();
    let bytes = b"proof";
    let published = publish_artifact(&store, "proof", bytes);
    let path = dir
        .path()
        .join("objects")
        .join(published.digest().to_string());
    let mut permissions = fs::metadata(&path).unwrap().permissions();
    permissions.set_mode(0o600);
    fs::set_permissions(&path, permissions).unwrap();
    let mut corrupt = OpenOptions::new().write(true).open(&path).unwrap();
    corrupt.seek(SeekFrom::End(-1)).unwrap();
    corrupt.write_all(b"X").unwrap();
    corrupt.sync_all().unwrap();
    let corrupted = fs::read(&path).unwrap();

    let mut staged = store.stage(&artifact_metadata("proof", bytes)).unwrap();
    staged.write_all(bytes).unwrap();
    let retry = store.validate(staged).unwrap();
    assert!(store.publish(retry).is_err());
    assert_eq!(fs::read(path).unwrap(), corrupted);
}

#[test]
#[cfg(unix)]
fn open_verified_recomputes_digest_after_metadata_tampering() {
    let dir = TestDir::new("metadata-domain");
    let store = ArtifactStore::open(dir.path()).unwrap();
    let published = publish_artifact(&store, "proof", b"proof");
    let path = dir
        .path()
        .join("objects")
        .join(published.digest().to_string());
    let mut permissions = fs::metadata(&path).unwrap().permissions();
    permissions.set_mode(0o600);
    fs::set_permissions(&path, permissions).unwrap();
    let mut object = OpenOptions::new().write(true).open(path).unwrap();
    object.seek(SeekFrom::Start(57)).unwrap();
    object.write_all(b"X").unwrap();
    object.sync_all().unwrap();

    assert!(store.open_verified(published.digest()).is_err());
}

#[test]
fn key_metadata_binds_full_proving_identity() {
    let dir = TestDir::new("keys");
    let store = KeyStore::open(dir.path()).unwrap();
    let bytes = b"proving and verifying key material";
    let metadata = KeyMetadata::new(
        digest(bytes),
        bytes.len() as u64,
        digest(b"trusted-srs-source"),
        ProofFlavorId::parse("halo2-kzg-bn256-shplonk-v1").unwrap(),
        digest(b"circuit"),
        19,
        4,
    )
    .unwrap();
    let expected = ObjectMetadata::Key(metadata.clone());
    let mut staged = store.stage(&expected).unwrap();
    staged.write_all(bytes).unwrap();

    let published = store.publish(store.validate(staged).unwrap()).unwrap();

    assert_eq!(published.metadata(), &ObjectMetadata::Key(metadata));
    let mut reopened = store.open_verified(published.digest()).unwrap();
    let mut actual = Vec::new();
    reopened.read_to_end(&mut actual).unwrap();
    assert_eq!(actual, bytes);
}

#[test]
fn stale_recovery_removes_owned_temps_and_quarantines_links_without_following_them() {
    let dir = TestDir::new("recover");
    let outside = TestDir::new("outside");
    let store = ArtifactStore::open(dir.path()).unwrap();
    let staged = store.stage(&artifact_metadata("stale", b"stale")).unwrap();
    let stale_path = fs::read_dir(dir.path().join("staging"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    drop(staged);
    let sentinel = outside.path().join("sentinel");
    fs::write(&sentinel, b"keep me").unwrap();
    let quarantine_sentinel = dir.path().join("quarantine").join("existing-evidence");
    fs::write(&quarantine_sentinel, b"preserve evidence").unwrap();

    #[cfg(unix)]
    std::os::unix::fs::symlink(
        &sentinel,
        dir.path()
            .join("staging")
            .join(format!("stage-{}.tmp", "d".repeat(64))),
    )
    .unwrap();

    let report = store.recover_stale().unwrap();

    assert!(!stale_path.exists());
    assert_eq!(fs::read(&sentinel).unwrap(), b"keep me");
    assert_eq!(fs::read(quarantine_sentinel).unwrap(), b"preserve evidence");
    assert_eq!(report.removed, 1);
    #[cfg(unix)]
    assert_eq!(report.quarantined, 1);
}

#[cfg(unix)]
#[test]
fn opening_store_rejects_symlinked_staging_directory() {
    let dir = TestDir::new("symlink-root");
    let outside = TestDir::new("symlink-outside");
    std::os::unix::fs::symlink(outside.path(), dir.path().join("staging")).unwrap();

    assert!(matches!(
        ArtifactStore::open(dir.path()),
        Err(StoreError::UnsafeFileType) | Err(StoreError::Fs(_))
    ));
}

#[test]
fn incomplete_store_identity_marker_fails_closed_without_replacing_it() {
    let dir = TestDir::new("incomplete-store-id");
    let marker = dir.path().join(".zkie-store-id");
    fs::write(&marker, b"partial").unwrap();

    assert!(matches!(
        ArtifactStore::open(dir.path()),
        Err(StoreError::StoreLocatorChanged)
    ));
    assert_eq!(fs::read(marker).unwrap(), b"partial");
}

#[test]
fn final_objects_reject_symlinks_and_non_regular_files() {
    let dir = TestDir::new("final-link");
    let outside = TestDir::new("final-link-outside");
    let store = ArtifactStore::open(dir.path()).unwrap();
    let bytes = b"proof";
    let target = outside.path().join("target");
    fs::write(&target, bytes).unwrap();
    let final_path = dir.path().join("objects").join(digest(bytes).to_string());

    #[cfg(unix)]
    std::os::unix::fs::symlink(&target, &final_path).unwrap();
    #[cfg(not(unix))]
    fs::create_dir(&final_path).unwrap();

    assert!(store.open_verified(digest(bytes)).is_err());
    assert_eq!(fs::read(target).unwrap(), bytes);
}

#[cfg(unix)]
#[test]
fn dangling_final_symlink_is_rejected_as_a_link() {
    let dir = TestDir::new("dangling-final-link");
    let store = ArtifactStore::open(dir.path()).unwrap();
    let expected = digest(b"missing proof");
    std::os::unix::fs::symlink(
        dir.path().join("does-not-exist"),
        dir.path().join("objects").join(expected.to_string()),
    )
    .unwrap();

    assert!(matches!(
        store.open_verified(expected),
        Err(StoreError::UnsafeFileType) | Err(StoreError::Fs(_))
    ));
}

#[test]
fn key_metadata_rejects_zero_aggregation_arity() {
    assert!(KeyMetadata::new(
        digest(b"key"),
        3,
        digest(b"srs"),
        ProofFlavorId::parse("flavor").unwrap(),
        digest(b"circuit"),
        10,
        0,
    )
    .is_err());
}
