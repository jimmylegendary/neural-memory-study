from __future__ import annotations

import hashlib
import multiprocessing
import os
import signal
import stat
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from fcntl import LOCK_EX, LOCK_UN, flock
from pathlib import Path
from threading import Barrier, Event, Lock, current_thread

import pytest

from stc_research import jsonl_store
from stc_research.jsonl_store import (
    DuplicateIdError,
    append_unique,
    read_jsonl,
    write_jsonl_atomic,
)


def _process_append_worker(path, number, barrier, outcomes, duplicate):
    try:
        barrier.wait(timeout=5)
        source_id = "SRC-STC-0001" if duplicate else f"SRC-STC-{number:04d}"
        append_unique(path, {"source_id": source_id}, "source_id")
    except DuplicateIdError:
        outcomes.put(("duplicate", number))
    else:
        outcomes.put(("success", number))


def _killed_lock_holder(path, ready):
    destination = Path(path)
    lock_path = destination.with_name(f"{destination.name}.lock")
    orphan = destination.with_name(
        f"{jsonl_store._temporary_prefix(destination)}killed.tmp"
    )
    with lock_path.open("a+b") as lock_stream:
        flock(lock_stream.fileno(), LOCK_EX)
        orphan.write_text("orphan", encoding="utf-8")
        ready.set()
        while True:
            time.sleep(60)


def _join_processes_or_fail(processes, timeout=10):
    deadline = time.monotonic() + timeout
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.kill()
        process.join(timeout=2)
    assert not alive, "multiprocessing workers exceeded stable timeout"


def test_write_jsonl_is_canonical_and_atomic(tmp_path):
    path = tmp_path / "sources.jsonl"
    write_jsonl_atomic(
        path,
        [
            {"source_id": "SRC-STC-0002", "title": "B"},
            {"source_id": "SRC-STC-0001", "title": "A"},
        ],
    )
    assert path.read_text(encoding="utf-8") == (
        '{"source_id":"SRC-STC-0001","title":"A"}\n'
        '{"source_id":"SRC-STC-0002","title":"B"}\n'
    )
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_write_failure_preserves_existing_file_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "sources.jsonl"
    path.write_text('{"source_id":"SRC-STC-0001"}\n', encoding="utf-8")
    expected_current_digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def fail_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        write_jsonl_atomic(
            path,
            [{"source_id": "SRC-STC-0002"}],
            expected_current_digest=expected_current_digest,
        )

    assert path.read_text(encoding="utf-8") == ('{"source_id":"SRC-STC-0001"}\n')
    assert not list(tmp_path.glob(".*.tmp"))


def test_absent_registry_uses_declared_default_permissions(tmp_path):
    path = tmp_path / "sources.jsonl"
    write_jsonl_atomic(path, [{"source_id": "SRC-STC-0001"}])

    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_existing_registry_replacement_preserves_permission_bits(tmp_path):
    path = tmp_path / "sources.jsonl"
    write_jsonl_atomic(path, [{"source_id": "SRC-STC-0001"}])
    path.chmod(0o640)
    expected_current_digest = hashlib.sha256(path.read_bytes()).hexdigest()

    write_jsonl_atomic(
        path,
        [{"source_id": "SRC-STC-0002"}],
        expected_current_digest=expected_current_digest,
    )

    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_existing_registry_replacement_strips_special_mode_bits(tmp_path):
    path = tmp_path / "sources.jsonl"
    write_jsonl_atomic(path, [{"source_id": "SRC-STC-0001"}])
    path.chmod(0o7751)
    assert stat.S_IMODE(path.stat().st_mode) == 0o7751
    expected_current_digest = hashlib.sha256(path.read_bytes()).hexdigest()

    write_jsonl_atomic(
        path,
        [{"source_id": "SRC-STC-0002"}],
        expected_current_digest=expected_current_digest,
    )

    assert stat.S_IMODE(path.stat().st_mode) == 0o751


def test_temp_fchmod_precedes_file_fsync_and_atomic_replace(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "sources.jsonl"
    events = []
    real_fchmod = os.fchmod
    real_fsync = os.fsync
    real_replace = os.replace

    def record_fchmod(descriptor, mode):
        events.append(("fchmod", stat.S_IMODE(mode)))
        return real_fchmod(descriptor, mode)

    def record_fsync(descriptor):
        descriptor_mode = os.fstat(descriptor).st_mode
        event = "file-fsync" if stat.S_ISREG(descriptor_mode) else "directory-fsync"
        events.append((event, None))
        return real_fsync(descriptor)

    def record_replace(source, destination):
        events.append(("replace", None))
        return real_replace(source, destination)

    monkeypatch.setattr(os, "fchmod", record_fchmod)
    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(os, "replace", record_replace)
    write_jsonl_atomic(path, [{"source_id": "SRC-STC-0001"}])

    event_names = [name for name, _ in events]
    assert event_names.index("fchmod") < event_names.index("file-fsync")
    assert event_names.index("file-fsync") < event_names.index("replace")
    assert event_names.index("replace") < event_names.index("directory-fsync")
    assert ("fchmod", 0o644) in events


@pytest.mark.parametrize("writer", ["bulk", "append"])
def test_writer_rejects_symlink_registry_destination(tmp_path, writer):
    target = tmp_path / "target.jsonl"
    target.write_text('{"source_id":"SRC-STC-0001"}\n', encoding="utf-8")
    target_bytes = target.read_bytes()
    path = tmp_path / "sources.jsonl"
    path.symlink_to(target.name)

    with pytest.raises(
        jsonl_store.RegistryIntegrityError,
        match="regular file",
    ):
        if writer == "bulk":
            write_jsonl_atomic(
                path,
                [{"source_id": "SRC-STC-0002"}],
                expected_current_digest=hashlib.sha256(target_bytes).hexdigest(),
            )
        else:
            append_unique(
                path,
                {"source_id": "SRC-STC-0002"},
                "source_id",
            )

    assert path.is_symlink()
    assert target.read_bytes() == target_bytes
    lock_path = path.with_name(f"{path.name}.lock")
    assert lock_path.exists()
    assert lock_path.stat().st_size == 0


@pytest.mark.parametrize("writer", ["bulk", "append"])
@pytest.mark.parametrize("destination_kind", ["directory", "fifo"])
def test_writer_rejects_non_regular_registry_destination(
    tmp_path,
    writer,
    destination_kind,
):
    path = tmp_path / "sources.jsonl"
    if destination_kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)

    with pytest.raises(
        jsonl_store.RegistryIntegrityError,
        match="regular file",
    ):
        if writer == "bulk":
            write_jsonl_atomic(path, [{"source_id": "SRC-STC-0001"}])
        else:
            append_unique(
                path,
                {"source_id": "SRC-STC-0001"},
                "source_id",
            )

    if destination_kind == "directory":
        assert path.is_dir()
    else:
        assert stat.S_ISFIFO(path.lstat().st_mode)
    lock_path = path.with_name(f"{path.name}.lock")
    assert lock_path.exists()
    assert lock_path.stat().st_size == 0


def test_append_unique_rejects_duplicate(tmp_path):
    path = tmp_path / "sources.jsonl"
    append_unique(path, {"source_id": "SRC-STC-0001"}, "source_id")
    with pytest.raises(DuplicateIdError, match="SRC-STC-0001"):
        append_unique(path, {"source_id": "SRC-STC-0001"}, "source_id")


def test_append_unique_rewrites_in_canonical_order(tmp_path):
    path = tmp_path / "sources.jsonl"
    append_unique(path, {"source_id": "SRC-STC-0002"}, "source_id")
    append_unique(path, {"source_id": "SRC-STC-0001"}, "source_id")
    assert [row["source_id"] for row in read_jsonl(path, lambda row: row)] == [
        "SRC-STC-0001",
        "SRC-STC-0002",
    ]


def test_read_jsonl_applies_factory_to_each_record(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"value":1}\n\n{"value":2}\n', encoding="utf-8")
    assert read_jsonl(path, lambda row: row["value"] * 10) == [10, 20]


@pytest.mark.parametrize("payload", ["[1]\n", "null\n", '"scalar"\n', "1\n"])
def test_read_jsonl_rejects_non_object_before_factory_with_line_context(
    tmp_path,
    payload,
):
    path = tmp_path / "records.jsonl"
    path.write_text(payload, encoding="utf-8")
    factory_called = False

    def unexpected_factory(row):
        nonlocal factory_called
        factory_called = True
        return row

    with pytest.raises(TypeError) as caught:
        read_jsonl(path, unexpected_factory)

    assert str(caught.value) == f"{path}:1: JSONL record must be an object"
    assert factory_called is False


def test_read_jsonl_rejects_malformed_json_with_path_and_line_context(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"value":\n', encoding="utf-8")
    factory_called = False

    def unexpected_factory(row):
        nonlocal factory_called
        factory_called = True
        return row

    with pytest.raises(ValueError) as caught:
        read_jsonl(path, unexpected_factory)

    assert str(caught.value) == f"{path}:1: invalid JSON"
    assert factory_called is False


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (
            '{"source_id":"SRC-STC-0001","source_id":"SRC-STC-0002"}\n',
            "duplicate object key",
        ),
        ('{"value":NaN}\n', "non-finite JSON constant"),
        ('{"value":Infinity}\n', "non-finite JSON constant"),
        ('{"value":-Infinity}\n', "non-finite JSON constant"),
        ('{"value":1e999}\n', "non-finite JSON number"),
        ('{"outer":{"value":1,"value":2}}\n', "duplicate object key"),
    ],
)
def test_read_jsonl_rejects_duplicate_keys_and_nonfinite_constants(
    tmp_path,
    line,
    message,
):
    path = tmp_path / "records.jsonl"
    path.write_text(line, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_jsonl(path, lambda row: row)


@pytest.mark.parametrize(
    ("existing_payload", "message"),
    [
        ('{"title":"missing primary id"}\n', "preexisting row 1.*source_id"),
        (
            '{"source_id":"SRC-STC-0000"}\n',
            "preexisting row 1.*canonical source_id",
        ),
        (
            '{"source_id":"EV-STC-00001"}\n',
            "preexisting row 1.*source_id.*source",
        ),
        (
            '{"source_id":"SRC-STC-0001"}\n{"source_id":"SRC-STC-0001"}\n',
            "duplicate preexisting source_id",
        ),
    ],
)
def test_append_unique_validates_entire_existing_registry_before_mutation(
    tmp_path,
    existing_payload,
    message,
):
    path = tmp_path / "sources.jsonl"
    path.write_text(existing_payload, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        append_unique(
            path,
            {"source_id": "SRC-STC-0002"},
            "source_id",
        )
    assert path.read_text(encoding="utf-8") == existing_payload


def test_append_unique_rejects_noncanonical_new_primary_id_before_data_mutation(
    tmp_path,
):
    path = tmp_path / "sources.jsonl"
    with pytest.raises(ValueError, match="canonical source_id"):
        append_unique(
            path,
            {"source_id": "SRC-STC-０００１"},
            "source_id",
        )
    assert not path.exists()


def test_bulk_write_and_append_share_one_lock_without_lost_update(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "sources.jsonl"
    first = {"source_id": "SRC-STC-0001"}
    bulk_only = {"source_id": "SRC-STC-0002"}
    appended = {"source_id": "SRC-STC-0003"}
    write_jsonl_atomic(path, [first])
    expected_current_digest = hashlib.sha256(path.read_bytes()).hexdigest()

    bulk_at_replace = Event()
    release_bulk_replace = Event()
    append_finished = Event()
    real_replace = os.replace

    def controlled_replace(source, destination):
        if current_thread().name.startswith("bulk-writer"):
            bulk_at_replace.set()
            assert release_bulk_replace.wait(timeout=5)
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", controlled_replace)

    def append_worker():
        try:
            append_unique(path, appended, "source_id")
        finally:
            append_finished.set()

    with (
        ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="bulk-writer",
        ) as bulk_executor,
        ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="append-writer",
        ) as append_executor,
    ):
        bulk_future = bulk_executor.submit(
            write_jsonl_atomic,
            path,
            [first, bulk_only],
            expected_current_digest=expected_current_digest,
        )
        assert bulk_at_replace.wait(timeout=5)
        append_future = append_executor.submit(append_worker)
        append_was_blocked = not append_finished.wait(timeout=0.2)
        release_bulk_replace.set()
        bulk_future.result(timeout=5)
        append_future.result(timeout=5)

    assert append_was_blocked
    assert read_jsonl(path, lambda row: row) == [first, bulk_only, appended]


def test_bulk_change_requires_expected_current_digest(tmp_path):
    path = tmp_path / "sources.jsonl"
    first = {"source_id": "SRC-STC-0001"}
    write_jsonl_atomic(path, [first])
    original_bytes = path.read_bytes()

    with pytest.raises(RuntimeError, match="expected-current"):
        write_jsonl_atomic(
            path,
            [first, {"source_id": "SRC-STC-0002"}],
        )
    assert path.read_bytes() == original_bytes


def test_bulk_compare_and_swap_accepts_current_canonical_digest(tmp_path):
    path = tmp_path / "sources.jsonl"
    first = {"source_id": "SRC-STC-0001"}
    replacement = [
        first,
        {"source_id": "SRC-STC-0002"},
    ]
    write_jsonl_atomic(path, [first])
    expected_current_digest = jsonl_store.canonical_jsonl_digest([first])

    write_jsonl_atomic(
        path,
        replacement,
        expected_current_digest=expected_current_digest,
    )
    assert read_jsonl(path, lambda row: row) == replacement


def test_canonical_jsonl_digest_matches_file_bytes_and_detaches_input(tmp_path):
    path = tmp_path / "sources.jsonl"
    mutable_values = ["before"]
    records = [
        {
            "source_id": "SRC-STC-0002",
            "metadata": {"labels": ("one", "two")},
        },
        {
            "source_id": "SRC-STC-0001",
            "metadata": {"values": mutable_values},
        },
    ]
    digest = jsonl_store.canonical_jsonl_digest(records)
    assert records[0]["metadata"]["labels"] == ("one", "two")

    mutable_values.append("after")
    pre_mutation_records = [
        {
            "source_id": "SRC-STC-0002",
            "metadata": {"labels": ["one", "two"]},
        },
        {
            "source_id": "SRC-STC-0001",
            "metadata": {"values": ["before"]},
        },
    ]
    write_jsonl_atomic(path, pre_mutation_records)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_identical_bulk_write_is_a_noop_without_compare_and_swap(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "sources.jsonl"
    records = [{"source_id": "SRC-STC-0001"}]
    write_jsonl_atomic(path, records)

    def unexpected_replace(source, destination):
        raise AssertionError("identical write must not replace")

    monkeypatch.setattr(os, "replace", unexpected_replace)
    write_jsonl_atomic(path, records)


def test_stale_bulk_snapshot_after_append_is_rejected_and_preserved(tmp_path):
    path = tmp_path / "sources.jsonl"
    first = {"source_id": "SRC-STC-0001"}
    appended = {"source_id": "SRC-STC-0002"}
    stale_bulk = [
        first,
        {"source_id": "SRC-STC-0003"},
    ]
    write_jsonl_atomic(path, [first])
    stale_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    append_unique(path, appended, "source_id")
    committed_bytes = path.read_bytes()

    with pytest.raises(RuntimeError, match="stale expected-current"):
        write_jsonl_atomic(
            path,
            stale_bulk,
            expected_current_digest=stale_digest,
        )
    assert path.read_bytes() == committed_bytes
    assert read_jsonl(path, lambda row: row) == [first, appended]


def test_append_first_reverse_race_makes_stale_bulk_fail_closed(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "sources.jsonl"
    first = {"source_id": "SRC-STC-0001"}
    appended = {"source_id": "SRC-STC-0002"}
    bulk_only = {"source_id": "SRC-STC-0003"}
    write_jsonl_atomic(path, [first])
    stale_digest = hashlib.sha256(path.read_bytes()).hexdigest()

    append_at_replace = Event()
    release_append_replace = Event()
    bulk_finished = Event()
    real_replace = os.replace

    def controlled_replace(source, destination):
        if current_thread().name.startswith("append-first"):
            append_at_replace.set()
            assert release_append_replace.wait(timeout=5)
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", controlled_replace)

    def bulk_worker():
        try:
            write_jsonl_atomic(
                path,
                [first, bulk_only],
                expected_current_digest=stale_digest,
            )
        except jsonl_store.StaleWriteError as error:
            return error
        finally:
            bulk_finished.set()
        return None

    with (
        ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="append-first",
        ) as append_executor,
        ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="bulk-second",
        ) as bulk_executor,
    ):
        append_future = append_executor.submit(
            append_unique,
            path,
            appended,
            "source_id",
        )
        assert append_at_replace.wait(timeout=5)
        bulk_future = bulk_executor.submit(bulk_worker)
        bulk_was_blocked = not bulk_finished.wait(timeout=0.2)
        release_append_replace.set()
        append_future.result(timeout=5)
        bulk_error = bulk_future.result(timeout=5)

    assert bulk_was_blocked
    assert isinstance(bulk_error, jsonl_store.StaleWriteError)
    assert read_jsonl(path, lambda row: row) == [first, appended]


def test_write_reports_unknown_commit_outcome_and_exact_retry_is_idempotent(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "sources.jsonl"
    records = [{"source_id": "SRC-STC-0001"}]
    real_fsync_directory = jsonl_store._fsync_directory
    calls = 0

    def fail_once(directory):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected directory fsync failure")
        return real_fsync_directory(directory)

    monkeypatch.setattr(jsonl_store, "_fsync_directory", fail_once)
    with pytest.raises(OSError) as caught:
        write_jsonl_atomic(path, records)

    assert isinstance(caught.value, jsonl_store.CommitOutcomeUnknownError)
    assert caught.value.safe_to_retry is True
    assert path.read_text(encoding="utf-8") == ('{"source_id":"SRC-STC-0001"}\n')

    write_jsonl_atomic(
        path,
        records,
        retry_token=caught.value.retry_token,
    )
    assert read_jsonl(path, lambda row: row) == records


def test_write_unknown_outcome_retry_refuses_to_overwrite_newer_state(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "sources.jsonl"
    original = [{"source_id": "SRC-STC-0001"}]
    newer = [{"source_id": "SRC-STC-0002"}]
    real_fsync_directory = jsonl_store._fsync_directory
    calls = 0

    def fail_once(directory):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected directory fsync failure")
        return real_fsync_directory(directory)

    monkeypatch.setattr(jsonl_store, "_fsync_directory", fail_once)
    with pytest.raises(OSError) as caught:
        write_jsonl_atomic(path, original)
    assert isinstance(caught.value, jsonl_store.CommitOutcomeUnknownError)

    write_jsonl_atomic(
        path,
        newer,
        expected_current_digest=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    with pytest.raises(jsonl_store.RetryConflictError, match="newer state"):
        write_jsonl_atomic(
            path,
            original,
            retry_token=caught.value.retry_token,
        )
    assert read_jsonl(path, lambda row: row) == newer


def test_append_unknown_outcome_retry_is_idempotent_but_normal_duplicate_is_not(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "sources.jsonl"
    record = {"source_id": "SRC-STC-0001"}
    real_fsync_directory = jsonl_store._fsync_directory
    calls = 0

    def fail_once(directory):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected directory fsync failure")
        return real_fsync_directory(directory)

    monkeypatch.setattr(jsonl_store, "_fsync_directory", fail_once)
    with pytest.raises(OSError) as caught:
        append_unique(path, record, "source_id")

    assert isinstance(caught.value, jsonl_store.CommitOutcomeUnknownError)
    append_unique(
        path,
        record,
        "source_id",
        retry_token=caught.value.retry_token,
    )
    assert read_jsonl(path, lambda row: row) == [record]
    with pytest.raises(DuplicateIdError):
        append_unique(path, record, "source_id")


def test_append_retry_uses_canonical_tuple_list_identity(tmp_path, monkeypatch):
    path = tmp_path / "sources.jsonl"
    record = {
        "source_id": "SRC-STC-0001",
        "metadata": {"labels": ("one", "two")},
    }
    real_fsync_directory = jsonl_store._fsync_directory
    calls = 0

    def fail_once(directory):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected directory fsync failure")
        return real_fsync_directory(directory)

    monkeypatch.setattr(jsonl_store, "_fsync_directory", fail_once)
    with pytest.raises(jsonl_store.CommitOutcomeUnknownError) as caught:
        append_unique(path, record, "source_id")

    assert read_jsonl(path, lambda row: row)[0]["metadata"]["labels"] == [
        "one",
        "two",
    ]
    list_retry_record = {
        "source_id": "SRC-STC-0001",
        "metadata": {"labels": ["one", "two"]},
    }
    append_unique(
        path,
        list_retry_record,
        "source_id",
        retry_token=caught.value.retry_token,
    )


@pytest.mark.parametrize(
    ("committed_value", "retry_value"),
    [
        (1, 1.0),
        (True, 1),
        (-0.0, 0.0),
    ],
)
def test_append_retry_distinguishes_json_scalar_identity(
    tmp_path,
    monkeypatch,
    committed_value,
    retry_value,
):
    path = tmp_path / "sources.jsonl"
    committed = {
        "source_id": "SRC-STC-0001",
        "value": committed_value,
    }
    retry = {
        "source_id": "SRC-STC-0001",
        "value": retry_value,
    }
    real_fsync_directory = jsonl_store._fsync_directory
    calls = 0

    def fail_once(directory):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected directory fsync failure")
        return real_fsync_directory(directory)

    monkeypatch.setattr(jsonl_store, "_fsync_directory", fail_once)
    with pytest.raises(jsonl_store.CommitOutcomeUnknownError) as caught:
        append_unique(path, committed, "source_id")
    committed_bytes = path.read_bytes()

    with pytest.raises(jsonl_store.RetryConflictError, match="exact"):
        append_unique(
            path,
            retry,
            "source_id",
            retry_token=caught.value.retry_token,
        )
    assert path.read_bytes() == committed_bytes


@pytest.mark.parametrize(
    ("committed_value", "retry_value"),
    [
        (1, 1.0),
        (True, 1),
        (-0.0, 0.0),
    ],
)
def test_append_retry_duplicate_branch_compares_canonical_scalar_identity(
    tmp_path,
    committed_value,
    retry_value,
):
    path = tmp_path / "sources.jsonl"
    committed = {
        "source_id": "SRC-STC-0001",
        "value": committed_value,
    }
    retry = {
        "source_id": "SRC-STC-0001",
        "value": retry_value,
    }
    append_unique(path, committed, "source_id")
    retry_line = jsonl_store._canonical_snapshot_line(
        jsonl_store._record_snapshot(retry)
    )
    retry_token = jsonl_store._operation_retry_token(
        path,
        "append:source_id",
        retry_line,
    )

    with pytest.raises(
        jsonl_store.RetryConflictError,
        match="exact canonical record",
    ):
        append_unique(
            path,
            retry,
            "source_id",
            retry_token=retry_token,
        )


@pytest.mark.parametrize("writer", ["bulk", "append"])
@pytest.mark.parametrize(
    ("invalid_value", "error_type"),
    [
        ({1: "integer key"}, TypeError),
        ({"nested": {1: "integer key"}}, TypeError),
        (deque(["unsupported"]), TypeError),
        (object(), TypeError),
        (float("nan"), ValueError),
        (float("inf"), ValueError),
    ],
)
def test_raw_writer_inputs_require_deep_canonical_json(
    tmp_path,
    writer,
    invalid_value,
    error_type,
):
    path = tmp_path / "sources.jsonl"
    record = {
        "source_id": "SRC-STC-0001",
        "value": invalid_value,
    }
    with pytest.raises(error_type, match="canonical JSON|finite"):
        if writer == "bulk":
            write_jsonl_atomic(path, [record])
        else:
            append_unique(path, record, "source_id")
    assert not path.exists()


def test_append_snapshots_nested_input_before_waiting_for_lock(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "sources.jsonl"
    lock_path = path.with_name(f"{path.name}.lock")
    values = ["before"]
    record = {
        "source_id": "SRC-STC-0001",
        "metadata": {"values": values},
    }
    snapshot_ready = Event()
    real_retry_token = jsonl_store._operation_retry_token

    def signal_snapshot(*args, **kwargs):
        token = real_retry_token(*args, **kwargs)
        snapshot_ready.set()
        return token

    monkeypatch.setattr(
        jsonl_store,
        "_operation_retry_token",
        signal_snapshot,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        with lock_path.open("a+b") as lock_stream:
            flock(lock_stream.fileno(), LOCK_EX)
            try:
                future = executor.submit(
                    append_unique,
                    path,
                    record,
                    "source_id",
                )
                observed_snapshot = snapshot_ready.wait(timeout=5)
                if observed_snapshot:
                    values.append("after")
            finally:
                flock(lock_stream.fileno(), LOCK_UN)
        future.result(timeout=5)
    assert observed_snapshot

    assert read_jsonl(path, lambda row: row)[0]["metadata"]["values"] == ["before"]


def test_bulk_write_snapshots_nested_input_before_waiting_for_lock(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "sources.jsonl"
    lock_path = path.with_name(f"{path.name}.lock")
    values = ["before"]
    record = {
        "source_id": "SRC-STC-0001",
        "metadata": {"values": values},
    }
    snapshot_ready = Event()
    real_retry_token = jsonl_store._operation_retry_token

    def signal_snapshot(*args, **kwargs):
        token = real_retry_token(*args, **kwargs)
        snapshot_ready.set()
        return token

    monkeypatch.setattr(
        jsonl_store,
        "_operation_retry_token",
        signal_snapshot,
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        with lock_path.open("a+b") as lock_stream:
            flock(lock_stream.fileno(), LOCK_EX)
            try:
                future = executor.submit(
                    write_jsonl_atomic,
                    path,
                    [record],
                )
                observed_snapshot = snapshot_ready.wait(timeout=5)
                if observed_snapshot:
                    values.append("after")
            finally:
                flock(lock_stream.fileno(), LOCK_UN)
        future.result(timeout=5)
    assert observed_snapshot

    assert read_jsonl(path, lambda row: row)[0]["metadata"]["values"] == ["before"]


def test_public_writer_annotations_use_a_to_dict_protocol():
    assert hasattr(jsonl_store, "SupportsToDict")
    assert "JsonlRecord" in write_jsonl_atomic.__annotations__["records"]


def _slow_real_writer(monkeypatch):
    real_writer = jsonl_store._write_jsonl_atomic_unlocked

    def slow_writer(*args, **kwargs):
        time.sleep(0.05)
        return real_writer(*args, **kwargs)

    monkeypatch.setattr(
        jsonl_store,
        "_write_jsonl_atomic_unlocked",
        slow_writer,
    )


def test_concurrent_unique_appends_do_not_lose_updates(tmp_path, monkeypatch):
    _slow_real_writer(monkeypatch)
    path = tmp_path / "sources.jsonl"
    worker_count = 8
    barrier = Barrier(worker_count)

    def worker(number):
        barrier.wait()
        append_unique(
            path,
            {"source_id": f"SRC-STC-{number:04d}"},
            "source_id",
        )

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(worker, number) for number in range(1, worker_count + 1)
        ]
        for future in futures:
            future.result()

    assert len(read_jsonl(path, lambda row: row)) == worker_count
    lock_path = path.with_name(f"{path.name}.lock")
    assert lock_path.exists()
    assert lock_path.stat().st_size == 0
    assert not list(tmp_path.glob("*.tmp"))


def test_concurrent_duplicate_append_has_one_success(tmp_path, monkeypatch):
    _slow_real_writer(monkeypatch)
    path = tmp_path / "sources.jsonl"
    worker_count = 8
    barrier = Barrier(worker_count)
    outcome_lock = Lock()
    outcomes = []

    def worker():
        barrier.wait()
        try:
            append_unique(path, {"source_id": "SRC-STC-0001"}, "source_id")
        except DuplicateIdError:
            outcome = "duplicate"
        else:
            outcome = "success"
        with outcome_lock:
            outcomes.append(outcome)

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(worker) for _ in range(worker_count)]
        for future in futures:
            future.result()

    assert outcomes.count("success") == 1
    assert outcomes.count("duplicate") == worker_count - 1
    assert read_jsonl(path, lambda row: row) == [{"source_id": "SRC-STC-0001"}]


@pytest.mark.parametrize(
    ("cleanup_name", "active_name"),
    [
        ("a", "a.b"),
        ("a.b", "a.b.c"),
        ("a[bc]", "ab.x"),
        ("a?", "ab.y"),
        ("a*", "anything"),
    ],
)
def test_different_destination_writer_never_deletes_another_active_temp(
    tmp_path,
    monkeypatch,
    cleanup_name,
    active_name,
):
    cleanup_path = tmp_path / cleanup_name
    active_path = tmp_path / active_name
    active_at_fchmod = Event()
    release_active_fchmod = Event()
    active_temporary_paths = []
    real_fchmod = os.fchmod

    def pause_active_fchmod(descriptor, mode):
        if current_thread().name.startswith("active-destination"):
            descriptor_path = os.readlink(f"/proc/self/fd/{descriptor}")
            active_temporary_paths.append(Path(descriptor_path))
            active_at_fchmod.set()
            assert release_active_fchmod.wait(timeout=5)
        return real_fchmod(descriptor, mode)

    monkeypatch.setattr(os, "fchmod", pause_active_fchmod)
    with (
        ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="active-destination",
        ) as active_executor,
        ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="cleanup-destination",
        ) as cleanup_executor,
    ):
        active_future = active_executor.submit(
            write_jsonl_atomic,
            active_path,
            [{"source_id": "SRC-STC-0002"}],
        )
        assert active_at_fchmod.wait(timeout=5)
        try:
            cleanup_future = cleanup_executor.submit(
                write_jsonl_atomic,
                cleanup_path,
                [{"source_id": "SRC-STC-0001"}],
            )
            cleanup_future.result(timeout=5)
            assert active_temporary_paths[0].exists()
        finally:
            release_active_fchmod.set()
        active_future.result(timeout=5)

    assert read_jsonl(cleanup_path, lambda row: row) == [{"source_id": "SRC-STC-0001"}]
    assert read_jsonl(active_path, lambda row: row) == [{"source_id": "SRC-STC-0002"}]


def test_reserved_lock_name_cannot_split_a_live_destination_lock(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "a"
    lock_path = tmp_path / "a.lock"
    active_at_fchmod = Event()
    release_active_fchmod = Event()
    third_writer_finished = Event()
    real_fchmod = os.fchmod

    def pause_first_writer_fchmod(descriptor, mode):
        if current_thread().name.startswith("first-a-writer"):
            active_at_fchmod.set()
            assert release_active_fchmod.wait(timeout=5)
        return real_fchmod(descriptor, mode)

    def third_writer():
        try:
            append_unique(
                path,
                {"source_id": "SRC-STC-0003"},
                "source_id",
            )
        finally:
            third_writer_finished.set()

    monkeypatch.setattr(os, "fchmod", pause_first_writer_fchmod)
    with (
        ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="first-a-writer",
        ) as first_executor,
        ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="third-a-writer",
        ) as third_executor,
    ):
        first_future = first_executor.submit(
            write_jsonl_atomic,
            path,
            [{"source_id": "SRC-STC-0001"}],
        )
        assert active_at_fchmod.wait(timeout=5)
        lock_destination_error = None
        first_writer_error = None
        try:
            try:
                write_jsonl_atomic(
                    lock_path,
                    [{"source_id": "SRC-STC-0002"}],
                    expected_current_digest=hashlib.sha256(b"").hexdigest(),
                )
            except jsonl_store.RegistryIntegrityError as error:
                lock_destination_error = error
            third_future = third_executor.submit(third_writer)
            third_writer_was_blocked = not third_writer_finished.wait(timeout=0.2)
        finally:
            release_active_fchmod.set()
        try:
            first_future.result(timeout=5)
        except FileNotFoundError as error:
            first_writer_error = error
        third_future.result(timeout=5)

    assert isinstance(
        lock_destination_error,
        jsonl_store.RegistryIntegrityError,
    )
    assert third_writer_was_blocked
    assert first_writer_error is None
    assert read_jsonl(path, lambda row: row) == [
        {"source_id": "SRC-STC-0001"},
        {"source_id": "SRC-STC-0003"},
    ]
    assert lock_path.stat().st_size == 0
    assert not list(tmp_path.glob(".stc-jsonl-*.tmp"))


@pytest.mark.parametrize("writer", ["bulk", "append"])
@pytest.mark.parametrize(
    "reserved_name",
    [
        "a.lock",
        "a.b.lock",
        f".stc-jsonl-{'a' * 64}.active.tmp",
        ".stc-jsonl-arbitrary.jsonl",
    ],
)
def test_public_writers_reject_reserved_sidecar_destination_names_without_artifacts(
    tmp_path,
    writer,
    reserved_name,
):
    path = tmp_path / reserved_name

    with pytest.raises(
        jsonl_store.RegistryIntegrityError,
        match="reserved sidecar namespace",
    ):
        if writer == "bulk":
            write_jsonl_atomic(path, [{"source_id": "SRC-STC-0001"}])
        else:
            append_unique(
                path,
                {"source_id": "SRC-STC-0001"},
                "source_id",
            )

    assert list(tmp_path.iterdir()) == []


def test_reserved_bulk_destination_rejects_before_parent_or_input_consumption(
    tmp_path,
):
    parent = tmp_path / "missing"
    path = parent / "sources.jsonl.lock"
    consumed = False

    def records():
        nonlocal consumed
        consumed = True
        yield {"source_id": "SRC-STC-0001"}

    with pytest.raises(
        jsonl_store.RegistryIntegrityError,
        match="reserved sidecar namespace",
    ):
        write_jsonl_atomic(path, records())

    assert consumed is False
    assert not parent.exists()


def test_reserved_append_destination_rejects_before_parent_or_to_dict_call(
    tmp_path,
):
    parent = tmp_path / "missing"
    path = parent / ".stc-jsonl-arbitrary.jsonl"

    class ObservedRecord:
        to_dict_called = False

        def to_dict(self):
            self.to_dict_called = True
            return {"source_id": "SRC-STC-0001"}

    record = ObservedRecord()
    with pytest.raises(
        jsonl_store.RegistryIntegrityError,
        match="reserved sidecar namespace",
    ):
        append_unique(path, record, "source_id")

    assert record.to_dict_called is False
    assert not parent.exists()


def test_destination_lock_defensively_rejects_reserved_name_before_open(tmp_path):
    path = tmp_path / "a.lock"

    with (
        pytest.raises(
            jsonl_store.RegistryIntegrityError,
            match="reserved sidecar namespace",
        ),
        jsonl_store._destination_lock(path),
    ):
        pytest.fail("reserved destination lock must not yield")

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("writer", ["bulk", "append"])
@pytest.mark.parametrize(
    "reserved_parent_name",
    ["a.lock", ".stc-jsonl-arbitrary"],
)
def test_reserved_missing_parent_component_rejects_before_any_side_effect(
    tmp_path,
    writer,
    reserved_parent_name,
):
    missing_root = tmp_path / "missing"
    path = missing_root / reserved_parent_name / "sources.jsonl"
    input_touched = False

    def records():
        nonlocal input_touched
        input_touched = True
        yield {"source_id": "SRC-STC-0001"}

    class ObservedRecord:
        def to_dict(self):
            nonlocal input_touched
            input_touched = True
            return {"source_id": "SRC-STC-0001"}

    with pytest.raises(
        jsonl_store.RegistryIntegrityError,
        match="reserved sidecar namespace",
    ):
        if writer == "bulk":
            write_jsonl_atomic(path, records())
        else:
            append_unique(path, ObservedRecord(), "source_id")

    assert input_touched is False
    assert not missing_root.exists()


def test_dangling_lock_symlink_is_rejected_without_creating_its_target(
    tmp_path,
):
    path = tmp_path / "sources.jsonl"
    outside = tmp_path / "outside"
    outside.mkdir()
    dangling_target = outside / "missing-lock-target"
    lock_path = path.with_name(f"{path.name}.lock")
    lock_path.symlink_to(dangling_target)

    with pytest.raises(
        jsonl_store.RegistryIntegrityError,
        match="coordination lock.*zero-byte regular file",
    ):
        write_jsonl_atomic(path, [{"source_id": "SRC-STC-0001"}])

    assert lock_path.is_symlink()
    assert not dangling_target.exists()
    assert not path.exists()


@pytest.mark.parametrize("lock_kind", ["directory", "fifo", "nonzero"])
def test_unsafe_preexisting_lock_sidecar_is_rejected(tmp_path, lock_kind):
    path = tmp_path / "sources.jsonl"
    lock_path = path.with_name(f"{path.name}.lock")
    if lock_kind == "directory":
        lock_path.mkdir()
    elif lock_kind == "fifo":
        os.mkfifo(lock_path)
    else:
        lock_path.write_bytes(b"not a lock")

    with pytest.raises(
        jsonl_store.RegistryIntegrityError,
        match="coordination lock.*zero-byte regular file",
    ):
        append_unique(
            path,
            {"source_id": "SRC-STC-0001"},
            "source_id",
        )

    assert not path.exists()


def test_normal_coordination_lock_is_private_regular_and_zero_byte(tmp_path):
    path = tmp_path / "sources.jsonl"
    append_unique(
        path,
        {"source_id": "SRC-STC-0001"},
        "source_id",
    )

    lock_path = path.with_name(f"{path.name}.lock")
    lock_stat = lock_path.lstat()
    assert stat.S_ISREG(lock_stat.st_mode)
    assert stat.S_IMODE(lock_stat.st_mode) == 0o600
    assert lock_stat.st_size == 0


def test_coordination_lock_open_uses_linux_nofollow_and_cloexec(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "sources.jsonl"
    lock_path = path.with_name(f"{path.name}.lock")
    observed_open = []
    real_open = os.open

    def capture_open(open_path, flags, mode=0o777):
        if Path(open_path) == lock_path:
            observed_open.append((flags, mode))
        return real_open(open_path, flags, mode)

    monkeypatch.setattr(os, "open", capture_open)
    append_unique(
        path,
        {"source_id": "SRC-STC-0001"},
        "source_id",
    )

    assert len(observed_open) == 1
    flags, mode = observed_open[0]
    assert flags & os.O_CREAT
    assert flags & os.O_RDWR
    assert flags & os.O_CLOEXEC
    assert flags & os.O_NOFOLLOW
    assert mode == 0o600


def test_coordination_lock_descriptor_closes_on_post_open_validation_error(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "sources.jsonl"
    lock_path = path.with_name(f"{path.name}.lock")
    opened_lock_descriptors = []
    real_open = os.open
    real_fstat = os.fstat

    class UnsafeLockState:
        st_mode = stat.S_IFREG | 0o600
        st_size = 1

    def capture_open(open_path, flags, mode=0o777):
        descriptor = real_open(open_path, flags, mode)
        if Path(open_path) == lock_path:
            opened_lock_descriptors.append(descriptor)
        return descriptor

    def unsafe_fstat(descriptor):
        if descriptor in opened_lock_descriptors:
            return UnsafeLockState()
        return real_fstat(descriptor)

    monkeypatch.setattr(os, "open", capture_open)
    monkeypatch.setattr(os, "fstat", unsafe_fstat)
    with pytest.raises(
        jsonl_store.RegistryIntegrityError,
        match="coordination lock.*zero-byte regular file",
    ):
        write_jsonl_atomic(path, [{"source_id": "SRC-STC-0001"}])

    assert len(opened_lock_descriptors) == 1
    with pytest.raises(OSError):
        real_fstat(opened_lock_descriptors[0])


@pytest.mark.parametrize("writer", ["bulk", "append"])
@pytest.mark.parametrize("ordinary_name", ["a", "a.b", "sources.jsonl"])
def test_public_writers_allow_ordinary_destination_names(
    tmp_path,
    writer,
    ordinary_name,
):
    path = tmp_path / ordinary_name
    if writer == "bulk":
        write_jsonl_atomic(path, [{"source_id": "SRC-STC-0001"}])
    else:
        append_unique(
            path,
            {"source_id": "SRC-STC-0001"},
            "source_id",
        )

    assert read_jsonl(path, lambda row: row) == [{"source_id": "SRC-STC-0001"}]


def test_temporary_prefix_is_deterministic_path_digest_namespace(tmp_path):
    destinations = [tmp_path / name for name in ("a", "a.b", "a.b.c", "a[bc]*?.jsonl")]
    prefixes = [
        jsonl_store._temporary_prefix(destination) for destination in destinations
    ]

    assert len(prefixes) == len(set(prefixes))
    for destination, prefix in zip(destinations, prefixes, strict=True):
        canonical_path = os.path.abspath(os.fspath(destination))
        expected_digest = hashlib.sha256(os.fsencode(canonical_path)).hexdigest()
        assert prefix == f".stc-jsonl-{expected_digest}."
        assert Path(prefix).name == prefix
        assert jsonl_store._temporary_prefix(destination) == prefix


def test_cleanup_only_removes_exact_destination_temp_namespace(tmp_path):
    destinations = [tmp_path / name for name in ("a", "a.b", "a.b.c", "a[bc]*?.jsonl")]
    temporary_paths = {
        destination: tmp_path
        / f"{jsonl_store._temporary_prefix(destination)}orphan.tmp"
        for destination in destinations
    }
    for temporary_path in temporary_paths.values():
        temporary_path.write_text("orphan", encoding="utf-8")

    for index, destination in enumerate(destinations):
        jsonl_store._cleanup_stale_temporary_files(destination)
        assert not temporary_paths[destination].exists()
        for untouched in destinations[index + 1 :]:
            assert temporary_paths[untouched].exists()


@pytest.mark.parametrize("name", ["a", "a.b", "a.b.c", "a[bc]*?.jsonl"])
def test_writer_temp_namespace_stays_in_destination_directory(
    tmp_path,
    monkeypatch,
    name,
):
    destination = tmp_path / name
    observed_sources = []
    real_replace = os.replace

    def capture_replace(source, target):
        observed_sources.append(Path(source))
        return real_replace(source, target)

    monkeypatch.setattr(os, "replace", capture_replace)
    write_jsonl_atomic(
        destination,
        [{"source_id": "SRC-STC-0001"}],
    )

    assert len(observed_sources) == 1
    assert observed_sources[0].parent == destination.parent
    assert observed_sources[0].name.startswith(
        jsonl_store._temporary_prefix(destination)
    )


def test_linux_process_unique_appends_preserve_every_record(tmp_path):
    context = multiprocessing.get_context("fork")
    path = tmp_path / "sources.jsonl"
    worker_count = 6
    barrier = context.Barrier(worker_count)
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=_process_append_worker,
            args=(str(path), number, barrier, outcomes, False),
        )
        for number in range(1, worker_count + 1)
    ]
    for process in processes:
        process.start()
    _join_processes_or_fail(processes)

    assert all(process.exitcode == 0 for process in processes)
    results = [outcomes.get(timeout=2) for _ in processes]
    assert [result[0] for result in results].count("success") == worker_count
    assert {row["source_id"] for row in read_jsonl(path, lambda row: row)} == {
        f"SRC-STC-{number:04d}" for number in range(1, worker_count + 1)
    }


def test_linux_process_duplicate_race_has_exactly_one_success(tmp_path):
    context = multiprocessing.get_context("fork")
    path = tmp_path / "sources.jsonl"
    worker_count = 6
    barrier = context.Barrier(worker_count)
    outcomes = context.Queue()
    processes = [
        context.Process(
            target=_process_append_worker,
            args=(str(path), number, barrier, outcomes, True),
        )
        for number in range(1, worker_count + 1)
    ]
    for process in processes:
        process.start()
    _join_processes_or_fail(processes)

    assert all(process.exitcode == 0 for process in processes)
    results = [outcomes.get(timeout=2)[0] for _ in processes]
    assert results.count("success") == 1
    assert results.count("duplicate") == worker_count - 1
    assert read_jsonl(path, lambda row: row) == [{"source_id": "SRC-STC-0001"}]


def test_linux_killed_lock_holder_recovers_and_cleans_orphan_temp(tmp_path):
    context = multiprocessing.get_context("fork")
    path = tmp_path / "sources.jsonl"
    ready = context.Event()
    holder = context.Process(
        target=_killed_lock_holder,
        args=(str(path), ready),
    )
    holder.start()
    assert ready.wait(timeout=5)
    os.kill(holder.pid, signal.SIGKILL)
    holder.join(timeout=5)
    if holder.is_alive():
        holder.kill()
        holder.join(timeout=2)
        pytest.fail("killed lock holder did not exit")
    assert holder.exitcode == -signal.SIGKILL
    temporary_pattern = f"{jsonl_store._temporary_prefix(path)}*.tmp"
    assert list(tmp_path.glob(temporary_pattern))

    append_unique(
        path,
        {"source_id": "SRC-STC-0001"},
        "source_id",
    )
    assert read_jsonl(path, lambda row: row) == [{"source_id": "SRC-STC-0001"}]
    assert not list(tmp_path.glob(temporary_pattern))
