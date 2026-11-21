"""Initial red test suite for the GPU ring command-buffer exercise.

Exactly fifteen tests call the public stubs in ``ring.buffer``. Every test
expresses one user-visible contract and must fail with ``NotImplementedError``
in the initial repository state, without collection or import errors.
"""

import copy
import time

import pytest

from ring.buffer import (
    RingError,
    alloc_block,
    fence_tick,
    plan_frame,
    reclaim,
    ring_batch,
    validate,
    wait_fence,
    wrap_check,
)


def make_ring(capacity=64, alignment=8):
    """Build the documented plain-dict ring shape."""
    return {
        "capacity": capacity,
        "alignment": alignment,
        "cursor": 0,
        "used": 0,
        "frames": {},
        "fences": {},
    }


def command(bytes_, frame=1):
    """Return a command with exactly the documented fields."""
    return {"bytes": bytes_, "frame": frame}


def frame_record(offset, size, committed=True):
    """Return one frame record with a single allocated block."""
    return {
        "blocks": [{"offset": offset, "size": size}],
        "committed": committed,
    }


# ---------------------------------------------------------------------------
# One regression test per documented defect.
# ---------------------------------------------------------------------------


def test_alloc_aligns_offsets_and_reserves_block_size():
    # Defect 1: allocating by raw byte count places the next command at an
    # unaligned offset and lets a write spill past its aligned block.
    ring = make_ring(64)
    first = alloc_block(ring, 5, 8)
    second = alloc_block(ring, 3, 8)

    assert first == {"offset": 0, "wrapped": False}
    assert second == {"offset": 8, "wrapped": False}
    assert ring["cursor"] == 16
    assert ring["used"] == 16


def test_alloc_wraps_with_tail_padding():
    # Defect 2: a command crossing the tail must be padded and moved whole to
    # offset 0, never split into two pieces.
    ring = make_ring(16)
    ring["cursor"] = 12
    ring["used"] = 0

    result = alloc_block(ring, 6, 4)

    assert result == {"offset": 0, "wrapped": True}
    assert ring["cursor"] == 8
    assert ring["used"] == 8
    assert wrap_check(12, 8, 16) is True


def test_reclaim_uses_frame_number_not_fence_tick():
    # Defect 3: reclaiming by tick releases a different frame than requested,
    # sometimes before its work is complete.
    ring = make_ring(64)
    ring["cursor"] = 24
    ring["used"] = 24
    ring["frames"] = {
        1: frame_record(0, 8),
        2: frame_record(8, 8),
        3: frame_record(16, 8),
    }
    ring["fences"] = {1: 100, 2: 200, 3: 300}

    result = reclaim(ring, 2)

    assert result == {"reclaimed": [1, 2], "kept": [3], "freed_bytes": 16}
    assert sorted(ring["frames"]) == [3]
    assert ring["used"] == 8


def test_reclaim_skips_uncommitted_frames():
    # Defect 4: reclaiming an uncommitted frame exposes half-written data.
    ring = make_ring(64)
    ring["cursor"] = 16
    ring["used"] = 16
    ring["frames"] = {
        1: frame_record(0, 8, committed=True),
        2: frame_record(8, 8, committed=False),
    }

    result = reclaim(ring, 2)

    assert result == {"reclaimed": [1], "kept": [2], "freed_bytes": 8}
    assert sorted(ring["frames"]) == [2]
    assert ring["used"] == 8
    assert ring["frames"][2]["committed"] is False


def test_plan_frame_honours_byte_budget():
    # Defect 5: splitting by a command count lets an 8+8 frame exceed a
    # 12-byte budget. Greedy byte packing must yield [8] then [8, 4].
    commands = [command(8, 1), command(8, 1), command(4, 1)]
    plan = plan_frame(commands, 12)

    assert plan == [[commands[0]], [commands[1], commands[2]]]
    assert [sum(item["bytes"] for item in group) for group in plan] == [8, 12]
    assert all(
        sum(item["bytes"] for item in group) <= 12 for group in plan
    )


def test_fence_tick_rejects_backwards_tick():
    # Defect 6: a decreasing fence tick makes later waits succeed too early.
    ring = make_ring(64)
    ring["fences"] = {1: 10}

    with pytest.raises(RingError):
        fence_tick(ring, 1, 9)

    assert ring["fences"] == {1: 10}
    assert fence_tick(ring, 1, 10) == {"frame": 1, "tick": 10}
    assert fence_tick(ring, 2, 5) == {"frame": 2, "tick": 5}
    assert ring["fences"] == {1: 10, 2: 5}


def test_wait_fence_is_nonblocking_before_target():
    # Defect 7: a busy-wait pins a CPU core instead of returning False once.
    ring = make_ring(64)
    ring["fences"] = {2: 50}
    before = copy.deepcopy(ring)

    started = time.monotonic()
    ready = wait_fence(ring, 2, 10)
    elapsed = time.monotonic() - started

    assert ready is False
    assert elapsed < 0.2
    assert ring == before
    assert wait_fence(ring, 2, 49) is True   # one-tick tolerance
    assert wait_fence(ring, 2, 50) is True
    assert wait_fence(ring, 3, 50) is False


def test_validate_reports_non_power_of_two_alignment():
    # Defect 8: an alignment of 3 is not a divisor-friendly power of two and
    # must be reported instead of silently used or divided by.
    ring = make_ring(64, alignment=3)
    issues = validate(ring, [command(8, 1)])

    assert issues
    assert any("alignment" in issue.lower() for issue in issues)


def test_batch_failure_rolls_back_every_frame():
    # Defect 9: frames sharing partial ring state leave frame 1 allocated when
    # frame 2 is rejected; the whole transaction must roll back.
    ring = make_ring(24)
    frames = [
        {"frame": 1, "commands": [command(8, 1)]},
        {"frame": 2, "commands": [command(20, 2)]},  # over the 16-byte budget
    ]
    config = {"ring": ring, "alignment": 8, "budget": 16}
    snapshot = copy.deepcopy(ring)

    with pytest.raises(RingError):
        ring_batch(frames, config)

    assert ring == snapshot
    assert ring["frames"] == {}
    assert ring["fences"] == {}
    assert ring["cursor"] == 0
    assert ring["used"] == 0


def test_alloc_rejects_zero_capacity():
    # Defect 10: capacity 0 must be rejected up front, not used in arithmetic
    # that can divide by zero or wrap into no space.
    ring = make_ring(0)

    with pytest.raises(RingError):
        alloc_block(ring, 1, 1)

    assert ring["cursor"] == 0
    assert ring["used"] == 0


# ---------------------------------------------------------------------------
# Five additional contract tests.
# ---------------------------------------------------------------------------


def test_single_command_fitting_tail_does_not_wrap():
    # A command that fits before the end must stay put; wrapping is only for
    # actual tail crossings.
    ring = make_ring(64)
    ring["cursor"] = 32
    ring["used"] = 16

    result = alloc_block(ring, 16, 8)

    assert result == {"offset": 32, "wrapped": False}
    assert ring["cursor"] == 48
    assert ring["used"] == 32


def test_alloc_succeeds_when_capacity_is_exact():
    # A block that ends exactly at capacity is not over the limit and needs no
    # padding; the next allocation then fails cleanly.
    ring = make_ring(64)

    result = alloc_block(ring, 64, 8)

    assert result == {"offset": 0, "wrapped": False}
    assert ring["cursor"] == 64
    assert ring["used"] == 64
    with pytest.raises(RingError):
        alloc_block(ring, 1, 8)


def test_alloc_rejects_alignment_three():
    # alignment=3 is a contract violation, not a request to round.
    ring = make_ring(64)
    snapshot = copy.deepcopy(ring)

    with pytest.raises(RingError):
        alloc_block(ring, 8, 3)

    assert ring == snapshot


def test_plan_frame_empty_commands_returns_empty_plan():
    assert plan_frame([], 16) == []


def test_batch_success_preserves_frame_order():
    # All frames succeed; input frame order, command order, and ticks must be
    # preserved, with independent per-frame block lists.
    ring = make_ring(128)
    frames = [
        {"frame": 3, "commands": [command(8, 3), command(8, 3)]},
        {"frame": 1, "commands": [command(8, 1)]},
        {"frame": 2, "commands": [command(8, 2)]},
    ]
    config = {
        "ring": ring,
        "alignment": 8,
        "budget": 16,
        "ticks": {3: 30, 1: 10, 2: 20},
    }

    result = ring_batch(frames, config)
    entries = result["frames"]

    assert [entry["frame"] for entry in entries] == [3, 1, 2]
    assert [entry["bytes"] for entry in entries] == [16, 8, 8]
    assert [
        [block["offset"] for block in entry["blocks"]] for entry in entries
    ] == [[0, 8], [16], [24]]
    assert [entry["tick"] for entry in entries] == [30, 10, 20]
    assert result["committed"] == [3, 1, 2]
    assert result["ring"] is ring
    assert ring["fences"] == {3: 30, 1: 10, 2: 20}
    assert ring["used"] == 32
    assert len({id(entry["blocks"]) for entry in entries}) == 3
