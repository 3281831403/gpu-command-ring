"""Ring command-buffer bugfix exercise.

The public ring state is a plain mutable ``dict`` so tests can build fixtures
without any helper code from this module::

    {
        "capacity": int,   # bytes in the ring, must be > 0
        "alignment": int,  # default block alignment, a positive power of two
        "cursor": int,     # next physical byte offset, 0 <= cursor <= capacity
        "used": int,       # live reserved bytes; alignment padding excluded
        "frames": {
            frame: {
                "blocks": [{"offset": int, "size": int}, ...],
                "committed": bool,
            },
            ...
        },
        "fences": {frame: tick, ...},
    }

A command is exactly ``{"bytes": int, "frame": int}``. All sizes and offsets
are byte counts. A **block size** is the command's byte count rounded up to the
requested power-of-two alignment. A **frame budget** is the maximum number of
command bytes that may be placed in one planned frame. A **fence tick** is an
explicit non-negative integer supplied by the caller; no clock or random source
is involved, and fence waits tolerate one tick.
"""

import copy
from collections.abc import Mapping, Sequence


class RingError(Exception):
    """Raised for an invalid ring, command, frame, alignment, or fence tick.

    Public operations validate their inputs before mutating the ring and use
    this exception for contract violations. A transaction that raises must
    leave the ring in its pre-call state.
    """


def _is_int(value) -> bool:
    """Return whether *value* is an integer, excluding ``bool``."""
    return isinstance(value, int) and not isinstance(value, bool)


def _check_non_negative(value, name: str) -> int:
    if not _is_int(value) or value < 0:
        raise RingError(f"{name} must be a non-negative integer")
    return value


def _check_alignment(alignment) -> int:
    """Return *alignment* if it is a positive power of two."""
    if not _is_int(alignment) or alignment <= 0:
        raise RingError("alignment must be a positive power of two")
    if alignment & (alignment - 1):
        raise RingError("alignment must be a positive power of two")
    return alignment


def _block_size(bytes_: int, alignment: int) -> int:
    """Round *bytes_* up to the next multiple of *alignment*."""
    return (bytes_ + alignment - 1) // alignment * alignment


def _check_capacity(capacity) -> int:
    if not _is_int(capacity) or capacity <= 0:
        raise RingError("capacity must be a positive integer")
    return capacity


def _validate_frames(frames) -> dict:
    """Validate the ``frames`` table and return it unchanged."""
    if not isinstance(frames, Mapping):
        raise RingError("frames must be a mapping")
    for number, record in frames.items():
        if not _is_int(number) or number < 0:
            raise RingError("frame numbers must be non-negative integers")
        if not isinstance(record, Mapping):
            raise RingError("frame records must be mappings")
        blocks = record.get("blocks")
        if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
            raise RingError("frame blocks must be a sequence")
        for block in blocks:
            if not isinstance(block, Mapping):
                raise RingError("blocks must be mappings")
            _check_non_negative(block.get("offset"), "block offset")
            _check_non_negative(block.get("size"), "block size")
        if not isinstance(record.get("committed"), bool):
            raise RingError("frame committed flag must be a bool")
    return frames


def _validate_state(ring: Mapping) -> tuple[int, int, int, dict]:
    """Validate capacity/cursor/used and the frames table."""
    if not isinstance(ring, Mapping):
        raise RingError("ring must be a mapping")
    for key in ("capacity", "alignment", "cursor", "used", "frames"):
        if key not in ring:
            raise RingError(f"ring is missing '{key}'")
    capacity = _check_capacity(ring["capacity"])
    alignment = _check_alignment(ring["alignment"])
    cursor = _check_non_negative(ring["cursor"], "cursor")
    used = _check_non_negative(ring["used"], "used")
    if cursor > capacity:
        raise RingError("cursor is out of range")
    if used > capacity:
        raise RingError("used is out of range")
    return capacity, alignment, cursor, _validate_frames(ring["frames"])


def _validate_fences(ring: Mapping) -> dict:
    """Validate the ``fences`` table and return it unchanged."""
    if not isinstance(ring, Mapping) or "fences" not in ring:
        raise RingError("ring is missing 'fences'")
    fences = ring["fences"]
    if not isinstance(fences, Mapping):
        raise RingError("fences must be a mapping")
    for frame, tick in fences.items():
        if not _is_int(frame) or frame < 0:
            raise RingError("fence frames must be non-negative integers")
        if not _is_int(tick) or tick < 0:
            raise RingError("fence ticks must be non-negative integers")
    return fences


def _check_command(command: Mapping) -> tuple[int, int]:
    """Validate one command mapping and return ``(bytes, frame)``."""
    if not isinstance(command, Mapping):
        raise RingError("commands must be mappings")
    if "bytes" not in command or "frame" not in command:
        raise RingError("commands need 'bytes' and 'frame'")
    bytes_ = _check_non_negative(command["bytes"], "command bytes")
    frame = _check_non_negative(command["frame"], "command frame")
    return bytes_, frame


def _reserve(
    capacity: int, cursor: int, used: int, block: int
) -> tuple[int, int, bool]:
    """Place *block* without splitting it; return ``(offset, cursor, wrapped)``.

    A block that would cross the tail consumes ``capacity - cursor`` bytes of
    padding and moves whole to offset 0. Free space must cover both the block
    and that padding, otherwise :class:`RingError` is raised.
    """
    tail = capacity - cursor
    if block <= tail:
        needed = block
        placement = (cursor, cursor + block, False)
    else:
        # The block crosses the tail: pad to the end and move it whole to 0.
        if block > capacity:
            raise RingError("block does not fit in the ring")
        needed = tail + block
        placement = (0, block, True)
    if needed > capacity - used:
        raise RingError("ring has no free space for the reservation")
    return placement


def alloc_block(ring: dict, bytes_: int, alignment: int) -> dict:
    """Reserve one aligned block in *ring* and return its placement.

    ``ring`` follows the module-level dict shape. ``bytes_`` is a non-negative
    byte count and ``alignment`` must be a positive power of two (1, 2, 4, 8,
    ...). The block size is ``ceil(bytes_ / alignment) * alignment``; that many
    bytes are reserved even though the command only needs ``bytes_``. Invalid
    values -- including ``alignment`` 0 or 3, a negative size, ``capacity <= 0``,
    or an out-of-range cursor/used count -- raise :class:`RingError` without
    changing the ring.

    If the block fits from ``ring["cursor"]`` to ``capacity``, place it there
    and return ``{"offset": cursor, "wrapped": False}``. Otherwise the block
    must not be split across the end: consume ``capacity - cursor`` bytes of
    tail padding, place the whole block at offset 0, advance the cursor to the
    block size, and return ``{"offset": 0, "wrapped": True}``. A block that is
    exactly as large as the remaining tail does not wrap. On either path
    ``used`` grows by the block size (padding is not counted), and a zero-byte
    command reserves nothing, returns a zero-size placement, and does not move
    the cursor.

    The caller must leave enough free bytes for the reservation and any wrap
    padding; if the ring cannot hold it, raise :class:`RingError` before
    mutating state. The result has exactly the keys ``offset`` and ``wrapped``.
    """
    capacity, _, cursor, _ = _validate_state(ring)
    bytes_ = _check_non_negative(bytes_, "bytes")
    alignment = _check_alignment(alignment)

    if bytes_ == 0:
        return {"offset": cursor, "wrapped": False}

    block = _block_size(bytes_, alignment)
    if block > capacity:
        raise RingError("block does not fit in the ring")

    offset, new_cursor, wrapped = _reserve(
        capacity, cursor, ring["used"], block
    )

    ring["cursor"] = new_cursor
    ring["used"] += block
    return {"offset": offset, "wrapped": wrapped}


def wrap_check(offset: int, size: int, capacity: int) -> bool:
    """Return whether a block would cross the end of the ring.

    ``offset`` and ``size`` are byte counts describing ``[offset, offset+size)``
    and ``capacity`` is the ring size in bytes. Return ``True`` exactly when
    ``offset + size > capacity``; such a command would be cut into two halves
    unless the allocator inserts padding and wraps it to offset 0. A block that
    ends exactly at ``capacity`` does not cross the tail. Zero-size blocks
    never wrap.

    Negative offsets or sizes and a non-positive capacity are invalid and raise
    :class:`RingError`. This function is a pure predicate; it never mutates any
    ring and never adds padding itself.
    """
    offset = _check_non_negative(offset, "offset")
    size = _check_non_negative(size, "size")
    capacity = _check_capacity(capacity)
    return offset + size > capacity


def reclaim(ring: dict, frame: int) -> dict:
    """Release committed frames up to and including *frame*.

    A frame is reclaimed by its frame number, never by its fence tick: every
    frame in ``ring["frames"]`` whose number is ``<= frame`` and whose
    ``"committed"`` flag is true is removed, even if its fence tick is larger
    than *frame*. Frames numbered above *frame* are left alone. An uncommitted
    frame is never reclaimed, so a reader can never observe a half-written
    frame; it stays in the ring and keeps its blocks.

    Removing a frame subtracts the sizes of its blocks from ``ring["used"]``
    and removes it from ``ring["frames"]``. The allocation cursor is not
    rewound. ``frame`` must be a non-negative integer; otherwise raise
    :class:`RingError` before changing anything.

    Return exactly::

        {
            "reclaimed": [frame numbers, ascending],
            "kept": [frame numbers still present, ascending],
            "freed_bytes": int,
        }
    """
    capacity, _, _, frames = _validate_state(ring)
    frame = _check_non_negative(frame, "frame")

    reclaimed = []
    freed = 0
    for number in sorted(frames):
        record = frames[number]
        if number <= frame and record["committed"]:
            reclaimed.append(number)
            freed += sum(block["size"] for block in record["blocks"])
    if freed > ring["used"]:
        raise RingError("frame block sizes exceed the used byte count")

    for number in reclaimed:
        del ring["frames"][number]
    ring["used"] -= freed

    kept = sorted(ring["frames"])
    return {"reclaimed": reclaimed, "kept": kept, "freed_bytes": freed}


def plan_frame(commands: Sequence[Mapping], budget: int) -> list[list]:
    """Pack *commands* into plan frames without exceeding *budget* bytes.

    ``commands`` is an ordered sequence of command mappings with exactly the
    keys ``{"bytes", "frame"}``. ``budget`` is the per-frame byte budget and
    must be a positive integer. Commands are visited in input order and packed
    greedily: add a command to the current plan frame while the sum of its
    command bytes would stay at or below ``budget``; otherwise start the next
    plan frame with that command. A single command whose ``bytes`` exceeds the
    budget cannot be placed and raises :class:`RingError`.

    The returned list contains lists of the original command objects in input
    order, so no command is split or duplicated; each list's byte sum is
    ``<= budget``. The command's ``"frame"`` field is retained untouched (it is
    metadata for validation, not a reason to exceed the byte budget). An empty
    command sequence returns ``[]``. Invalid commands or budgets raise
    :class:`RingError` without mutating the input.
    """
    if not _is_int(budget) or budget <= 0:
        raise RingError("budget must be a positive integer")
    if isinstance(commands, (str, bytes)) or not isinstance(
        commands, Sequence
    ):
        raise RingError("commands must be a sequence")

    checked = [_check_command(command) for command in commands]

    plan: list[list] = []
    current: list = []
    current_bytes = 0
    for command, (bytes_, _) in zip(commands, checked):
        if bytes_ > budget:
            raise RingError("command exceeds the frame budget")
        if current and current_bytes + bytes_ > budget:
            plan.append(current)
            current = []
            current_bytes = 0
        current.append(command)
        current_bytes += bytes_
    if current:
        plan.append(current)
    return plan


def fence_tick(ring: dict, frame: int, tick: int) -> dict:
    """Record the fence *tick* for *frame* in *ring*.

    Fence ticks are explicit, non-negative integers. A frame's recorded tick is
    monotonic: the first call for a frame accepts any valid tick, an equal tick
    is an idempotent update, and a smaller tick is a regression that raises
    :class:`RingError` without changing ``ring["fences"]``. A frame is not
    required to be committed before its tick is recorded. Fence ticks are not
    frame numbers and must never be used to decide which frames to reclaim.

    On success store the tick in ``ring["fences"][frame]`` and return exactly
    ``{"frame": frame, "tick": tick}``.
    """
    fences = _validate_fences(ring)
    frame = _check_non_negative(frame, "frame")
    tick = _check_non_negative(tick, "tick")

    if frame in fences and tick < fences[frame]:
        raise RingError("fence tick must not move backwards")
    fences[frame] = tick
    return {"frame": frame, "tick": tick}


def wait_fence(ring: dict, frame: int, tick: int) -> bool:
    """Check once whether *frame*'s fence has been reached at current *tick*.

    This is a non-blocking poll: it must not sleep, spin, or mutate the ring,
    and it returns immediately. If the frame has no recorded fence tick, return
    ``False``. Otherwise compare the caller's current ``tick`` with the
    recorded fence tick using a tolerance of one tick: return ``True`` when
    ``tick + 1 >= fence_tick`` (so being one tick early counts as reached), and
    ``False`` when the current tick is further behind.

    ``frame`` and ``tick`` must be non-negative integers; invalid values raise
    :class:`RingError`. Waiting for a later tick is the caller's job -- this
    function never busy-waits and never advances any tick.
    """
    fences = _validate_fences(ring)
    frame = _check_non_negative(frame, "frame")
    tick = _check_non_negative(tick, "tick")

    if frame not in fences:
        return False
    return tick + 1 >= fences[frame]


def validate(ring: dict, commands: Sequence[Mapping]) -> list[str]:
    """Return human-readable problems with a ring/command set.

    An empty list means the inputs are valid. This function reports problems
    instead of raising :class:`RingError`, and never mutates the ring or the
    commands. It checks at least:

    * ``ring["capacity"]`` is a positive integer;
    * ``ring["alignment"]`` is a positive power of two -- an alignment such as
      3 is reported, not silently rounded or used as a divisor;
    * every command has integer ``bytes >= 0`` whose **block size**, rounded up
      to the ring alignment, fits within ``capacity`` (an out-of-bounds command
      is reported rather than allowed to overwrite the ring);
    * every command has a non-negative integer ``frame``; frame numbers must
      not be reopened after a different frame has started (a repeated frame run
      separated by another frame is a duplicate-frame problem).

    Each issue is a short string and the result preserves discovery order:
    first ring-level problems, then command problems in input order.
    """
    issues: list[str] = []
    problems: list[str] = []

    capacity = None
    if not isinstance(ring, Mapping):
        problems.append("ring must be a mapping")
    else:
        raw_capacity = ring.get("capacity")
        if not _is_int(raw_capacity) or raw_capacity <= 0:
            problems.append("capacity must be a positive integer")
        else:
            capacity = raw_capacity

        alignment = ring.get("alignment")
        if (
            not _is_int(alignment)
            or alignment <= 0
            or alignment & (alignment - 1)
        ):
            problems.append("alignment must be a positive power of two")

        if capacity is not None:
            cursor = ring.get("cursor", 0)
            if not _is_int(cursor) or cursor < 0 or cursor > capacity:
                problems.append("cursor is out of range")
            used = ring.get("used", 0)
            if not _is_int(used) or used < 0 or used > capacity:
                problems.append("used is out of range")

    issues.extend(problems)

    if (
        not isinstance(commands, Sequence)
        or isinstance(commands, (str, bytes))
    ):
        issues.append("commands must be a sequence")
        return issues

    first_run: dict[int, int] = {}
    run = -1
    previous = object()
    for index, command in enumerate(commands):
        label = f"command {index}"
        if not isinstance(command, Mapping):
            issues.append(f"{label}: must be a mapping")
            continue
        bytes_ = command.get("bytes")
        frame = command.get("frame")
        bytes_ok = _is_int(bytes_) and bytes_ >= 0
        if not bytes_ok:
            issues.append(f"{label}: bytes must be a non-negative integer")
        frame_ok = _is_int(frame) and frame >= 0
        if not frame_ok:
            issues.append(f"{label}: frame must be a non-negative integer")

        if bytes_ok and capacity is not None:
            alignment = ring.get("alignment")
            if (
                _is_int(alignment)
                and alignment > 0
                and not alignment & (alignment - 1)
                and _block_size(bytes_, alignment) > capacity
            ):
                issues.append(f"{label}: block size exceeds capacity")

        if frame_ok:
            if frame != previous:
                run += 1
                previous = frame
            if frame in first_run and first_run[frame] != run:
                issues.append(f"{label}: frame {frame} is reopened")
            else:
                first_run.setdefault(frame, run)

    return issues


def ring_batch(frames: Sequence[Mapping], config: Mapping) -> dict:
    """Record a batch of frames as one all-or-nothing transaction.

    ``frames`` is an ordered sequence of ``{"frame": int, "commands": [...]}``
    mappings. Every command is exactly ``{"bytes": int, "frame": int}`` and its
    frame number must equal the enclosing frame's number. ``config`` contains
    ``{"ring": dict, "alignment": int, "budget": int}`` and optionally
    ``"ticks": {frame: tick}``. The per-frame budget is in command bytes; the
    alignment is a positive power of two.

    Frames and their commands are recorded in the supplied order. Each frame's
    command bytes must total at most ``budget``; all commands are allocated as
    aligned blocks (including tail padding), then the frame is marked committed
    in the ring. If ``config["ticks"]`` has an entry for a frame, its fence tick
    is recorded too. Frames must be unique, and command frame numbers must
    match their enclosing frame.

    Any invalid frame, duplicate outer frame, budget overflow, malformed
    command, alignment/capacity failure, allocation failure, or backwards
    fence tick raises :class:`RingError`. In that case the transaction is rolled
    back completely: ``ring["cursor"]``, ``ring["used"]``, ``ring["frames"]``
    and ``ring["fences"]`` are restored byte-for-byte, with no partial frame,
    block, or tick left behind. Frame state must not leak from one frame into
    the next.

    On success return::

        {
            "frames": [
                {
                    "frame": int,
                    "bytes": int,   # total command bytes for this frame
                    "blocks": [
                        {"offset": int, "size": int, "wrapped": bool},
                        ...
                    ],
                    "tick": int | None,
                },
                ...
            ],
            "committed": [frame numbers in input order],
            "ring": ring,
        }

    An empty batch returns empty ``frames`` and ``committed`` lists without
    changing the ring.
    """
    if not isinstance(config, Mapping):
        raise RingError("config must be a mapping")
    ring = config.get("ring")
    if not isinstance(ring, dict):
        raise RingError("config['ring'] must be a dict")
    capacity, _, cursor, _ = _validate_state(ring)
    fences = _validate_fences(ring)
    alignment = _check_alignment(config.get("alignment"))
    budget = config.get("budget")
    if not _is_int(budget) or budget <= 0:
        raise RingError("budget must be a positive integer")

    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes)):
        raise RingError("frames must be a sequence")

    ticks = config.get("ticks")
    if ticks is None:
        ticks = {}
    elif not isinstance(ticks, Mapping):
        raise RingError("ticks must be a mapping")
    for tick_frame, tick in ticks.items():
        _check_non_negative(tick_frame, "tick frame")
        _check_non_negative(tick, "tick")

    # Validate and plan the whole batch before touching the ring, so a bad
    # frame cannot leave earlier frames half-recorded. The cursor and used
    # counts accumulate across frames while each frame gets its own block list.
    seen_frames: set[int] = set()
    planned: list[dict] = []
    next_cursor = cursor
    next_used = ring["used"]
    for entry in frames:
        if not isinstance(entry, Mapping):
            raise RingError("frames must be mappings")
        if "frame" not in entry or "commands" not in entry:
            raise RingError("frames need 'frame' and 'commands'")
        frame = _check_non_negative(entry["frame"], "frame")
        if frame in seen_frames:
            raise RingError("duplicate frame in batch")
        if frame in ring["frames"]:
            raise RingError("frame is already recorded in the ring")
        seen_frames.add(frame)

        commands = entry["commands"]
        if not isinstance(commands, Sequence) or isinstance(
            commands, (str, bytes)
        ):
            raise RingError("frame commands must be a sequence")

        total = 0
        checked = []
        for command in commands:
            bytes_, command_frame = _check_command(command)
            if command_frame != frame:
                raise RingError("command frame does not match its frame")
            total += bytes_
            checked.append((command, bytes_))
        if total > budget:
            raise RingError("frame exceeds its byte budget")

        blocks = []
        frame_cursor = next_cursor
        frame_used = next_used
        for _, bytes_ in checked:
            block = _block_size(bytes_, alignment)
            if block > capacity:
                raise RingError("block does not fit in the ring")
            offset, frame_cursor, wrapped = _reserve(
                capacity, frame_cursor, frame_used, block
            )
            frame_used += block
            blocks.append(
                {"offset": offset, "size": block, "wrapped": wrapped}
            )
        next_cursor = frame_cursor
        next_used = frame_used

        tick = ticks.get(frame) if frame in ticks else None
        planned.append(
            {
                "frame": frame,
                "bytes": total,
                "blocks": blocks,
                "tick": tick,
                "cursor": frame_cursor,
                "used": frame_used,
            }
        )

    # Fence ticks are monotonic per frame; check against both the existing
    # ring fences and ticks recorded earlier in this same batch.
    pending_ticks: dict[int, int] = {}
    for item in planned:
        tick = item["tick"]
        if tick is None:
            continue
        frame = item["frame"]
        if frame in fences and tick < fences[frame]:
            raise RingError("fence tick must not move backwards")
        if frame in pending_ticks and tick < pending_ticks[frame]:
            raise RingError("fence tick must not move backwards")
        pending_ticks[frame] = tick

    snapshot = copy.deepcopy(ring)
    try:
        for item in planned:
            frame = item["frame"]
            ring["frames"][frame] = {
                "blocks": item["blocks"],
                "committed": True,
            }
            if item["tick"] is not None:
                ring["fences"][frame] = item["tick"]
            ring["cursor"] = item["cursor"]
            ring["used"] = item["used"]
    except Exception:
        restored = copy.deepcopy(snapshot)
        ring.clear()
        ring.update(restored)
        raise

    return {
        "frames": [
            {
                "frame": item["frame"],
                "bytes": item["bytes"],
                "blocks": item["blocks"],
                "tick": item["tick"],
            }
            for item in planned
        ],
        "committed": [item["frame"] for item in planned],
        "ring": ring,
    }


__all__ = [
    "RingError",
    "alloc_block",
    "wrap_check",
    "reclaim",
    "plan_frame",
    "fence_tick",
    "wait_fence",
    "validate",
    "ring_batch",
]
