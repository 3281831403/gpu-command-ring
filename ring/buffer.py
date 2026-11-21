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

from collections.abc import Mapping, Sequence


class RingError(Exception):
    """Raised for an invalid ring, command, frame, alignment, or fence tick.

    Public operations validate their inputs before mutating the ring and use
    this exception for contract violations. A transaction that raises must
    leave the ring in its pre-call state.
    """


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
