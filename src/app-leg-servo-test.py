#!/usr/bin/env python3
"""Conservative, one-channel-at-a-time test for the three V2 leg outputs.

This tool intentionally does not import the normal servo controller because that
module initializes hardware at import time and exposes multi-servo movements.
"""

from __future__ import annotations

import argparse
import configparser
import os
from dataclasses import dataclass
from pathlib import Path
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
LIVE_CONFIG = SCRIPT_DIR / "config.ini"
TEMPLATE_CONFIG = SCRIPT_DIR / "config.ini.template"
MAX_JOG_COUNTS = 5


@dataclass(frozen=True)
class ServoProfile:
    channel: int
    expected_function: str
    anchor: int
    minimum: int
    maximum: int


CHANNEL_CONFIG = {
    0: ("central leg lift (raise/lower), unverified", "neutralHeight", "upHeight", "downHeight"),
    1: ("starboard/left leg rotation, unverified", "neutralStarboard", "forwardStarboard", "backStarboard"),
    2: ("port/right leg rotation, unverified", "neutralPort", "forwardPort", "backPort"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely jog one V2 leg servo output by a few PWM counts."
    )
    parser.add_argument("--channel", type=int, choices=sorted(CHANNEL_CONFIG), required=True)
    parser.add_argument(
        "--step",
        type=int,
        default=2,
        help=f"Jog size in PWM counts (1-{MAX_JOG_COUNTS}; default: 2).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Enable hardware output. Without this flag the script is a dry run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Config file override (defaults to config.ini, or the template for dry runs).",
    )
    return parser.parse_args()


def resolve_config_path(requested: Path | None, execute: bool) -> Path:
    if requested is not None:
        path = requested.expanduser().resolve()
    elif LIVE_CONFIG.exists():
        path = LIVE_CONFIG
    else:
        path = TEMPLATE_CONFIG

    if not path.is_file():
        raise RuntimeError(f"Configuration file not found: {path}")
    if execute and path == TEMPLATE_CONFIG:
        raise RuntimeError("Hardware tests require the device's live src/config.ini file.")
    return path


def load_profile(config_path: Path, channel: int) -> ServoProfile:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(config_path)

    if not parser.has_section("SERVO"):
        raise RuntimeError(f"Missing [SERVO] section in {config_path}")
    if parser.get("SERVO", "MOVEMENT_VERSION", fallback="").strip().upper() != "V2":
        raise RuntimeError("This tester only supports MOVEMENT_VERSION = V2.")

    expected_function, anchor_key, first_limit_key, second_limit_key = CHANNEL_CONFIG[channel]
    anchor = parser.getint("SERVO", anchor_key)
    first_limit = parser.getint("SERVO", first_limit_key)
    second_limit = parser.getint("SERVO", second_limit_key)

    return ServoProfile(
        channel=channel,
        expected_function=expected_function,
        anchor=anchor,
        minimum=min(first_limit, second_limit),
        maximum=max(first_limit, second_limit),
    )


def find_conflicting_processes() -> list[str]:
    """Find another TARS or servo tester process without invoking shell tools."""
    conflicts = []
    own_pid = os.getpid()
    proc = Path("/proc")
    if not proc.is_dir():
        return conflicts

    watched_names = ("app.py", "app-servotester.py", "app-leg-servo-test.py")
    for cmdline_path in proc.glob("[0-9]*/cmdline"):
        try:
            pid = int(cmdline_path.parent.name)
            if pid == own_pid:
                continue
            command = cmdline_path.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except (OSError, ValueError):
            continue
        if any(name in command for name in watched_names):
            conflicts.append(f"PID {pid}: {command.strip()}")
    return conflicts


def pwm_to_duty_cycle(pwm_count: int) -> int:
    """Use the same 12-bit-count conversion as module_servoctl_v2.py."""
    return int((pwm_count / 4095.0) * 65535)


def set_output(pca, channel: int, pwm_count: int) -> None:
    pca.channels[channel].duty_cycle = pwm_to_duty_cycle(pwm_count)


def run_hardware_test(profile: ServoProfile, step: int) -> None:
    conflicts = find_conflicting_processes()
    if conflicts:
        details = "\n".join(f"  {item}" for item in conflicts)
        raise RuntimeError(f"Another process may control the servos:\n{details}")

    print("\nHARDWARE SAFETY CHECK")
    print("- Support the robot so it cannot fall or drive.")
    print("- Keep hands clear and keep servo power immediately reachable.")
    print("- Ideally disconnect the linkage or remove the horn for the first mapping test.")
    print(f"- The first command is the configured anchor ({profile.anchor}); it may cause a larger move.")
    confirmation = input(f"Type MOVE CHANNEL {profile.channel} to arm this one output: ").strip()
    if confirmation != f"MOVE CHANNEL {profile.channel}":
        print("Confirmation did not match; no hardware was initialized.")
        return

    try:
        import board
        import busio
        from adafruit_pca9685 import PCA9685
    except ImportError as exc:
        raise RuntimeError(f"Required Raspberry Pi servo library is unavailable: {exc}") from exc

    i2c = None
    pca = None
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c, address=0x40)
        pca.frequency = 50

        answer = input(
            f"Press Enter to energize ONLY channel {profile.channel} at {profile.anchor}, "
            "or type q to quit: "
        ).strip().lower()
        if answer == "q":
            return

        set_output(pca, profile.channel, profile.anchor)
        print(f"Channel {profile.channel} is holding at {profile.anchor}.")

        answer = input(
            f"Press Enter for a +{step} count jog and return, or type q to disable: "
        ).strip().lower()
        if answer == "q":
            return
        set_output(pca, profile.channel, profile.anchor + step)
        time.sleep(0.5)
        set_output(pca, profile.channel, profile.anchor)
        time.sleep(0.5)
        print("Positive jog complete. Note which physical servo moved and its direction.")

        answer = input(
            f"Press Enter for a -{step} count jog and return, or type q to disable: "
        ).strip().lower()
        if answer == "q":
            return
        set_output(pca, profile.channel, profile.anchor - step)
        time.sleep(0.5)
        set_output(pca, profile.channel, profile.anchor)
        time.sleep(0.5)
        print("Negative jog complete. Note which physical servo moved and its direction.")
    finally:
        if pca is not None:
            try:
                pca.channels[profile.channel].duty_cycle = 0
                print(f"Channel {profile.channel} output disabled; it is no longer holding position.")
            finally:
                pca.deinit()
        elif i2c is not None:
            i2c.deinit()


def main() -> int:
    args = parse_args()
    if not 1 <= args.step <= MAX_JOG_COUNTS:
        print(f"ERROR: --step must be between 1 and {MAX_JOG_COUNTS} counts.", file=sys.stderr)
        return 2

    try:
        config_path = resolve_config_path(args.config, args.execute)
        profile = load_profile(config_path, args.channel)
    except (configparser.Error, KeyError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not profile.minimum <= profile.anchor - args.step <= profile.maximum:
        print("ERROR: Negative jog would leave the configured servo limits.", file=sys.stderr)
        return 2
    if not profile.minimum <= profile.anchor + args.step <= profile.maximum:
        print("ERROR: Positive jog would leave the configured servo limits.", file=sys.stderr)
        return 2

    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"\nMode: {mode}")
    print(f"Config: {config_path}")
    print(f"Raw PCA9685 channel: {profile.channel}")
    print(f"Repository's expected function: {profile.expected_function}")
    print(f"Configured anchor: {profile.anchor}")
    print(f"Configured range: {profile.minimum}..{profile.maximum}")
    print(f"Planned sequence: {profile.anchor} -> {profile.anchor + args.step} -> "
          f"{profile.anchor} -> {profile.anchor - args.step} -> {profile.anchor} -> OFF")
    print("Only the selected channel is written; all other channels are untouched.")

    if not args.execute:
        print(f"\nDry run only. Add --execute on the Raspberry Pi after the physical safety check.")
        return 0

    try:
        run_hardware_test(profile, args.step)
    except (OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted; selected channel is being disabled.")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
