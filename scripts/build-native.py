#!/usr/bin/env python3
"""Build laya with the repo's static-link flags plus -C target-cpu=native.

Same logic as scripts/build-release.py but keeps local paths and redirects
the output to target/native/ so it never clobbers the official build.
"""
import os
import pathlib
import subprocess
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
TARGET = "x86_64-unknown-linux-gnu"


def main():
    config = tomllib.loads((ROOT / ".cargo/config.toml").read_text())
    flags = list(config["target"][TARGET]["rustflags"])
    flags.extend(["-C", "target-cpu=native"])
    env = dict(os.environ)
    env.pop("RUSTFLAGS", None)  # RUSTFLAGS would override config rustflags
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
    env["CARGO_TARGET_DIR"] = str(ROOT / "target/native")
    subprocess.run(["cargo", "build", "--release", "--locked"], cwd=ROOT, env=env, check=True)
    print("Native build OK:", ROOT / "target/native" / TARGET / "release" / "laya")


if __name__ == "__main__":
    main()
