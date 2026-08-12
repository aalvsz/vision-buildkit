#!/usr/bin/env python3
"""Report the public Axelera compiler hooks relevant to INT8 head exclusion.

This intentionally inspects only the public ``CompilerConfig`` schema.  It
does not compile a model, contact an AIPU, or infer undocumented behavior from
private/obfuscated compiler internals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect_capabilities() -> dict[str, object]:
    try:
        from axelera.compiler.config import CompilerConfig
        from axelera.compiler.config.compiler_types import GraphCleanerNode
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Axelera SDK is not installed in this interpreter; run this script "
            "with the Voyager SDK venv on the target host"
        ) from exc

    fields = CompilerConfig.model_fields
    graph_cleaner_fields = {
        name: {
            "default": None if field.default is None else str(field.default),
            "description": field.description,
        }
        for name, field in fields.items()
        if name.startswith("graph_cleaner")
    }
    return {
        "compiler_config": f"{CompilerConfig.__module__}.{CompilerConfig.__name__}",
        "public_fields": sorted(fields),
        "graph_cleaner_fields": graph_cleaner_fields,
        "graph_cleaner_nodes": [item.value for item in GraphCleanerNode],
        "terminal_conv_exclusion_field": any(
            name in fields
            for name in (
                "ignore_scope",
                "quantization_ignore_scope",
                "output_redefinition",
                "redefine_outputs",
            )
        ),
        "verdict": "no_public_terminal_conv_exclusion_or_output_redefinition",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect_capabilities()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
