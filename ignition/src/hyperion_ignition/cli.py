"""CLI entrypoint: `hyperion-compile --model <id> --hw <sm_90|sm_120> --source <dir> --out <file>`.

Drives the compile lifecycle: build a model-specific compiler from the
registry, select a hardware recipe, run the pipeline, sign, write the
`/hyperion` package + the `.manifest.json` sidecar the runtime reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from .compiler.registry import build_compiler
from .hardware.registry import build_recipe
from .manifest_schema import validate_manifest


@click.command()
@click.option("--model", "model_id", required=True, help="Model id: llama3, qwen2 or deepseek")
@click.option("--hw", "target_arch", required=True, help="Target compute capability, e.g. sm_90")
@click.option("--out", "out_path", type=click.Path(path_type=Path), required=True)
@click.option("--source", "source_path", type=click.Path(path_type=Path), required=True)
def main(model_id: str, target_arch: str, out_path: Path, source_path: Path) -> None:
    """Compile a model into a `.hyperion` package for the given hardware."""
    try:
        compiler = build_compiler(model_id)
        recipe = build_recipe(target_arch)
        click.echo(f"compiling {model_id} for {target_arch} via {compiler.__class__.__name__}")
        manifest = compiler.compile(source=source_path, recipe=recipe)
        document = manifest.to_json_dict()
        validate_manifest(document)  # nothing is written unless it satisfies the schema
    except (KeyError, ValueError, OSError, NotImplementedError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) else str(exc)
        raise click.ClickException(message) from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    click.echo(f"wrote manifest -> {manifest_path}")

    # TODO: write the encrypted weight blob to out_path using the manifest's
    #       layer byte offsets, then sign and append the ed25519 signature.


if __name__ == "__main__":
    main()
