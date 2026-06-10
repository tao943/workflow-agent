from pathlib import Path

from src.cli.json_args import parse_json_object


def test_parse_json_object_accepts_inline_json() -> None:
    value, error = parse_json_object('{"text":"hello docker"}')

    assert error is None
    assert value == {"text": "hello docker"}


def test_parse_json_object_accepts_file_reference() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    args_file = output_dir / "test_cli_json_args.json"
    try:
        args_file.write_text('{"text":"hello docker"}', encoding="utf-8")

        value, error = parse_json_object(f"@{args_file}")

        assert error is None
        assert value == {"text": "hello docker"}
    finally:
        args_file.unlink(missing_ok=True)


def test_parse_json_object_reports_missing_file() -> None:
    value, error = parse_json_object("@missing-args.json")

    assert value is None
    assert error is not None
    assert "Cannot read JSON args file" in error
