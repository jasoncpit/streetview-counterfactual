from types import SimpleNamespace

import pytest

from src.integrations.replicate_client import ReplicateClient, ReplicateModels


@pytest.fixture()
def models():
    return ReplicateModels(
        dino_model="dino",
        sam_model="sam",
        inpaint_model="inpaint",
        flux_kontext_model="black-forest-labs/flux-kontext-max",
        nano_banana_model="banana",
        inpaint_params={},
        flux_kontext_params={
            "aspect_ratio": "match_input_image",
            "output_format": "jpg",
            "safety_tolerance": 2,
            "prompt_upsampling": False,
        },
    )


@pytest.fixture()
def client(monkeypatch, models):
    monkeypatch.setenv("REPLICATE_API_TOKEN", "test-token")
    return ReplicateClient(models=models)


@pytest.mark.parametrize(
    "model,expected_key",
    [
        ("google/nano-banana-pro", "image_input"),
        ("bytedance/seedream-4", "image_input"),
        ("openai/gpt-image-1.5", "input_images"),
        ("black-forest-labs/flux-kontext-max", "input_image"),
    ],
)
def test_build_baseline_payload_keys(client, model, expected_key):
    payload, output_format = client._build_baseline_payload(
        model,
        image_handle=SimpleNamespace(name="dummy"),
        formatted_prompt="test prompt",
        use_alt=False,
    )
    assert expected_key in payload
    assert output_format in {None, "png", "jpg", "jpeg"}


def test_build_baseline_payload_gpt_image_alt_key(client):
    payload, _ = client._build_baseline_payload(
        "openai/gpt-image-1.5",
        image_handle=SimpleNamespace(name="dummy"),
        formatted_prompt="test prompt",
        use_alt=True,
    )
    assert "image" in payload


@pytest.mark.parametrize(
    "fmt,expected",
    [
        (None, ".png"),
        ("png", ".png"),
        ("jpg", ".jpg"),
        ("jpeg", ".jpg"),
        ("JPG", ".jpg"),
    ],
)
def test_suffix_from_format(client, fmt, expected):
    assert client._suffix_from_format(fmt) == expected


def test_normalize_result(client):
    assert client._normalize_result([1, 2, 3]) == 1
    assert client._normalize_result((4, 5)) == 4
    assert client._normalize_result("x") == "x"


def test_match_size_cover_crop(tmp_path, client):
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    from PIL import Image

    Image.new("RGB", (400, 200), color="red").save(input_path)
    Image.new("RGB", (200, 400), color="blue").save(output_path)

    client._match_size(output_path, str(input_path))
    out = Image.open(output_path)
    assert out.size == (400, 200)
