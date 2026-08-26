from __future__ import annotations

import io
import os
from typing import Any

import gradio as gr
import httpx
from PIL import Image

API_URL = os.getenv(
    "GRADIO_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

ANATOM_SITE_CHOICES = [
    "abdomen",
    "acral",
    "back",
    "chest",
    "ear",
    "face",
    "foot",
    "genital",
    "hand",
    "lower extremity",
    "neck",
    "scalp",
    "trunk",
    "upper extremity",
]


def predict_via_api(
    image: Image.Image | None,
    age: float | None,
    sex: str | None,
    anatom_site: str | None,
) -> dict[str, Any]:
    if image is None:
        raise gr.Error("Please upload a dermatoscopic image.")

    buffer = io.BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=95,
    )

    files = {
        "file": (
            "image.jpg",
            buffer.getvalue(),
            "image/jpeg",
        )
    }

    data: dict[str, str] = {
        "top_k": "3",
    }

    if age is not None:
        data["age"] = str(age)

    if sex:
        data["sex"] = sex

    if anatom_site:
        data["anatom_site"] = anatom_site

    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(
                f"{API_URL}/predict",
                files=files,
                data=data,
            )

        response.raise_for_status()

    except httpx.HTTPStatusError as error:
        try:
            detail = error.response.json().get(
                "detail",
                error.response.text,
            )
        except ValueError:
            detail = error.response.text

        raise gr.Error(
            f"API returned {error.response.status_code}: {detail}"
        ) from error

    except httpx.RequestError as error:
        raise gr.Error(
            f"Could not connect to the API at {API_URL}."
        ) from error

    return response.json()


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Skin Lesion Classification Demo",
    ) as demo:
        gr.Markdown(
            """
# Skin Lesion Classification Demo

Upload a dermatoscopic image and optionally provide metadata.

"""
        )

        with gr.Row():
            with gr.Column():
                image = gr.Image(
                    type="pil",
                    label="Dermatoscopic image",
                )

                age = gr.Number(
                    label="Age",
                    minimum=0,
                    maximum=120,
                )

                sex = gr.Dropdown(
                    choices=[
                        "female",
                        "male",
                    ],
                    value=None,
                    label="Sex",
                )

                anatom_site = gr.Dropdown(
                    choices=ANATOM_SITE_CHOICES,
                    value=None,
                    label="Anatomical site",
                )

                predict_button = gr.Button(
                    "Predict",
                    variant="primary",
                )

            with gr.Column():
                output = gr.JSON(
                    label="Prediction",
                )

        predict_button.click(
            fn=predict_via_api,
            inputs=[
                image,
                age,
                sex,
                anatom_site,
            ],
            outputs=output,
        )

    return demo


def main() -> None:
    demo = build_demo()

    demo.launch(
        server_name=os.getenv(
            "GRADIO_SERVER_NAME",
            "127.0.0.1",
        ),
        server_port=int(
            os.getenv(
                "GRADIO_SERVER_PORT",
                "7860",
            )
        ),
        show_error=True,
    )


if __name__ == "__main__":
    main()