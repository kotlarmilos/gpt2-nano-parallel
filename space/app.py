"""Standalone Hugging Face Space for the analytical ZeRO calculator."""

import gradio as gr


def calculate(parameters: int, world_size: int, stage: int) -> dict[str, float | str]:
    if parameters <= 0 or world_size <= 0 or stage not in {0, 1, 2, 3}:
        raise gr.Error("Use positive sizes and a ZeRO stage from 0 through 3")
    parameter_divisor = world_size if stage >= 3 else 1
    gradient_divisor = world_size if stage >= 2 else 1
    state_divisor = world_size if stage >= 1 else 1
    components = {
        "parameters_mib": parameters * 2 / parameter_divisor / 2**20,
        "gradients_mib": parameters * 2 / gradient_divisor / 2**20,
        "adam_moments_mib": parameters * 8 / state_divisor / 2**20,
        "master_parameters_mib": parameters * 4 / state_divisor / 2**20,
    }
    components["total_mib"] = sum(components.values())
    return {
        "evidence": "analytical",
        "assumptions": "BF16 parameters and gradients, FP32 Adam moments and master weights",
        **components,
    }


demo = gr.Interface(
    fn=calculate,
    inputs=[
        gr.Number(value=44_000_000, precision=0, label="Parameters"),
        gr.Number(value=8, precision=0, label="World size"),
        gr.Slider(0, 3, value=3, step=1, label="ZeRO stage"),
    ],
    outputs=gr.JSON(label="Per-rank model-state memory"),
    title="gpt2-nano ZeRO memory calculator",
    description="Analytical calculator only. It does not report measured GPU memory or scaling.",
)

if __name__ == "__main__":
    demo.launch()
