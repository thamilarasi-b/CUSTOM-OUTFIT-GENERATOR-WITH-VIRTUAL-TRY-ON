from huggingface_hub import login
login("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")

# ── Imports ──────────────────────────────────────────────────────────────────────
import gradio as gr
import torch, cv2, numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)
from controlnet_aux import OpenposeDetector
import mediapipe as mp

# ── Device & dtype ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# ── Load ControlNet (OpenPose) + SD15 base once at start ─────────────────────────
print("Loading ControlNet-OpenPose…")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", torch_dtype=dtype
).to(device)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=dtype,
).to(device)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
#pipe.enable_xformers_memory_efficient_attention()
pipe.safety_checker = None

# ── Mediapipe pose-drawing helper ────────────────────────────────────────────────
mp_pose = mp.solutions.pose

def draw_pose(image_np: np.ndarray) -> np.ndarray:
    h, w, _ = image_np.shape
    canvas = np.zeros_like(image_np)
    with mp_pose.Pose(static_image_mode=True) as pose:
        res = pose.process(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
        if res.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                canvas,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    thickness=2, circle_radius=2, color=(0,255,0)
                ),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    thickness=2, color=(255,0,0)
                ),
            )
    return canvas

# ── OpenPose detector ────────────────────────────────────────────────────────────
openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

# ── Inference Function ───────────────────────────────────────────────────────────
def generate_outfit(
    input_img: Image.Image,
    gender: str,
    season: str,
    style: str,
    parts: str,
    custom_prompt: str,
    guidance: float,
    steps: int,
):
    if custom_prompt.strip():
        prompt = custom_prompt.strip()
    else:
        prompt = f"{season} {style} {parts} outfit for a {gender}, full-body photo, high-quality fashion editorial"

    pose_img = openpose(input_img)

    gen = pipe(
        prompt=prompt,
        negative_prompt="deformed, crooked, lowres, blurry, extra limbs",
        image=pose_img,
        num_inference_steps=steps,
        guidance_scale=guidance,
    ).images[0]

    pose_preview = Image.fromarray(draw_pose(np.array(input_img)))

    return input_img, gen, pose_preview

# ── Gradio UI ────────────────────────────────────────────────────────────────────
def main():
    with gr.Blocks(
        title="AI-Powered Custom Outfit Generator",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            "🍬 **AI-Powered Custom Outfit Generator**  \n"
            "Real Stable Diffusion + ControlNet Fashion Technology  \n"
            "_Upload your photo (or use your webcam) and let AI renew your wardrobe!_"
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("📥 **Input**")
                input_image = gr.Image(
                    label="Upload Photo or Use Webcam",
                    type="pil",
                    height=480,
                )

            with gr.Column():
                gr.Markdown("🎯 **AI Generated Results**")
                gen_image = gr.Image(label="AI Generated Outfit", height=480)

        with gr.Row():
            with gr.Column(scale=1):
                pass
            with gr.Column(scale=1):
                pose_preview = gr.Image(label="🤖 Detected Pose Structure", height=240)

        with gr.Accordion("🎨 Style Preferences", open=True):
            gender = gr.Dropdown(["male", "female", "unisex"], label="Gender", value="male")
            season = gr.Dropdown(["summer", "winter", "spring", "autumn"], label="Season", value="summer")
            style = gr.Dropdown(["casual", "formal", "streetwear", "business", "sporty", "retro", "grunge"], label="Style", value="casual")
            parts = gr.Dropdown(["full body", "top only", "bottom only"], label="Outfit Parts", value="full body")
            custom_prompt = gr.Textbox(label="📝 Optional: Describe your outfit idea", placeholder="e.g. futuristic techwear with neon highlights")
            guidance = gr.Slider(3, 15, value=9, label="CFG Scale")
            steps = gr.Slider(10, 50, value=25, label="Inference Steps")

        btn = gr.Button("✨ Generate Outfit")

        btn.click(
            fn=generate_outfit,
            inputs=[input_image, gender, season, style, parts, custom_prompt, guidance, steps],
            outputs=[input_image, gen_image, pose_preview],
        )

    demo.launch()

if __name__ == "__main__":
    main()
