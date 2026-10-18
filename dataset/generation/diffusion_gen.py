import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["DEEPFACE_LOG_LEVEL"] = "40"
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.disable(logging.CRITICAL)
import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
import random
import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline
from deepface import DeepFace
from tqdm import tqdm
from compel import Compel
from variation import *

MODEL_NAME = "serenity"
MODEL_FILE = "serenity_v21.safetensors"

START_AGE = 1
END_AGE = 100
OUTPUT_BASE_DIR = f"dataset/generation/gen/images"
MODEL_BASE_DIR = "dataset/generation/pretrained_models"
IMAGE_SIZE = 512

GUIDANCE_SCALE = 4.5
NUM_INFERENCE_STEPS = 10

MIN_CROP_FRACTION = 0.55
FACE_CROP_MARGIN = 0.6

BASE_SEED = 42
IMAGE_COUNTER = 0


def make_seed(seed_index):
    return (seed_index * 2654435761) & 0x7FFFFFFF


NEGATIVE_PROMPT = (
    "cartoon, painting, cgi, deformed, disfigured, extra limbs, blurry, low quality, "
    "bindi, bad anatomy, multiple people, watermark, text, monochrome, black and white, "
    "full body, extreme close-up, only eyes, profile view, overprocessed, eyebrow shadow, "
    "hand on face, forehead marks, forehead lines, forehead shadows, dark bands, headband, "
    "dark streaks, shadow artifacts, harsh shadows, heavy brow shadow"
)

device = "cuda" if torch.cuda.is_available() else "cpu"
def build_prompt(target_age, seed):
    rng = random.Random(seed)
    ethnicity = rng.choice(ETHNICITIES)
    traits = ETHNICITY_TRAITS[ethnicity]
    style = rng.choice(PORTRAIT_STYLES)
    lighting = rng.choice(LIGHTING)
    expression = rng.choice(EXPRESSIONS)
    hair_color = select_hair_color(target_age, traits, rng)
    hair_texture = rng.choice(traits["hair_textures"])
    hair_len = rng.choice(HAIR_LENGTHS)
    eye_color = rng.choice(traits["eye_colors"])
    gender = rng.choice(GENDERS)
    face_shape = rng.choice(FACE_SHAPES)
    nose_type = rng.choice(NOSE_TYPES)
    face_weight = rng.choice(FACE_WEIGHTS)
    shirt_type = rng.choice(CLOTHING_TYPES)
    face_mod = rng.choice(FACE_MODS)
    eyebrow_type = rng.choice(EYEBROW_TYPES)
    lip_type = rng.choice(LIP_TYPES)
    head_pose = rng.choice(HEAD_POSES)
    subject = get_subject(target_age, ethnicity)
    environment = get_environment(target_age, rng)
    if gender == "Male" and hair_len == "long hair":
        hair_len = rng.choice(["long hair", "short hair"])
    return (
        f"{style} in a {environment} of {subject} {gender} with {hair_texture} {hair_len} natural {hair_color} hair and {eye_color} eyes, "
        f"{face_weight}, {face_shape}, {face_mod}, natural {eyebrow_type}, {lip_type}, {nose_type}, "
        f"{head_pose}, {shirt_type}, {expression}, "
        "medium headshot, complete hair visible, upper chest and shoulders in frame, "
        f"{lighting}, detailed face, sharp focus"
    )

def get_num_per_age(target_age):
    if target_age < 8:
        return 6_500
    if target_age < 14:
        return 6_000
    if target_age < 16:
        return 4_500
    if target_age < 18:
        return 3_000
    if target_age < 25:
        return 2_000
    if target_age < 50:
        return 1_000
    if target_age < 55:
        return 2_000
    if target_age < 59:
        return 3_000
    if target_age < 63:
        return 4_000
    if target_age < 66:
        return 4_500
    return 5_000

def detect_face_box(pil_img):
    """Return (fx, fy, fw, fh) for the largest detected face, or None if none found."""
    rgb_array = np.array(pil_img)
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    try:
        faces = DeepFace.extract_faces(
            img_path=bgr_array,
            detector_backend="yunet",
            enforce_detection=True,
            align=False,
        )
        if len(faces) != 1:
            return None
        area = faces[0]["facial_area"]
        return area["x"], area["y"], area["w"], area["h"]
    except Exception as e:
        print(f"Face detect err: {e}")
        return None

def crop_to_face(pil_img):
    face_box = detect_face_box(pil_img)
    if face_box is None:
        return None
    
    width, height = pil_img.size
    fx, fy, fw, fh = face_box
    if not (
        0 <= fx < width and
        0 <= fy < height and
        fw > 0 and
        fh > 0 and
        fx + fw <= width and
        fy + fh <= height
    ):
        print("INVALID FACE BOX", face_box, (width, height))
        return None

    cx = fx + fw / 2
    # bias cy upward 15% of fh to preserve hair/forehead
    cy = fy + fh / 2 - fh * .15 
    side = int(max(fw, fh) * (1 + FACE_CROP_MARGIN))
    side = max(side, int(min(width, height) * MIN_CROP_FRACTION))
    side = min(side, width, height)
    half = side // 2

    left = int(np.clip(cx - half, 0, width - side))
    top = int(np.clip(cy - half, 0, height - side))

    cropped = pil_img.crop((left, top, left + side, top + side))
    if (
        left < 0 or
        top < 0 or
        left + side > width or
        top + side > height
    ):
        print(
            f"BAD CROP: "
            f"img=({width},{height}) "
            f"crop=({left},{top},{left+side},{top+side})"
        )
        print(
            f"img=({width},{height}) "
            f"face=({fx},{fy},{fw},{fh}) "
            f"cy={cy:.1f} "
            f"side={side} "
            f"left={left} "
            f"top={top}"
        )
    resized = cropped.resize((IMAGE_SIZE, IMAGE_SIZE))
    rw, rh = resized.size
    new_face = detect_face_box(resized)
    if new_face is None:
        return None
    new_fx, new_fy, new_fw, new_fh = new_face
    # Check for x-centeredness
    new_cx = new_fx + new_fw / 2
    new_cy = new_fy + new_fh / 2
    img_cx = rw / 2
    x_offset = abs(new_cx - img_cx) / rw
    max_offset = .1
    if x_offset > max_offset:
        return None
    # Check for small faces
    min_face_w = .45
    if new_fw < min_face_w * rw:
        return None
    # top/bottom margin
    min_top_margin = .3 * new_fh
    if new_fy < min_top_margin:
        return None
    min_bottom_margin = .05 * new_fh
    if rh - new_fy - new_fh < min_bottom_margin:
        return None
    return resized


def generate_face_at_age(target_age, seed, pipe, compel):
    prompt = build_prompt(target_age, seed)
    prompt_embeds = compel(prompt)
    negative_prompt_embeds = compel(NEGATIVE_PROMPT)

    generator = torch.Generator(device=device).manual_seed(seed)

    raw_image = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        generator=generator,
    ).images[0]

    return crop_to_face(raw_image)

def main():
    global BASE_SEED, IMAGE_COUNTER
    model_loc = f"{MODEL_BASE_DIR}/{MODEL_FILE}"
    save_loc = f"{OUTPUT_BASE_DIR}/{MODEL_NAME}"
    print(f"[{model_loc}], [{save_loc}]")
    pipe = StableDiffusionPipeline.from_single_file(
        model_loc,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)

    total_images = sum(
        get_num_per_age(age) for age in range(START_AGE, END_AGE + 1)
    )
    print(
        f"Generating dataset: Ages {START_AGE} to {END_AGE}, "
        f"{total_images} images total."
    )
    with tqdm(total=total_images, desc="Generating Images", unit="img") as pbar:
        for age in range(START_AGE, END_AGE + 1):
            age_dir = os.path.join(save_loc, str(age))
            os.makedirs(age_dir, exist_ok=True)
            num_images = get_num_per_age(age)
            for idx in range(num_images):
                trials = 0
                while(True):
                    seed = make_seed(BASE_SEED)
                    BASE_SEED += 1
                    trials += 1
                    image = generate_face_at_age(
                        target_age=age, 
                        seed=seed,
                        pipe=pipe,
                        compel=compel
                    )
                    if image is not None or trials > 20:
                        break
                if image is None:
                    continue
                
                output_path = os.path.join(age_dir, f"synth_{IMAGE_COUNTER}.jpg")
                image.save(output_path, format="JPEG", quality=95)
                IMAGE_COUNTER += 1
                pbar.set_postfix(
                    age=age, idx=f"{idx + 1}/{num_images}"
                )
                pbar.update(1)

    print(f"\nDone. Dataset saved to '{OUTPUT_BASE_DIR}'.")

if __name__ == "__main__":
    main()
