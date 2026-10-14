from __future__ import annotations

import random
import os
import shutil
import threading
import tkinter as tk
from queue import Empty, Queue
from pathlib import Path
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "sort"


def count_sorted_files() -> int:
    total = 0
    stack = [OUTPUT_ROOT / "good", OUTPUT_ROOT / "bad"]
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(Path(entry.path))
                    elif entry.is_file():
                        total += 1
        except OSError:
            continue
    return total

# ----------------------------------------------------------------------------------
# Manual sorting app - Anthony Baird
#   App to assist in manual classification of good/bad images
#   
# ----------------------------------------------------------------------------------
class ManualSortApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Manual Image Sorter")
        self.root.resizable(False, False)

        self.source_dirs: list[Path] = []
        self.current_dir_index = 0
        self.current_image: Path | None = None
        self.image_history: dict[Path, set[Path]] = {}
        self.last_image: dict[Path, Path] = {}
        self.sorted_count = 0
        self.sorted_count_ready = False
        self.sorted_count_queue = Queue()
        self.photo: ImageTk.PhotoImage | None = None

        self.source_label = tk.Label(root, text="No source directory selected", anchor="w")
        self.source_label.pack(fill="x", padx=20, pady=(15, 5))
        tk.Button(root, text="Choose source directory", command=self.choose_source).pack(
            fill="x", padx=20
        )

        self.directory_label = tk.Label(root, text="", font=("TkDefaultFont", 10))
        self.directory_label.pack(pady=(12, 2))
        self.image_label = tk.Label(root, text="Choose a source directory to begin", width=70, height=25)
        self.image_label.pack(padx=20, pady=5)

        self.count_label = tk.Label(root, text="Sorted: counting...", font=("TkDefaultFont", 11, "bold"))
        self.count_label.pack(pady=(4, 8))

        button_frame = tk.Frame(root)
        button_frame.pack(fill="x", padx=20, pady=(0, 15))
        self.good_button = tk.Button(
            button_frame, text="Good (G)", command=lambda: self.sort_current("good"), state="disabled"
        )
        self.good_button.pack(side="left", expand=True, fill="x", padx=(0, 5))
        self.bad_button = tk.Button(
            button_frame, text="Bad (B)", command=lambda: self.sort_current("bad"), state="disabled"
        )
        self.bad_button.pack(side="left", expand=True, fill="x", padx=(5, 0))

        self.root.bind("<KeyPress-g>", lambda _event: self.sort_current("good"))
        self.root.bind("<KeyPress-b>", lambda _event: self.sort_current("bad"))
        self.root.after(50, self.poll_sorted_count)
        threading.Thread(
            target=lambda: self.sorted_count_queue.put(count_sorted_files()),
            daemon=True,
        ).start()

    def poll_sorted_count(self):
        try:
            self.sorted_count = self.sorted_count_queue.get_nowait()
        except Empty:
            self.root.after(50, self.poll_sorted_count)
            return

        self.sorted_count_ready = True
        self.count_label.config(text=f"Sorted: {self.sorted_count}")
        self.set_sort_buttons(self.current_image is not None)

    def choose_source(self):
        selected = filedialog.askdirectory(parent=self.root)
        if not selected:
            return

        source_root = Path(selected)
        source_dirs = [
            path for path in sorted(source_root.iterdir(), key=lambda path: path.name.lower())
            if path.is_dir() and self.image_paths(path)
        ]
        if not source_dirs:
            messagebox.showerror(
                "No image subdirectories",
                "The selected directory must contain subdirectories with images.",
            )
            return

        self.source_dirs = source_dirs
        self.current_dir_index = 0
        self.current_image = None
        self.image_history.clear()
        self.last_image.clear()
        self.source_label.config(text=str(source_root))
        self.set_sort_buttons(self.sorted_count_ready)
        self.show_next_image()

    @staticmethod
    def image_paths(directory: Path) -> list[Path]:
        return sorted(
            path for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    def set_sort_buttons(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.good_button.config(state=state)
        self.bad_button.config(state=state)

    def choose_image(self, directory: Path) -> Path | None:
        images = self.image_paths(directory)
        if not images:
            return None

        history = self.image_history.setdefault(directory, set())
        available = [
            image for image in images
            if image not in history
            and not any(
                (OUTPUT_ROOT / category / f"{directory.name}_{image.name}").is_file()
                for category in ("good", "bad")
            )
        ]
        if not available:
            return None

        previous = self.last_image.get(directory)
        if len(available) > 1 and previous in available:
            available = [image for image in available if image != previous]

        selected = random.choice(available)
        history.add(selected)
        self.last_image[directory] = selected
        return selected

    def show_next_image(self):
        if not self.source_dirs:
            return

        for _ in range(len(self.source_dirs)):
            directory = self.source_dirs[self.current_dir_index]
            self.current_dir_index = (self.current_dir_index + 1) % len(self.source_dirs)
            image_path = self.choose_image(directory)
            if image_path is not None:
                break
        else:
            self.current_image = None
            self.directory_label.config(text="No unsorted images remain")
            self.image_label.config(image="", text="All images in this source have already been sorted.")
            self.set_sort_buttons(False)
            return

        self.current_image = image_path
        self.directory_label.config(text=f"Directory: {directory.name}    Image: {image_path.name}")
        self.display_image(image_path)

    def display_image(self, image_path: Path):
        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((760, 560), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.photo, text="", width=760, height=560)
        except (OSError, ValueError) as error:
            self.image_label.config(image="", text=f"Could not display image: {error}")

    def sort_current(self, category: str):
        if not self.sorted_count_ready or self.current_image is None:
            return

        source_image = self.current_image
        destination_name = f"{source_image.parent.name}_{source_image.name}"
        destination = OUTPUT_ROOT / category / destination_name
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, destination)
        except OSError as error:
            messagebox.showerror("Copy failed", f"Could not copy:\n{source_image}\n\n{error}")
            return

        self.sorted_count += 1
        self.count_label.config(text=f"Sorted: {self.sorted_count}")
        self.current_image = None
        self.show_next_image()


def main():
    root = tk.Tk()
    ManualSortApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
