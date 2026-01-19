import os
import glob
import json
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Det:
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    cls_name: str
    conf: float


def list_images(folder: str, recursive: bool) -> List[str]:
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        return []
    pattern = os.path.join(folder, "**", "*") if recursive else os.path.join(folder, "*")
    paths = []
    for p in glob.glob(pattern, recursive=recursive):
        if os.path.isfile(p) and os.path.splitext(p)[1].lower() in IMG_EXTS:
            paths.append(os.path.abspath(p))
    paths.sort()
    return paths


def load_thresholds_json(path: Optional[str]) -> Dict[str, float]:
    """
    支援幾種常見格式（best-effort）：
    1) {"Scratch":0.35,"Burr":0.5}
    2) {"thresholds":{"Scratch":0.35}}
    3) [{"name":"Scratch","th":0.35}, ...]
    """
    if not path:
        return {}
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Threshold JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        if "thresholds" in data and isinstance(data["thresholds"], dict):
            return {str(k): float(v) for k, v in data["thresholds"].items()}
        return {str(k): float(v) for k, v in data.items() if isinstance(v, (int, float))}

    if isinstance(data, list):
        out = {}
        for item in data:
            if isinstance(item, dict):
                name = item.get("name") or item.get("class") or item.get("label")
                th = item.get("th") or item.get("threshold") or item.get("score")
                if name is not None and th is not None:
                    try:
                        out[str(name)] = float(th)
                    except Exception:
                        pass
        return out

    return {}


def infer_ultralytics(model, img_bgr: np.ndarray, conf: float, iou: float) -> Tuple[List[Det], float]:
    """
    回傳：dets, inference_ms
    """
    t0 = time.perf_counter()
    results = model.predict(source=img_bgr, conf=conf, iou=iou, verbose=False)
    t1 = time.perf_counter()
    ms = (t1 - t0) * 1000.0

    dets: List[Det] = []
    if not results:
        return dets, ms

    r = results[0]
    names = getattr(r, "names", {}) or {}

    if r.boxes is None or len(r.boxes) == 0:
        return dets, ms

    xyxy = r.boxes.xyxy.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    confs = r.boxes.conf.cpu().numpy()

    for (x1, y1, x2, y2), cid, cf in zip(xyxy, cls, confs):
        cname = names.get(int(cid), str(int(cid)))
        dets.append(Det(int(x1), int(y1), int(x2), int(y2), int(cid), str(cname), float(cf)))

    return dets, ms


def split_by_class_thresholds(dets: List[Det], thresholds: Dict[str, float]) -> Tuple[List[Det], List[Det]]:
    """
    回傳：passed(>=th), filtered(<th)
    若 thresholds 為空，全部視為 passed。
    """
    if not thresholds:
        return dets, []
    passed, filtered = [], []
    for d in dets:
        th = thresholds.get(d.cls_name)
        if th is None:
            passed.append(d)
        else:
            (passed if d.conf >= float(th) else filtered).append(d)
    return passed, filtered



def draw_boxes(img_bgr: np.ndarray, dets: List[Det], color=(0, 255, 0)) -> np.ndarray:
    out = img_bgr.copy()
    for d in dets:
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        label = f"{d.cls_name} {d.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        y0 = max(d.y1 - th - 6, 0)
        cv2.rectangle(out, (d.x1, y0), (d.x1 + tw + 6, y0 + th + 6), color, -1)
        cv2.putText(out, label, (d.x1 + 3, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    return out



class ViewerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YOLO Viewer (Tkinter, no-save)")

        self.model = None
        self.model_path = ""
        self.folder = ""
        self.images: List[str] = []
        self.idx = 0

        self.thresholds: Dict[str, float] = {}
        self.th_json_path = ""

        self.conf = 0.25
        self.iou = 0.45
        self.recursive = True

        # UI layout
        self._build_ui()

        # key binds
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.bind("a", lambda e: self.prev_image())
        self.root.bind("d", lambda e: self.next_image())
        self.root.bind("f", lambda e: self.jump(+20))
        self.root.bind("r", lambda e: self.jump(-20))
        self.root.bind("g", lambda e: self.goto_dialog())
        self.root.bind("q", lambda e: self.root.quit())
        self.root.bind("<Escape>", lambda e: self.root.quit())

    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=8)

        btn_model = tk.Button(top, text="選模型(.pt)", command=self.pick_model)
        btn_model.pack(side=tk.LEFT)

        btn_folder = tk.Button(top, text="選資料夾", command=self.pick_folder)
        btn_folder.pack(side=tk.LEFT, padx=(8, 0))

        btn_json = tk.Button(top, text="選門檻JSON(可選)", command=self.pick_json)
        btn_json.pack(side=tk.LEFT, padx=(8, 0))

        self.var_recursive = tk.BooleanVar(value=True)
        chk_rec = tk.Checkbutton(top, text="遞迴子資料夾", variable=self.var_recursive, command=self.on_toggle_recursive)
        chk_rec.pack(side=tk.LEFT, padx=(12, 0))
        self.var_show_filtered = tk.BooleanVar(value=True)
        chk_f = tk.Checkbutton(top, text="顯示未過門檻框(OK也顯示)", variable=self.var_show_filtered, command=self.refresh)
        chk_f.pack(side=tk.LEFT, padx=(12, 0))


        mid = tk.Frame(self.root)
        mid.pack(side=tk.TOP, fill=tk.X, padx=8)

        # conf slider
        self.var_conf = tk.DoubleVar(value=self.conf)
        s_conf = tk.Scale(mid, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                          label="conf", variable=self.var_conf, command=self.on_conf_change, length=260)
        s_conf.pack(side=tk.LEFT)

        # iou slider
        self.var_iou = tk.DoubleVar(value=self.iou)
        s_iou = tk.Scale(mid, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                         label="iou", variable=self.var_iou, command=self.on_iou_change, length=260)
        s_iou.pack(side=tk.LEFT, padx=(12, 0))

        nav = tk.Frame(self.root)
        nav.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        tk.Button(nav, text="<< 上一張 (A/←)", command=self.prev_image).pack(side=tk.LEFT)
        tk.Button(nav, text="下一張 (D/→) >>", command=self.next_image).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(nav, text="快轉+20 (F)", command=lambda: self.jump(+20)).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(nav, text="倒轉-20 (R)", command=lambda: self.jump(-20)).pack(side=tk.LEFT, padx=(8, 0))
        tk.Button(nav, text="跳到 (G)", command=self.goto_dialog).pack(side=tk.LEFT, padx=(8, 0))

        self.info = tk.Label(self.root, text="尚未載入模型/資料夾", anchor="w", justify="left")
        self.info.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 6))

        # image panel
        self.canvas = tk.Label(self.root, bd=2, relief=tk.GROOVE)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=8)

        help_text = (
            "快捷鍵：A/← 上一張，D/→ 下一張，F +20，R -20，G 跳到，Q/ESC 離開\n"
            "上方滑桿可調 conf / iou。完全不存圖，只顯示疊框結果。"
        )
        self.help = tk.Label(self.root, text=help_text, anchor="w", justify="left", fg="#444")
        self.help.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(0, 8))

    def on_toggle_recursive(self):
        self.recursive = bool(self.var_recursive.get())
        if self.folder:
            self.reload_folder()

    def on_conf_change(self, _=None):
        self.conf = float(self.var_conf.get())
        self.refresh()

    def on_iou_change(self, _=None):
        self.iou = float(self.var_iou.get())
        self.refresh()

    def pick_model(self):
        path = filedialog.askopenfilename(
            title="選擇 YOLO 模型(.pt)",
            filetypes=[("YOLO model", "*.pt"), ("All files", "*.*")]
        )
        if not path:
            return
        if YOLO is None:
            messagebox.showerror("錯誤", "找不到 ultralytics。請先 pip install ultralytics")
            return
        try:
            self.model = YOLO(path)
            self.model_path = path
            self.update_info()
            self.refresh()
        except Exception as e:
            messagebox.showerror("載入模型失敗", str(e))

    def pick_folder(self):
        folder = filedialog.askdirectory(title="選擇影像資料夾")
        if not folder:
            return
        self.folder = folder
        self.reload_folder()

    def pick_json(self):
        path = filedialog.askopenfilename(
            title="選擇 per-class 門檻 JSON（可選）",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.thresholds = load_thresholds_json(path)
            self.th_json_path = path
            self.update_info()
            self.refresh()
        except Exception as e:
            messagebox.showerror("讀取 JSON 失敗", str(e))

    def reload_folder(self):
        self.images = list_images(self.folder, recursive=self.recursive)
        self.idx = 0
        if not self.images:
            messagebox.showwarning("提示", "此資料夾找不到影像檔。")
        self.update_info()
        self.refresh()

    def update_info(self, extra: str = ""):
        model_name = os.path.basename(self.model_path) if self.model_path else "(未選)"
        folder_name = self.folder if self.folder else "(未選)"
        json_name = os.path.basename(self.th_json_path) if self.th_json_path else "(未選)"
        count = len(self.images)
        pos = f"{self.idx + 1}/{count}" if count else "0/0"
        txt = (
            f"Model: {model_name}\n"
            f"Folder: {folder_name}  |  Images: {pos}  |  recursive={self.recursive}\n"
            f"JSON(threshold): {json_name}  |  conf={self.conf:.2f}, iou={self.iou:.2f}\n"
        )
        if extra:
            txt += extra
        self.info.config(text=txt)

    def prev_image(self):
        if not self.images:
            return
        self.idx = max(0, self.idx - 1)
        self.refresh()

    def next_image(self):
        if not self.images:
            return
        self.idx = min(len(self.images) - 1, self.idx + 1)
        self.refresh()

    def jump(self, delta: int):
        if not self.images:
            return
        self.idx = max(0, min(len(self.images) - 1, self.idx + delta))
        self.refresh()

    def goto_dialog(self):
        if not self.images:
            return
        win = tk.Toplevel(self.root)
        win.title("跳到指定張")
        win.geometry("260x110")
        tk.Label(win, text=f"輸入 1..{len(self.images)}").pack(pady=8)
        ent = tk.Entry(win)
        ent.pack()
        ent.focus_set()

        def ok():
            try:
                v = int(ent.get().strip()) - 1
                self.idx = max(0, min(len(self.images) - 1, v))
                win.destroy()
                self.refresh()
            except Exception:
                win.destroy()

        tk.Button(win, text="OK", command=ok).pack(pady=10)

    def refresh(self):
        if not self.images:
            self.canvas.config(image="", text="請先選擇資料夾", compound="center")
            self.update_info()
            return

        path = self.images[self.idx]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            self.update_info(extra="讀圖失敗，已自動下一張\n")
            self.next_image()
            return

        extra = ""
        dets: List[Det] = []
        ms = 0.0

        if self.model is not None:
            try:
                dets, ms = infer_ultralytics(self.model, img, conf=self.conf, iou=self.iou)
                passed, filtered = split_by_class_thresholds(dets, self.thresholds)
                img_draw = img
                # 綠框：通過門檻（NG）
                img_draw = draw_boxes(img_draw, passed, color=(0, 255, 0))
                # 黃框：未過門檻（OK 也顯示）
                if self.var_show_filtered.get():
                    img_draw = draw_boxes(img_draw, filtered, color=(0, 255, 255))
                result = "NG" if len(passed) > 0 else "OK"
                extra = f"File: {os.path.basename(path)} | infer={ms:.1f} ms | raw={len(dets)} pass={len(passed)} filt={len(filtered)} | RESULT={result}\n"
            except Exception as e:
                img_draw = img
                extra = f"Infer error: {e}\n"
        else:
            img_draw = img
            extra = f"File: {os.path.basename(path)} (未載入模型，僅顯示原圖)\n"

        self.update_info(extra=extra)

        # Convert BGR->RGB for PIL
        rgb = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)

        # Fit to window (keep aspect)
        w = self.root.winfo_width()
        h = self.root.winfo_height()

        # Reserve some height for top UI; estimate 220
        max_w = max(400, w - 40)
        max_h = max(300, h - 260)

        pil.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

        tk_img = ImageTk.PhotoImage(pil)
        self.canvas.image = tk_img
        self.canvas.config(image=tk_img, text="", compound="center")


def main():
    parser = argparse.ArgumentParser(description="YOLO Viewer Tkinter (no-save, folder browsing)")
    parser.add_argument("--model", default="", help="optional model path (.pt)")
    parser.add_argument("--folder", default="", help="optional image folder")
    parser.add_argument("--json", default="", help="optional per-class threshold json")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    root = tk.Tk()
    app = ViewerApp(root)

    app.conf = args.conf
    app.iou = args.iou
    app.var_conf.set(app.conf)
    app.var_iou.set(app.iou)

    if args.recursive:
        app.recursive = True
        app.var_recursive.set(True)

    # optional preload
    if args.model:
        if YOLO is None:
            messagebox.showerror("錯誤", "找不到 ultralytics。請先 pip install ultralytics")
        else:
            try:
                app.model = YOLO(args.model)
                app.model_path = args.model
            except Exception as e:
                messagebox.showerror("載入模型失敗", str(e))

    if args.json:
        try:
            app.thresholds = load_thresholds_json(args.json)
            app.th_json_path = args.json
        except Exception as e:
            messagebox.showerror("讀取 JSON 失敗", str(e))

    if args.folder:
        app.folder = args.folder
        app.images = list_images(args.folder, recursive=app.recursive)

    app.update_info()
    app.refresh()

    root.mainloop()


if __name__ == "__main__":
    main()
