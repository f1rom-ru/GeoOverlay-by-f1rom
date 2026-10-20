"""
F6 -> скриншот экрана -> GeoCLIP предсказывает координаты -> оверлей в левом верхнем углу,
      координаты самого вероятного варианта копируются в буфер обмена ("lat, lon").
F7 -> скрыть оверлей, Esc -> выход.

Установка (Windows, запускать от имени администратора, т.к. нужна библиотека keyboard):
    pip install geoclip mss keyboard pillow

Что ускорено по сравнению с первой версией:
  * признаки 100 000 точек галереи считаются ОДИН раз (и кешируются в файл),
    а не при каждом нажатии F6;
  * скриншот не пишется на диск, а сразу уменьшается до размера входа модели;
  * модель прогревается при старте;
  * если есть видеокарта с CUDA, используется она (fp16).

Совет: в GeoGuessr включите оконный/безрамочный режим,
иначе оверлей tkinter может не отображаться поверх игры.
"""
import math
import os
import queue
import subprocess
import sys
import threading


def _crash_hook(exc_type, exc, tb):
    """В собранном .exe не закрываем окно сразу при ошибке, чтобы её можно было прочитать."""
    import traceback
    traceback.print_exception(exc_type, exc, tb)
    try:
        input("\nОшибка. Нажмите Enter, чтобы закрыть окно...")
    except Exception:
        pass


if getattr(sys, "frozen", False):
    sys.excepthook = _crash_hook

import time
import tkinter as tk

import keyboard
import mss
import torch
import torch.nn.functional as F
from geoclip import GeoCLIP
from PIL import Image

TOP_K = 3                 # сколько вариантов показывать
INPUT_SIZE = 224          # размер входа CLIP
BASE_SIDE = 448           # короткая сторона базовой картинки, из которой режутся ракурсы
TIME_BUDGET = 2.5         # сек: сколько максимум "думать" (берём столько ракурсов, сколько успеем)
CLUSTER_KM = 100          # точки ближе этого радиуса считаются одним и тем же местом
N_CAND = 1000             # сколько самых вероятных точек галереи кластеризуем
EARTH_KM = 6371.0
# В собранном .exe кеш лежит рядом с exe, а не во временной папке распаковки
BASE_DIR = (os.path.dirname(sys.executable) if getattr(sys, "frozen", False)
            else os.path.dirname(os.path.abspath(__file__)))
CACHE_FILE = os.path.join(BASE_DIR, "geoclip_loc_feats.pt")

torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Загрузка модели... (устройство: {device})")
model = GeoCLIP().to(device).eval()
gps_gallery = model.gps_gallery  # (100000, 2)


def build_location_features():
    """Нормализованные признаки всех точек галереи. Считаем один раз."""
    if os.path.exists(CACHE_FILE):
        try:
            feats = torch.load(CACHE_FILE, map_location=device, weights_only=True)
            if feats.shape[0] == gps_gallery.shape[0]:
                return feats
        except Exception:
            pass
    print("Первый запуск: считаю признаки галереи (разово, до минуты)...")
    chunks = []
    with torch.inference_mode():
        for i in range(0, gps_gallery.shape[0], 5000):
            part = model.location_encoder(gps_gallery[i:i + 5000])
            chunks.append(F.normalize(part, dim=1))
    feats = torch.cat(chunks)
    try:
        torch.save(feats.cpu(), CACHE_FILE)
    except Exception:
        pass
    return feats.to(device)


loc_feats = build_location_features()          # (100000, 512)
logit_scale = model.logit_scale.exp().to(device)


# Единичные 3D-векторы точек галереи: по ним быстро считаются расстояния между точками
_lat = torch.deg2rad(gps_gallery[:, 0])
_lon = torch.deg2rad(gps_gallery[:, 1])
gallery_xyz = torch.stack(
    [torch.cos(_lat) * torch.cos(_lon), torch.cos(_lat) * torch.sin(_lon), torch.sin(_lat)],
    dim=1,
)
COS_R = math.cos(CLUSTER_KM / EARTH_KM)


def view_boxes(w, h):
    """Ракурсы (рамки обрезки), от самых важных к менее важным.
    Одна центральная обрезка теряет края кадра, поэтому берём ещё левый/правый края,
    весь кадр целиком и приближенные куски: так видно больше подсказок."""
    s = min(w, h)

    def square(f, size=1.0):
        side = s * size
        if w >= h:
            cx, cy = side / 2 + (w - side) * f, h / 2
        else:
            cx, cy = w / 2, side / 2 + (h - side) * f
        return (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)

    boxes = [
        square(0.5),            # центр
        (0, 0, w, h),           # весь кадр целиком (сжатый до квадрата)
        square(0.0),            # левый край
        square(1.0),            # правый край
        square(0.5, 0.6),       # приближенный центр
        square(0.25),
        square(0.75),
        square(0.0, 0.6),       # приближенные левый и правый края
        square(1.0, 0.6),
    ]
    uniq = []
    for b in boxes:
        r = tuple(round(v) for v in b)
        if r not in uniq:
            uniq.append(r)
    return uniq


def view_probs(view):
    """Распределение вероятностей по 100K точкам галереи для одного ракурса."""
    x = model.image_encoder.preprocess_image(view).to(device)
    if device == "cuda":
        with torch.autocast("cuda", dtype=torch.float16):
            f = model.image_encoder(x)
    else:
        f = model.image_encoder(x)
    f = F.normalize(f.float(), dim=1)
    return (logit_scale * (f @ loc_feats.t())).softmax(dim=-1)[0]


def cluster_top(probs):
    """Вместо одной самой вероятной точки ищем самое вероятное МЕСТО: суммируем вероятность
    всех точек в радиусе CLUSTER_KM и берём центр масс лучшего скопления. Так одинокий
    случайный выброс проигрывает району, на который "голосуют" многие точки и ракурсы."""
    p, idx = torch.topk(probs, N_CAND)
    xyz = gallery_xyz[idx]
    close = (xyz @ xyz.t()) >= COS_R
    closef = close.float()
    alive = torch.ones(N_CAND, dtype=torch.bool, device=p.device)
    out = []
    for _ in range(TOP_K):
        if not bool(alive.any()):
            break
        mass = (closef @ (p * alive)).masked_fill(~alive, -1.0)
        c = int(mass.argmax())
        members = close[c] & alive
        wts = p * members
        v = (xyz * wts[:, None]).sum(0)
        v = v / v.norm()
        lat = math.degrees(math.asin(float(v[2].clamp(-1, 1))))
        lon = math.degrees(math.atan2(float(v[1]), float(v[0])))
        out.append((lat, lon, float(wts.sum())))
        alive = alive & ~members
    return out


def predict_image(img: Image.Image, budget=TIME_BUDGET):
    """PIL-картинка -> ([(lat, lon, prob), ...], число использованных ракурсов).
    Берём ракурсы по очереди, пока укладываемся в budget секунд, и усредняем вероятности."""
    t0 = time.perf_counter()
    w, h = img.size
    k = BASE_SIDE / min(w, h)
    base = img.resize((max(1, round(w * k)), max(1, round(h * k))),
                      Image.BICUBIC, reducing_gap=2.0)
    total, n = None, 0
    with torch.inference_mode():
        for box in view_boxes(*base.size):
            elapsed = time.perf_counter() - t0
            if n and elapsed + elapsed / n > budget:   # следующий ракурс уже не влезет
                break
            view = base.crop(box).resize((INPUT_SIZE, INPUT_SIZE), Image.BICUBIC)
            p = view_probs(view)
            total = p if total is None else total + p
            n += 1
        return cluster_top(total / n), n


# Прогрев: первый прогон всегда медленнее остальных
predict_image(Image.new("RGB", (640, 360)), budget=0)
print("Готово. F6 - определить место, F7 - скрыть, Esc - выход.")

q = queue.Queue()
busy = False

root = tk.Tk()
root.overrideredirect(True)          # без рамки
root.attributes("-topmost", True)    # поверх всех окон
root.attributes("-alpha", 0.85)
root.geometry("+10+10")              # левый верхний угол
label = tk.Label(
    root,
    text="F6 - определить место",
    font=("Consolas", 14),
    fg="white",
    bg="black",
    justify="left",
    padx=8,
    pady=4,
)
label.pack()


def copy_to_clipboard(text):
    """Копирование в системный буфер обмена. На Windows - через clip.exe."""
    try:
        if os.name == "nt":
            r = subprocess.run(
                ["clip"],
                input=text.encode("ascii"),   # цифры, запятая, точка, пробел
                creationflags=subprocess.CREATE_NO_WINDOW,
                timeout=5,
            )
            return r.returncode == 0
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        return True
    except Exception:
        return False


def worker(img):
    global busy
    try:
        t0 = time.perf_counter()
        res, n_views = predict_image(img)
        dt = time.perf_counter() - t0
        lines = [f"{lat:.5f}, {lon:.5f}  ({p * 100:.0f}%)" for lat, lon, p in res]
        lines.append(f"{dt:.1f} c, ракурсов: {n_views}")
        best_lat, best_lon, _ = res[0]  # topk отсортирован: первый - самый вероятный
        ok = copy_to_clipboard(f"{best_lat:.6f}, {best_lon:.6f}")
        lines.append("(скопировано)" if ok else "(не удалось скопировать)")
        q.put(("result", "\n".join(lines)))
    except Exception as e:
        q.put(("result", f"Ошибка: {e}"))
    finally:
        busy = False


def take_shot():
    """Скриншот без самого оверлея, затем предсказание в отдельном потоке."""
    global busy
    if busy:
        return
    busy = True
    root.withdraw()          # прячем оверлей, чтобы он не попал в кадр
    root.update()
    root.after(40, _grab)    # даём окну исчезнуть


def _grab():
    with mss.mss() as sct:
        shot = sct.grab(sct.monitors[1])  # основной монитор, прямо в память
    img = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
    label.config(text="Анализ...")
    root.deiconify()
    threading.Thread(target=worker, args=(img,), daemon=True).start()


def poll():
    try:
        while True:
            msg = q.get_nowait()
            if msg == "shot":
                take_shot()
            elif msg == "hide":
                root.withdraw()
            elif msg == "quit":
                root.destroy()
                return
            elif isinstance(msg, tuple) and msg[0] == "result":
                label.config(text=msg[1])
                root.deiconify()
    except queue.Empty:
        pass
    root.after(50, poll)


keyboard.add_hotkey("f6", lambda: q.put("shot"))
keyboard.add_hotkey("f7", lambda: q.put("hide"))
keyboard.add_hotkey("esc", lambda: q.put("quit"))

root.after(50, poll)
root.mainloop()
