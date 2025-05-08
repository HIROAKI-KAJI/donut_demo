import flet as ft
import cv2
import threading
import base64
import io
import re
from PIL import Image
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

from document_scanner import DocumentScanner

# Donut モデルとプロセッサの読み込み
processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

sccanner = DocumentScanner(output_size=(960, 1280))  # スキャン後の画像サイズを指定
last_sccan_frame = None  # スキャンした画像を保持する変数

# 推論関数
def run_ocr(image: Image.Image) -> str:
    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    pixel_values = processor(image, return_tensors="pt").pixel_values

    outputs = model.generate(
        pixel_values.to(device),
        decoder_input_ids=decoder_input_ids.to(device),
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    sequence = processor.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", sequence, count=1).strip()
    return str(processor.token2json(sequence))

# WebカメラのキャプチャとUI更新
def live_camera_update(img_control, img_control_sccand, cap, page):
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # OpenCV BGR -> RGB
        #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 横が広い画像であり、Donutの入力は縦長画像。よって90度回転させる。
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        sccan_statas, sccand_frame = sccanner.scan(frame)
        # スキャンの精度が悪いので最後にスキャンできたものを保持しておく
        if sccan_statas:
            global last_sccan_frame
            last_sccan_frame = sccand_frame
            _, buffer = cv2.imencode('.jpg', last_sccan_frame)
            img_data = base64.b64encode(buffer).decode("utf-8")
            img_control_sccand.src_base64 = img_data
            img_control_sccand.update()

        _, buffer = cv2.imencode('.jpg', frame)
        img_data = base64.b64encode(buffer).decode("utf-8")
        img_control.src_base64 = img_data
        img_control.update()
        page.update()

# Flet アプリのメイン関数
def main(page: ft.Page):
    page.title = "Donut OCR + Webcam"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.scroll = True

    # カメラ映像と推論結果表示
    img_control = ft.Image(width=480, height=360)
    img_control_sccand = ft.Image(width=480, height=360)
    result_text = ft.Text("OCR結果がここに表示されます", selectable=True)

    # キャプチャして推論する
    def check_and_run(e):
        global last_sccan_frame
        if last_sccan_frame is None:
            result_text.value = "スキャンされた画像がありません。"
            page.update()
            return

        # OpenCV BGR → RGB → PIL.Image
        pil_image = Image.fromarray(last_sccan_frame)
        result_text.value = "推論中..."
        page.update()

        result = run_ocr(pil_image)
        result_text.value = result
        page.update()

    capture_button = ft.ElevatedButton("スキャンされたスナップショットに対してOCR", on_click=check_and_run)

    # UI レイアウト
    page.add(
        ft.Row(
            [
                ft.Column([img_control], alignment=ft.MainAxisAlignment.CENTER),
                ft.Column([img_control_sccand, capture_button], alignment=ft.MainAxisAlignment.CENTER),
                ft.Column([result_text], alignment=ft.MainAxisAlignment.START, width=400)
            ],
            alignment=ft.MainAxisAlignment.CENTER
        )
    )

    # 別スレッドでカメラ映像更新
    threading.Thread(target=live_camera_update, args=(img_control, img_control_sccand, cap, page), daemon=True).start()

# OpenCV カメラ初期化
cap = cv2.VideoCapture(0)  # 0 はデフォルトカメラ

if __name__ == "__main__":
    ft.app(target=main, view=ft.WEB_BROWSER)
