# Donut

　テンソルの入力画像は(height, width) = (1280, 960)（高さ1280px × 幅960px）で固定されているらしい。

モデル入力の以下の部分で自動的にリサイズを行っているようだ。

    pixel_values = processor(image, return_tensors="pt").pixel_values


入力画像はリサイズされることを考慮して、同様のアスペクト（４：３）で入力するほうが良い。
    
    dOCUMENT SCANNER.PY
    
にて自動的に画像整形を行う。

　そのまますると時々見切れたので、


