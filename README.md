# capture_human_detection
リアルタイムな人検出

## ファイル構成

### メインファイル
- **`cap_human_detection.py`** - リアルタイム人検出のメインスクリプト。カメラからの映像に対してリアルタイムで人検出を行い、メインフレームとRGB各チャンネル（赤・緑・青）の4つのウィンドウに検出結果を表示
- **`config.py`** - モデル設定ファイル。検出閾値、バッチサイズ、エポック数などのパラメータとFCOS ResNet-50 FPNモデルの構築を定義

### 学習・評価関連
- **`train.py`** - モデル学習用スクリプト。アノテーション付きデータセットを使用してFCOSモデルの学習を実行
- **`evaluation_model.py`** - 学習済みモデルの評価スクリプト。IoU計算による精度評価とメトリクス算出
- **`load_dataset_annot.py`** - アノテーション付きデータセットの読み込み。PyTorchのDatasetクラスを継承したカスタムデータローダー

### 推論・予測関連
- **`predict_1img.py`** - 単一画像に対する人検出。1枚の画像ファイルを入力として検出結果を出力
- **`predict_dir_imgs.py`** - ディレクトリ内の複数画像に対する一括人検出処理
- **`predict_crop_mp4.py`** - 動画ファイル（MP4）に対する人検出。指定した領域をクロップして検出処理を実行

### データ処理関連
- **`images_human_crop.py`** - MediaPipeを使用した人姿勢検出によるデータフィルタリング。人が写っている画像のみを抽出

### PyTorch検出ユーティリティ（pyt_det/）
- **`engine.py`** - 学習・評価エンジン。訓練ループと評価ループの実装
- **`transforms.py`** - データ拡張パイプライン。学習時の画像変換処理
- **`utils.py`** - 分散学習用ユーティリティ関数とメトリクスロガー
- **`coco_eval.py`** - COCO形式での評価メトリクス計算
- **`coco_utils.py`** - COCOデータセット統合とフォーマット変換

## 使用方法

### リアルタイム検出
```bash
python cap_human_detection.py [model_path]
```

### 単一画像検出
```bash
python predict_1img.py [model_path] [image_path]
```

### ディレクトリ一括検出
```bash
python predict_dir_imgs.py [model_path] [directory_path]
```

### モデル学習
```bash
python train.py
```

## 特徴
- FCOS ResNet-50 FPNベースの高精度人検出
- リアルタイム処理対応
- RGB各チャンネル別の検出結果表示
- COCO形式データセット対応
- 分散学習サポート
