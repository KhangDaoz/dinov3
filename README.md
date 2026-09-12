# Uncertainty-aware DINOv3 for Fine-grained Image Retrieval

Repository nghiên cứu truy hồi ảnh fine-grained trên CUB-200-2011 với backbone DINOv3 ViT-B/16 pretrained. Giai đoạn đầu, **E2A — Feature Representation Analysis**, so sánh các cách tạo embedding để chọn representation trước khi phát triển Pair-wise Confidence Learning.

## Phạm vi thực nghiệm

E2A giữ frozen toàn bộ backbone, chuẩn hóa L2 embedding và dùng cosine similarity để đánh giá Recall@1/2/4/8. Các phương pháp dùng chung protocol dữ liệu, preprocessing và evaluator để so sánh nhất quán.

| Giai đoạn | Representation | Cách tạo embedding | Thư mục kết quả |
|---|---|---|---|
| M1 | CLS baseline | CLS token lớp cuối, 768 chiều | `outputs/cls/` |
| M2 | Mean Patch | Trung bình patch tokens, không gồm CLS/register tokens | `outputs/mean_patch/` |
| M3 | CLS + Mean Patch Fusion | Ghép CLS và Mean Patch, projection về không gian embedding | `outputs/fusion/` |
| M4 | Attention Pooling | Học trọng số tổng hợp patch tokens | `outputs/attention_pool/` |

## Cấu trúc repository

```text
.
├── configs/
│   └── cub_e2a.yaml
├── data/
│   └── CUB_200_2011/
├── src/
│   ├── dataset.py
│   ├── dinov3_backbone.py
│   ├── representations.py
│   ├── retrieval.py
│   ├── train_projection.py
│   └── utils.py
├── scripts/
│   ├── extract_features.py
│   ├── evaluate.py
│   └── run_e2a.py
├── outputs/
│   ├── cls/
│   ├── mean_patch/
│   ├── fusion/
│   └── attention_pool/
└── requirements.txt
```

| Thành phần | Trách nhiệm |
|---|---|
| `configs/cub_e2a.yaml` | Cấu hình dữ liệu, backbone, representation, trích xuất và đánh giá |
| `src/dataset.py` | Đọc CUB, chia split, cung cấp ảnh và nhãn |
| `src/dinov3_backbone.py` | Load processor/backbone, freeze và lấy output tokens |
| `src/representations.py` | Tạo embedding CLS, Mean Patch, Fusion và Attention Pooling |
| `src/retrieval.py` | Cosine retrieval, loại self-match và tính Recall@K |
| `src/train_projection.py` | Huấn luyện thành phần học được cho các representation cần training |
| `src/utils.py` | Tiện ích thực sự dùng chung như seed, đọc cấu hình và lưu/tải kết quả |
| `scripts/extract_features.py` | Trích xuất và lưu embedding theo representation |
| `scripts/evaluate.py` | Đánh giá từ embedding đã lưu |
| `scripts/run_e2a.py` | Điều phối thực nghiệm theo cấu hình |

Giữ cấu trúc gọn, ưu tiên bổ sung vào đúng file có trách nhiệm tương ứng; không tự thêm wrapper, class trung gian hoặc module ngoài cấu trúc đã thống nhất. Chỉ tạo file khi triển khai chức năng cần dùng, không tạo file rỗng để lấp đầy cây thư mục.

## Dữ liệu và protocol

Đặt CUB-200-2011 tại `data/CUB_200_2011/`, giữ thư mục `images/` cùng các manifest gốc, gồm `images.txt` và `image_class_labels.txt`.

- Train: 100 lớp đầu, nhãn 0–99; dữ liệu hiện tại có 5.864 ảnh.
- Eval/test: 100 lớp cuối, nhãn 100–199; dữ liệu hiện tại có 5.924 ảnh.
- Chia theo lớp, không dùng cách chia ảnh trong `train_test_split.txt` thay thế protocol này.
- Đọc ảnh RGB và dùng processor đi kèm checkpoint; giữ preprocessing cố định khi so sánh các phương pháp.
- Eval/test làm cả query và gallery; loại chính ảnh query trước khi lấy top-K.

## Quy ước kết quả

Mỗi representation lưu kết quả trong thư mục riêng với cùng cấu trúc:

```text
outputs/<representation>/
├── train_embeddings.pt
├── test_embeddings.pt
├── labels.pt
└── metrics.json
```

- `train_embeddings.pt`, `test_embeddings.pt`: tensor float32 trên CPU, shape `[N, D]`, chuẩn hóa L2; `D` theo representation. Tên `test` tương ứng split `eval` trong loader.
- `labels.pt`: dictionary có hai khóa `train`, `test`; mỗi khóa chứa tensor `labels` và danh sách `paths`, cùng thứ tự hàng với embedding tương ứng.
- `metrics.json`: `status`, `extraction` (cấu hình/processor/runtime/device/dtype/thời gian trích xuất), `cache_sha256`; sau evaluate có thêm `config` đã resolve và `evaluation` (Recall@1/2/4/8 dạng 0–1, số query/gallery, self-exclusion/tie policy, runtime và thời gian đánh giá). Chạy riêng extract đặt `status: extracted`; chạy đầy đủ đặt `status: evaluated`.

Bảng so sánh cần ghi representation, dimension, preprocessing, số ảnh và Recall@K. Giữ metadata đầy đủ để có thể đánh giá lại từ cache.

## Tài liệu

- [Đặc tả thực nghiệm E2A](docs/e2a.md)

README mô tả mục tiêu, cấu trúc và cách sử dụng chung của repository. Tiến độ,
kết quả từng giai đoạn và lịch sử kiểm chứng được ghi trong kế hoạch, không đưa vào README.
