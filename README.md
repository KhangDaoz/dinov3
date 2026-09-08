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

M1/M2 không cần training loop. Các thành phần học được của M3/M4 được huấn luyện và chọn tham số trên dữ liệu train; không dùng test để chọn tham số. Uncertainty và reranking nằm ngoài phạm vi E2A.

## Cấu trúc repository

Cấu trúc chung của dự án. `train_projection.py` và các thư mục kết quả cho phương pháp chưa triển khai được giữ trong thiết kế; chỉ tạo khi bắt đầu công việc tương ứng.

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

## Môi trường

Sử dụng Python 3.11. Các dependency trực tiếp được ghi phiên bản trong `requirements.txt`.

Thiết lập môi trường mới:

```bash
conda create -n e2a python=3.11 pip
conda activate e2a
python -m pip install -r requirements.txt
python -m pip check
```

Nếu đã có môi trường `e2a` thì chỉ cần kích hoạt. `requirements.txt` chốt các dependency trực tiếp, chưa phải lock toàn bộ dependency bắc cầu. Chọn `device` trong cấu hình phù hợp với phần cứng; cấu hình mặc định dùng CPU.

Checkpoint đang dùng: `facebook/dinov3-vitb16-pretrain-lvd1689m`. Máy mới cần quyền truy cập checkpoint trên Hugging Face và đăng nhập bằng `hf auth login` nếu được yêu cầu. Không lưu token trong repo. Chỉ bật `HF_HUB_OFFLINE=1` khi đã có đầy đủ checkpoint trong cache.

## Quy trình sử dụng

Quy trình chung: chuẩn bị dữ liệu và môi trường → xác nhận preprocessing/backbone → huấn luyện thành phần học được nếu phương pháp yêu cầu → trích xuất embedding → đánh giá retrieval → so sánh Recall@K.

Các lệnh hiện tại, chạy từ thư mục gốc repo trong môi trường `e2a`:

```bash
conda activate e2a

# Chạy thực nghiệm theo cấu hình: extract → lưu cache → evaluate → in metrics
HF_HUB_OFFLINE=1 python scripts/run_e2a.py --config configs/cub_e2a.yaml

# Với cache đã có: kiểm tra cache rồi đánh giá lại, không chạy backbone
HF_HUB_OFFLINE=1 python scripts/run_e2a.py --config configs/cub_e2a.yaml --reuse-cache

# Chạy riêng từng bước
HF_HUB_OFFLINE=1 python scripts/extract_features.py --config configs/cub_e2a.yaml
HF_HUB_OFFLINE=1 python scripts/evaluate.py --config configs/cub_e2a.yaml

# Kiểm tra dữ liệu/model/metric
HF_HUB_OFFLINE=1 python -m src.dataset --full
HF_HUB_OFFLINE=1 python -m src.dinov3_backbone
python -m src.retrieval --self-test
```

Cấu hình gồm `model_name`, `model_revision` (commit SHA cố định), `representation`,
`data_root`, `split`, `batch_size`, `num_workers`, `device`, `recall_k`,
`retrieval_chunk_size`, `output_dir`, `seed`, `num_threads`. Đường dẫn dữ liệu/output
tương đối được resolve từ gốc repo. Hiện chỉ hỗ trợ `representation: cls`,
`split: [train, eval]`.

Chạy mới sẽ dừng nếu đích đã có artifacts. Muốn trích xuất lại và thay thế kết quả,
truyền `--overwrite` cho `run_e2a.py` hoặc `extract_features.py`.
`--reuse-cache` và `--overwrite` loại trừ nhau. Đánh giá lại chỉ cập nhật phần đánh giá
trong `metrics.json`, giữ metadata trích xuất và các file `.pt`.

Cache được kiểm tra model/revision/representation, processor, checksum SHA-256,
shape/dtype/norm và nhãn/đường dẫn theo dataset trước khi sử dụng.
Thiếu metadata hoặc cache không khớp sẽ bị từ chối; không tự suy đoán nguồn cache.
Seed được đặt cho Python, NumPy và PyTorch trước khi chạy mới.

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
