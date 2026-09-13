# Uncertainty-aware DINOv3 for Fine-grained Image Retrieval

Repository nghiên cứu DINOv3 cho truy xuất ảnh fine-grained trên CUB-200-2011, tập trung vào lựa chọn biểu diễn và ước lượng độ tin cậy của kết quả retrieval. Backbone mục tiêu là DINOv3 ViT-B/16 pretrained và được đóng băng trong các thực nghiệm hiện tại.

```text
DINOv3 feature extraction
          │
          ├── E1: Evidential uncertainty và image-level reranking
          ├── E2A: Phân tích feature representation
          └── E2B: Pair-wise confidence và reranking (dùng representation tốt nhất từ E2A)
```

## Mục tiêu thực nghiệm

### E1 — Evidential uncertainty

E1 xây dựng hai nhánh từ DINOv3: embedding từ CLS token để tính cosine similarity và Evidential Head để ước lượng uncertainty ở mức ảnh. Thực nghiệm so sánh baseline DINOv3 + cosine với uncertainty-aware reranking, đồng thời phân tích các truy xuất sai để đánh giá liệu image-level uncertainty có phản ánh retrieval correctness hay không.

### E2A — Feature representation analysis

E2A lựa chọn representation tốt nhất khi chưa sử dụng uncertainty hoặc reranking. Bốn phương pháp cần so sánh là:

1. CLS token lớp cuối, kích thước 768 — baseline.
2. Mean pooling trên patch tokens.
3. Ghép CLS với mean patch rồi projection về không gian embedding.
4. Attention pooling học trọng số trên patch tokens.

Mọi embedding được chuẩn hóa L2 trước khi tính cosine similarity. Register tokens không được đưa vào patch pooling.

### E2B — Pair-wise confidence learning

E2B dùng representation tốt nhất từ E2A và học độ tin cậy trực tiếp cho cặp `(query, candidate)`. Positive pair gồm hai ảnh cùng lớp; negative pair gồm hai ảnh khác lớp. MLP nhận đầu vào `[feature_a, feature_b, |feature_a - feature_b|]`, dự đoán confidence trong `[0, 1]` và được huấn luyện bằng Binary Cross Entropy.

Điểm xếp hạng cuối:

```text
final_score = lambda * cosine_similarity + (1 - lambda) * pair_confidence
```

Cần khảo sát nhiều giá trị `lambda` và ablation riêng cosine similarity, pair confidence, cùng phép fusion.

## Dữ liệu và protocol đánh giá

Đặt CUB-200-2011 tại `data/CUB_200_2011/`, giữ thư mục `images/` và các manifest gốc như `images.txt`, `image_class_labels.txt`.

- Train: 100 lớp đầu, nhãn 0–99; 5.864 ảnh theo dữ liệu hiện tại.
- Eval/test: 100 lớp cuối, nhãn 100–199; 5.924 ảnh theo dữ liệu hiện tại.
- Chia theo lớp; không thay thế bằng image split trong `train_test_split.txt`.
- Eval/test đồng thời làm query và gallery; loại self-match trước khi lấy top-K.
- Giữ nguyên RGB preprocessing và processor của checkpoint giữa các phương pháp.
- Báo cáo `Recall@1`, `Recall@2`, `Recall@4` và `Recall@8` với cosine similarity.

## Quy trình thực nghiệm

1. Chuẩn bị dataset và kiểm tra class split.
2. Trích xuất, cache CLS/register/patch tokens bằng frozen DINOv3.
3. Chạy E2A và chọn representation theo Recall@K trên protocol cố định.
4. Huấn luyện, đánh giá E1 và E2B; không dùng nhãn eval/test để tạo pair hoặc chọn hyperparameter.
5. Lưu config, seed, checkpoint, embedding, uncertainty/confidence, Recall@K và failure cases.

Các entry point dự kiến nhận config theo dạng:

```bash
python scripts/run_<experiment>.py --config configs/<config_name>.yaml
```

Tên script và config phải tương ứng với thực nghiệm; chỉ sử dụng lệnh này sau khi module liên quan đã được triển khai.

## Cấu trúc repository dự kiến

```text
.
├── AGENTS.md
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── cub_e1.yaml
│   ├── cub_e2a_m1.yaml
│   ├── cub_e2a_m2.yaml
│   ├── cub_e2a_m3.yaml
│   ├── cub_e2a_m4.yaml
│   └── cub_e2b.yaml
├── data/                              # Không commit
│   └── CUB_200_2011/
├── src/uncertainty_retrieval/
│   ├── data/                          # CUB loader và patch cache
│   ├── models/                        # DINOv3, representations và confidence heads
│   ├── training/                      # Training loops
│   ├── evaluation/                    # Retrieval và Recall@K
│   └── sampling/                      # Pair sampling
├── scripts/                           # Entry points cho extract/train/evaluate
├── tests/
│   ├── unit/
│   └── integration/
├── docs/
│   ├── experiments/                   # Mục tiêu và protocol từng thực nghiệm
│   └── plan/                          # Kế hoạch triển khai
├── reports/                           # Báo cáo LaTeX
└── outputs/                           # Không commit
    ├── e1_evidential/
    ├── e2a_cls/
    ├── e2a_mean_patch/
    ├── e2a_fusion/
    ├── e2a_attention_pool/
    └── e2b_pair_confidence/
```

Chỉ thêm module dự kiến khi đã có chức năng hoạt động. Chi tiết từng experiment nằm tại [`docs/experiments/`](docs/experiments/).

## Kết quả và khả năng tái lập

Mỗi run phải lưu config đã resolve, seed (mặc định `42`), checkpoint, embedding và bảng Recall@K trong thư mục riêng dưới `outputs/`. E1 lưu thêm uncertainty score; E2B lưu confidence score và giá trị `lambda`. Ghi rõ checkpoint DINOv3, phiên bản dependency, thay đổi so với phương pháp tham khảo và tách biệt reproduction khỏi DINOv3 adaptation hoặc ablation.

Repository nhắm tới môi trường Python 3.11 và 2× NVIDIA T4. Các workload CUDA phù hợp nên dùng cả hai GPU qua `torchrun`/DistributedDataParallel và mixed precision khi bảo đảm ổn định số học. Dataset, checkpoint và nội dung trong `outputs/` không được commit.
