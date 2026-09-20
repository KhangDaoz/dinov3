# DINOv3 Representation and Pair-wise Confidence for Fine-grained Image Retrieval

Repository nghiên cứu DINOv3 cho truy xuất ảnh fine-grained trên CUB-200-2011, tập trung vào lựa chọn biểu diễn và ước lượng độ tin cậy của kết quả retrieval. Backbone mục tiêu là DINOv3 ViT-B/16 pretrained và được đóng băng trong các thực nghiệm hiện tại.

```text
DINOv3 feature extraction
          │
          ├── E2A: Ablation 4 feature representation trên Development Set
          └── E2B: 4 nhánh Pair-wise Confidence và reranking độc lập
```

## Mục tiêu thực nghiệm

### E2A — Representation ablation

E2A là ablation study của bốn representation; không chọn trước M1 và không dùng
Final Test để chọn representation:

1. M1 — CLS token lớp cuối, kích thước 768 (không có module trainable chính).
2. M2 — mean pooling trên patch tokens (không có module trainable chính).
3. M3 — ghép CLS với mean patch rồi projection về không gian embedding.
4. M4 — attention pooling học trọng số trên patch tokens.

Mọi embedding được chuẩn hóa L2 trước khi tính cosine similarity. Register tokens không được đưa vào patch pooling.
M3 và M4 chỉ được train trên Development/Train và chọn checkpoint trên
Development/Validation. Sau bước này, cả M1–M4 đều được khóa và chuyển nguyên
vẹn sang E2B.

### E2B — Pair-wise confidence learning

E2B không dùng một representation thắng cuộc từ E2A. Thay vào đó, E2B huấn
luyện bốn Pair-wise Confidence Network độc lập cho M1–M4. Kiến trúc mạng,
sampling protocol, optimizer, quy tắc chọn checkpoint và không gian
hyperparameter phải giống nhau giữa bốn nhánh, để biến độc lập duy nhất là
representation.

Positive pair gồm hai ảnh khác nhau cùng lớp; negative pair gồm hai ảnh khác
lớp. Cả hai đầu pair huấn luyện đều phải thuộc Development/Train. MLP nhận đầu
vào `[feature_a, feature_b, |feature_a - feature_b|]`, dự đoán confidence trong
`[0, 1]` và được huấn luyện bằng Binary Cross Entropy.

Điểm xếp hạng cuối:

```text
final_score = lambda * cosine_similarity + (1 - lambda) * pair_confidence
```

Cần khảo sát nhiều giá trị `lambda` trên Development/Validation và ablation
riêng cosine similarity, pair confidence, cùng phép fusion. Chỉ sau khi khóa
checkpoint và `lambda` của cả bốn nhánh mới được chạy Final Test.

Protocol mới đầy đủ nằm tại
[E2A–E2B new protocol](docs/experiments/e2a-e2b-new.md). Các kế hoạch cũ trong
`docs/plan/` mô tả protocol test-based selection trước đây và không được dùng
để lựa chọn model cho lần chạy mới.

## Dữ liệu và protocol đánh giá

Đặt CUB-200-2011 tại `data/CUB_200_2011/`, giữ thư mục `images/` và các manifest gốc như `images.txt`, `image_class_labels.txt`.

- **Development Set:** classes 0–99, 5.864 ảnh theo dữ liệu hiện tại. Trong
  từng class, chia có seed 80% ảnh vào Train và 20% vào Validation; hai tập có
  cùng classes nhưng không trùng image ID.
- **Final Unseen Test Set:** classes 100–199, 5.924 ảnh theo dữ liệu hiện tại;
  giữ nguyên toàn bộ và không đọc kết quả trước khi khóa mọi representation,
  checkpoint, kiến trúc, threshold, `lambda` và hyperparameter.
- Giữ class boundary 0–99/100–199; không thay bằng image split trong
  `train_test_split.txt`.
- Khi đánh giá retrieval, mỗi split đồng thời làm query và gallery; loại
  self-match theo image ID trước khi lấy top-K.
- Giữ nguyên RGB preprocessing và processor của checkpoint giữa các phương pháp.
- Giữ cố định cosine similarity, tie policy và cách tính `Recall@1`,
  `Recall@2`, `Recall@4`, `Recall@8` giữa mọi nhánh.

## Quy trình thực nghiệm

1. Chuẩn bị dataset; tạo và lưu manifest Development/Train 80% và
   Development/Validation 20% theo từng class.
2. Trích xuất, cache CLS/register/patch tokens bằng frozen DINOv3.
3. Chạy E2A cho M1–M4; train M3/M4 trên Train, chọn checkpoint trên Validation,
   rồi khóa cả bốn representation.
4. Với từng M1–M4, tạo pair trên Train, huấn luyện Pair-wise Confidence
   Network và chọn checkpoint/hyperparameter/`lambda` trên Validation.
5. Khóa toàn bộ bốn pipeline, sau đó mới chạy một lần trên Final Test classes
   100–199 cho ba chế độ: cosine, pairwise confidence và fusion.
6. Lưu config đã resolve, seed, split manifest, checkpoint, embedding,
   confidence, Recall@K, ranking và training log.

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
│   ├── cub_e2a_m1.yaml
│   ├── cub_e2a_m2.yaml
│   ├── cub_e2a_m3.yaml
│   ├── cub_e2a_m4.yaml
│   ├── cub_e2b_m1.yaml
│   ├── cub_e2b_m2.yaml
│   ├── cub_e2b_m3.yaml
│   └── cub_e2b_m4.yaml
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
    ├── e2a_cls/
    ├── e2a_mean_patch/
    ├── e2a_fusion/
    ├── e2a_attention_pool/
    └── e2b_pair_confidence/
        ├── m1/
        ├── m2/
        ├── m3/
        └── m4/
```

Chỉ thêm module dự kiến khi đã có chức năng hoạt động. Chi tiết từng experiment nằm tại [`docs/experiments/`](docs/experiments/).

## Kết quả và khả năng tái lập

Mỗi run phải lưu config đã resolve, seed (mặc định `42`), split manifest,
checkpoint, embedding và bảng Recall@K trong thư mục riêng dưới `outputs/`.
Từng nhánh E2B lưu confidence score và giá trị `lambda`. Ghi rõ checkpoint
DINOv3, phiên bản dependency, thay đổi so với phương pháp tham khảo và tách
biệt reproduction khỏi DINOv3 adaptation hoặc ablation.

Gói kết quả E2A–E2B phải có tối thiểu:

- `cosine_results.csv`, `pairwise_results.csv`, `fusion_results.csv`, mỗi file
  có các hàng M1–M4 và cột R@1/R@2/R@4/R@8;
- checkpoint M3, checkpoint M4 và bốn checkpoint Pair-wise Confidence;
- training log, seed và configuration đủ để tái lập.

Không được dùng Final Test để chọn M1–M4, checkpoint, kiến trúc, threshold,
`lambda` hoặc bất kỳ hyperparameter nào. Mục tiêu là đo representation nào phù
hợp nhất cho cosine retrieval và Pair-wise Confidence Learning dưới cùng một
protocol, không phải chứng minh trước rằng M1 tốt nhất.

Repository nhắm tới môi trường Python 3.11 và 2× NVIDIA T4. Các workload CUDA phù hợp nên dùng cả hai GPU qua `torchrun`/DistributedDataParallel và mixed precision khi bảo đảm ổn định số học. Dataset, checkpoint và nội dung trong `outputs/` không được commit.
