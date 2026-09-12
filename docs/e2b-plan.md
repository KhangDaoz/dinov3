# Kế hoạch E2B — Pair-wise Confidence Learning

## 1. Mục tiêu

E2B học trực tiếp xác suất hai ảnh cùng lớp thay vì suy ra độ tin cậy từ uncertainty của từng ảnh. Thực nghiệm phải tách rõ chất lượng của CLS cosine, Pair Confidence độc lập, phép kết hợp hai tín hiệu và khác biệt so với image-level uncertainty của E1.

E2B là một **DINOv3 adaptation lấy cảm hứng từ IDML**, không phải tái hiện IDML. IDML học uncertainty embedding và Introspective Similarity Metric; E2B dùng binary pair classifier theo đặc tả của repository.

## 2. Protocol và điều kiện tiên quyết

- Dùng CLS embedding 768 chiều từ **outputs/cls/**; không trích xuất lại nếu SHA-256 hợp lệ.
- Dùng E1-A/E1-B trong **outputs/e1_evidential/metrics.json** làm đối chứng và không sửa artifact E1.
- Giữ split CUB class-disjoint: 5.864 ảnh/100 lớp train và 5.924 ảnh/100 lớp test.
- DINOv3 đóng băng; không có gradient đi vào backbone hoặc CLS cache.
- Giữ cosine, self-match exclusion, stable tie, Recall@1/2/4/8, seed 42 và thứ tự query/gallery.
- Không dùng labels 100--199 để tạo pair, mining, chọn epoch, candidate depth hoặc hệ số kết hợp.
- Thiết bị auto ưu tiên CUDA. Cache lưu trên CPU; model, batch, mining và metric nặng chạy trên resolved device.

E1 phải ở trạng thái evaluated trước khi chạy E2B đầy đủ. Hiện chưa có artifact trong **outputs/e1_evidential/**; đây là điều kiện chạy, không cản trở cài đặt.

## 3. Pair Confidence Network

Với hai CLS embedding L2-normalized, feature cặp là:

$$
\phi(a,b)=[z_a;z_b;|z_a-z_b|]\in\mathbb{R}^{2304}.
$$

MLP mặc định: LayerNorm(2304), Linear(2304,512), GELU, Dropout(0,1), Linear(512,128), GELU, Dropout(0,1), Linear(128,1).

Vì phép nối phụ thuộc thứ tự, confidence suy luận được ép đối xứng:

$$
c(a,b)=\frac{\sigma(f_\theta(\phi(a,b)))+
\sigma(f_\theta(\phi(b,a)))}{2}.
$$

Trong train, mỗi cặp xuất hiện ở cả hai hướng. Loss là binary cross entropy với logits:

$$
\mathcal L_{\mathrm{pair}}=
-y\log\sigma(\ell)-(1-y)\log(1-\sigma(\ell)).
$$

Không đưa cosine vào input MLP vì cosine được giữ thành nhánh riêng trong final score, giúp ablation đo contribution của từng thành phần.

## 4. Sinh pair không vật liệu hóa toàn bộ

Không lưu hoặc duyệt mọi cặp $O(N^2)$. Dataset chỉ giữ CLS, labels và index theo lớp; pair index được sinh quyết định bởi seed cộng epoch.

### 4.1. Ranh giới train/validation

Chia 90/10 phân tầng theo từng lớp, cùng hàm và seed với E1. Pair train chỉ chứa ảnh thuộc 90%; pair validation chỉ chứa ảnh thuộc 10%. Không để một ảnh xuất hiện ở cả hai phía.

### 4.2. Positive và negative

Mỗi anchor lấy hai positive khác chính nó:

- một positive ngẫu nhiên cùng lớp;
- một hard positive có cosine thấp trong cùng lớp.

Mỗi anchor lấy hai negative:

- 50% hard negative: ảnh khác lớp gần anchor nhất theo cosine;
- 50% random negative: lấy đều từ lớp khác.

Hard-negative bank được tính theo chunk trên CUDA và chỉ lưu top index trên CPU. Bank được xây một lần cho mỗi training stage vì CLS frozen. Bank train tuyệt đối không chứa validation/test. Pair cân bằng 1:1; cấu hình cho phép thay tỷ lệ hard/random trong ablation.

## 5. Huấn luyện không rò rỉ

| Tham số | Giá trị ban đầu |
|---|---:|
| Epoch tối đa | 30 |
| Anchor mỗi batch | 64 |
| Positive/anchor | 2 |
| Negative/anchor | 2 |
| Optimizer | AdamW |
| Learning rate | $10^{-3}$ |
| Weight decay | $10^{-4}$ |
| Gradient clipping | 10 |
| Validation fraction | 0,1 |

Quy trình:

1. Xây hard-negative bank riêng trên 90% train.
2. Train MLP và đánh giá pair validation cố định mỗi epoch.
3. Chọn epoch theo validation pair AUROC cao nhất; tie-break bằng BCE thấp hơn rồi epoch nhỏ hơn.
4. Khởi tạo lại, train đến selected epoch trên 90% và sinh validation confidence.
5. Chọn candidate depth $M$ và $\lambda$ chỉ bằng retrieval validation.
6. Khởi tạo lại, xây bank trên toàn train và refit đúng selected epoch.
7. Đóng băng MLP và đánh giá test đúng một lần.

Báo cáo validation BCE, AUROC, AUPRC, accuracy tại 0,5 và ECE 15 bins.

## 6. Retrieval và final score

Lấy top-$M$ candidate bằng CLS cosine. Chỉ candidate pool này được chấm Pair Confidence; suffix gallery giữ thứ tự cosine. Cách này tránh chạy MLP trên khoảng 35 triệu cặp test.

$$
\tilde s_{\cos}(q,g)=\frac{s_{\cos}(q,g)+1}{2},
$$

$$
s_{\mathrm{final}}(q,g)=
\lambda\tilde s_{\cos}(q,g)+(1-\lambda)c(q,g).
$$

Khảo sát trên validation:

$$
M\in\{32,64,128\},\qquad
\lambda\in\{0,0.25,0.5,0.75,0.9,1.0\}.
$$

Chọn theo Recall@1, rồi Recall@2, Recall@4, $\lambda$ lớn hơn và $M$ nhỏ hơn. Tie final score giữ thứ tự cosine ban đầu. Pair inference chạy theo chunk trên CUDA, không tạo tensor queries × gallery × 2304.

## 7. So sánh và ablation

| Mã | Phương pháp | Vai trò |
|---|---|---|
| B0 | CLS cosine | E1-A/E2A-M1 |
| B1 | Image uncertainty reranking | E1-B, chỉ đọc kết quả |
| B2 | Pair Confidence, $\lambda=0$ | Khả năng độc lập |
| B3 | Cosine + Pair Confidence | Phương pháp E2B chính |
| B4 | Cosine control, $\lambda=1$ | Phải bằng B0 |

Ablation bắt buộc:

- random negative so với mixed hard/random negative;
- input $[A,B]$, $[|A-B|]$ và $[A,B,|A-B|]$;
- one-way so với symmetric two-way confidence;
- toàn bộ $\lambda$ grid với candidate depth đã chọn.

Chạy ablation sau canonical E2B và lưu output riêng. “Evidential embedding” trong đặc tả ban đầu chưa được E1 tạo ra; ghi **not evaluated** thay vì suy diễn kết quả.

## 8. Thay đổi mã nguồn dự kiến

### Cấu hình

Tạo **configs/cub_e2b.yaml** gồm experiment e2b, đường dẫn CLS/E1, kiến trúc pair network, pair sampling, training, candidate top-$N$ grid, lambda grid và output **outputs/e2b_pair_confidence**. Mở rộng validation trong **src/utils.py**; token giữ nguyên trong config kế thừa và bị loại khỏi metadata public.

### Mô-đun

- **src/pair_confidence.py**: feature builder, MLP, symmetric confidence, BCE và quy tắc selection.
- **src/pair_sampling.py**: index theo lớp, hard-negative bank và deterministic pair sampler.
- Mở rộng **src/retrieval.py** để trả cosine score kèm candidate index và stable rerank bằng external score.
- Tái sử dụng binary AUROC/AUPRC và ECE; chỉ chuyển metric chung khỏi evidential.py nếu cần tránh import sai ngữ nghĩa.

### Entry points

- **scripts/train_pair_confidence.py**: cache validation, split, mining, model selection và refit.
- **scripts/evaluate_e2b.py**: chunked confidence, validation search và test evaluation.
- **scripts/run_e2b.py**: điều phối train/evaluate và hỗ trợ overwrite.

Không sửa hành vi runner E2A/E1.

## 9. Artifact

Lưu tại **outputs/e2b_pair_confidence/**:

- checkpoint.pt;
- training_history.json;
- pair_validation.pt;
- hard_negative_manifest.json;
- hyperparameter_search.json;
- test_rankings.pt;
- failure_cases.json;
- metrics.json;
- manifest.json.

Checkpoint chứa model state, selected epoch, selected $M/\lambda$, kiến trúc và CLS cache hash. Pair validation chỉ lưu index, label, logits/confidence. Test rankings chỉ lưu candidate index, cosine, confidence và final score. Failure cases lưu ca B0 sai nhưng B3 đúng và ngược lại. Manifest ghi paper/version/URL, code provenance/license, công thức, deviations, config public, device, runtime và SHA-256.

Không lưu toàn bộ pair features, mọi pair combination hoặc ma trận gallery score đầy đủ.

## 10. Unit test và smoke test

- Pair feature đúng shape 2304, thứ tự và trị tuyệt đối.
- Đảo pair cho cùng symmetric confidence.
- BCE hữu hạn, backward được và chỉ cập nhật MLP.
- Không self-pair; positive cùng lớp; negative khác lớp.
- Sampler cân bằng, deterministic theo seed và đổi theo epoch.
- Hard-negative bank chỉ chứa đúng split và khác lớp.
- Train/validation index và pair không giao nhau.
- Score thuộc $[0,1]$; $\lambda=0$ là confidence, $\lambda=1$ là normalized cosine.
- Stable tie giữ cosine order; candidate suffix không đổi.
- Chọn epoch, $M/\lambda$ đúng tie-break.
- B0/B4 tái tạo CLS Recall.
- Cache/hash/schema/device lỗi phải fail sớm.

Smoke test dùng tensor nhỏ, chạy 1--2 optimizer step trên resolved device. Không tải DINOv3, không đọc toàn bộ CUB hoặc dùng test labels.

## 11. Trình tự triển khai

1. Khóa config, artifact schema và selection rule.
2. Cài Pair Confidence Network và symmetric inference.
3. Cài deterministic sampler và split-local hard-negative bank.
4. Cài train/validation loop CUDA-first.
5. Mở rộng candidate scoring và stable fusion reranking.
6. Cài validation search cho $M/\lambda$.
7. Cài evaluator, comparison matrix và failure-case export.
8. Thêm unit test và smoke test nhỏ.
9. Sau xác nhận của người dùng, chạy canonical E2B đầy đủ.
10. Khóa canonical artifact trước khi chạy ablation.

## 12. Tiêu chí hoàn thành

- Không có test leakage trong sampling, mining hoặc selection.
- Backbone và CLS cache không đổi; mọi trainable parameter thuộc MLP.
- B0 và B4 khớp CLS baseline trong sai số $10^{-6}$.
- B0, B1, B2 và B3 dùng cùng query/gallery, self-exclusion và Recall implementation.
- Pair inference theo chunk, không tạo ma trận feature đầy đủ.
- Artifact hữu hạn, nạp được trên CPU, có checksum và provenance.
- Chỉ kết luận E2B cải thiện nếu B3 vượt B0 trên test với $M/\lambda$ đã khóa từ validation.
