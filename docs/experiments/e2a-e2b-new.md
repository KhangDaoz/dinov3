Mục tiêu: chạy lại E2A–E2B theo protocol mới để tránh selection bias và đánh giá ảnh hưởng của representation trong Pair-wise Confidence Learning.
1. Mục tiêu
Nguyên tắc chính:
• Classes 0–99: Development Set.
• Classes 100–199: Final Unseen Test Set.
• Không dùng classes 100–199 để chọn representation, checkpoint hoặc hyperparameter.
• Không chọn M1 trước. E2A được thực hiện dưới dạng ablation study, đưa cả M1–M4 vào E2B.
2. Dataset split
Classes 0–99 – Development: với mỗi class, chia 80% ảnh → Train và 20% ảnh → Validation. Train và Validation có cùng class nhưng không trùng ảnh.

Classes 100–199 – Final Test: giữ nguyên toàn bộ và chỉ sử dụng ở bước final evaluation.
3. E2A – Representation Ablation
Giữ 4 representation:
• M1 – CLS: DINOv3 → CLS Token.
• M2 – Mean Patch: DINOv3 → Patch Tokens → Mean Pooling.
• M3 – CLS + Mean Patch: CLS + Mean Patch → Projection.
• M4 – Attention Pooling: Patch Tokens → Attention Pooling.

Không dùng Recall trên classes 100–199 để chọn representation.
4. Chọn checkpoint E2A
M1, M2: không có module trainable chính, sử dụng trực tiếp.
M3, M4: train trên Train set classes 0–99 và chọn checkpoint bằng Validation set classes 0–99.
Sau khi chọn checkpoint, khóa M1–M4.
5. E2B – Pair-wise Confidence
Không chọn một representation duy nhất. Chạy 4 nhánh độc lập:
M1 → Pairwise Confidence Network
M2 → Pairwise Confidence Network
M3 → Pairwise Confidence Network
M4 → Pairwise Confidence Network

Kiến trúc Pairwise Confidence Network phải giống nhau ở cả 4 nhánh; mục đích là chỉ so sánh ảnh hưởng của representation.
6. Pair construction
Trên Train set của classes 0–99:
• Positive pair: Image A + Image B, same class → label 1.
• Negative pair: Image A + Image B, different class → label 0.

Giữ sampling protocol giống E2B hiện tại để đảm bảo so sánh công bằng.
7. Training E2B
Với từng M1–M4:
• Train: classes 0–99 / Train.
• Validation: classes 0–99 / Validation.
• Chọn checkpoint trên Validation.
• Chọn các hyperparameter cần thiết trên Validation.

Không được sử dụng classes 100–199 trong quá trình này.
8. Final Evaluation
Sau khi toàn bộ 4 model đã được khóa, mới chạy Final Test trên classes 100–199.
A. Cosine Retrieval: Scos
B. Pairwise Confidence: Cpair
C. Fusion: Sfinal = λ * Scos + (1 - λ) * Cpair
Giữ cách tính và các giá trị λ giống E2B hiện tại.
9. Bảng kết quả 
A. Cosine Retrieval
Representation	R@1	R@2	R@4	R@8
M1 – CLS				
M2 – Mean Patch				
M3 – CLS + Mean Patch				
M4 – Attention Pooling				
B. Pairwise Confidence
Representation	R@1	R@2	R@4	R@8
M1				
M2				
M3				
M4				
C. Fusion
Representation	R@1	R@2	R@4	R@8
M1				
M2				
M3				
M4				
10. Các file cần gửi lại
1. cosine_results.csv
2. pairwise_results.csv
3. fusion_results.csv
4. Checkpoint M3
5. Checkpoint M4
6. Checkpoint Pairwise Network của M1–M4
7. Training log
8. Seed và configuration
11. Lưu ý quan trọng
Không được dùng Final Test (classes 100–199) để chọn M1/M2/M3/M4, chọn checkpoint, chọn λ, chỉnh threshold hoặc thay đổi architecture. Final Test chỉ dùng để đánh giá và báo cáo kết quả cuối cùng.
12. Mục tiêu của lần chạy này
Không phải chứng minh M1 luôn tốt nhất. Mục tiêu là trả lời: “Representation nào phù hợp nhất cho Retrieval và Pair-wise Confidence Learning?”

Có thể M1, M2, M3 hoặc M4 tốt nhất. Không chọn trước và không điều chỉnh theo kết quả Final Test.

Sau khi hoàn thành, gửi toàn bộ bảng kết quả + checkpoint + configuration để kiểm tra trước khi chốt thực nghiệm.