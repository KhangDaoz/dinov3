II. THỰC NGHIỆM E2B: Pair-wise Confidence Learning for Reliable Retrieval
1. Mục tiêu
Khắc phục hạn chế phát hiện trong Experiment 1: image-level uncertainty không phản ánh chính xác retrieval correctness. E2B học độ tin cậy của quan hệ giữa hai ảnh (pair-wise confidence).
2. Ý tưởng phương pháp
Thay vì dự đoán uncertainty cho từng ảnh riêng lẻ, mô hình học confidence score cho một cặp ảnh (query, candidate). Confidence được kết hợp với cosine similarity để tạo ranking score cuối cùng.
3. Feature extraction
Sử dụng embedding tốt nhất từ E2A. Backbone DINOv3 được giữ frozen.
4. Xây dựng pair training data
Positive pair: hai ảnh cùng class, label = 1.
Negative pair: hai ảnh khác class, label = 0.
5. Pair Confidence Network
Input: [feature A, feature B, |feature A - feature B|].
Mạng MLP dự đoán confidence score trong khoảng [0,1].
Loss: Binary Cross Entropy.
6. Retrieval inference
Final score = lambda * Cosine Similarity + (1-lambda) * Pair Confidence. Khảo sát các giá trị lambda khác nhau.
7. So sánh thực nghiệm
So sánh: DINOv3 baseline, Evidential embedding, Image uncertainty reranking, Pair-wise Confidence, Fusion Cosine + Confidence.
8. Ablation study
Đánh giá ảnh hưởng của cosine similarity, pair confidence và hệ số lambda.
9. Kết quả cần lưu
Lưu checkpoint mô hình, confidence score, embedding, bảng Recall@K và các failure cases để phân tích.