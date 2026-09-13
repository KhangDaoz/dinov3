*Lưu ý chung
Giữ nguyên dataset split và protocol đánh giá so với baseline. Mục tiêu của E2A là tìm representation tốt nhất; mục tiêu của E2B là xây dựng retrieval confidence phù hợp với bài toán.

I. THỰC NGHIỆM E2A: DINOv3 Feature Representation Analysis
1. Mục tiêu
Đánh giá ảnh hưởng của các phương pháp trích xuất đặc trưng từ backbone DINOv3 ViT-B/16 đối với bài toán Image Retrieval trên CUB-200-2011. Mục tiêu là lựa chọn representation tốt nhất trước khi xây dựng Pair-wise Confidence Learning.
2. Thiết lập chung
Dataset: CUB-200-2011. Giữ nguyên protocol đánh giá baseline.
Backbone: DINOv3 ViT-B/16 pretrained. Frozen toàn bộ backbone, không fine-tuning.
Similarity: Cosine Similarity.
3. Các phương pháp cần thử nghiệm
Method 1 - CLS Token (Baseline): sử dụng CLS token lớp cuối cùng làm embedding 768 chiều.

Method 2 - Mean Patch Pooling: lấy trung bình toàn bộ patch tokens của DINOv3 để tạo embedding.

Method 3 - CLS + Mean Patch Fusion: ghép CLS token và Mean Patch, sau đó projection về không gian embedding.

Method 4 - Attention Pooling học trọng số các patch tokens để tạo representation.
4. Đánh giá
Sử dụng Recall@1, Recall@2, Recall@4, Recall@8. Không sử dụng uncertainty hoặc reranking trong thực nghiệm này.
5. Kết quả cần lưu
Lưu bảng so sánh Recall@K giữa các representation. Lưu embedding của từng phương pháp để sử dụng cho các thực nghiệm tiếp theo.