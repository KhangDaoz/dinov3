# Ghi chú ký hiệu cho báo cáo E1–E2A–E2B

Tài liệu này giải thích hệ ký hiệu trong `reports/e1_e2b.tex`. Mục tiêu là tránh dùng một chữ cho nhiều ý nghĩa và giúp đối chiếu công thức với code.

## Quy ước dữ liệu và chỉ số

| Ký hiệu | Ý nghĩa |
|---|---|
| \(\mathcal D=\{(x_i,y_i)\}_{i=1}^{M}\) | Tập dữ liệu gồm \(M\) ảnh và nhãn |
| \(x_i, y_i\) | Ảnh thứ \(i\) và nhãn lớp dạng số nguyên |
| \(i,j\) | Chỉ số query và candidate |
| \(c\in\{1,\ldots,C\}\) | Chỉ số lớp; \(C=100\) trong mỗi nửa CUB |
| \(t\in\{1,\ldots,T\}\) | Chỉ số Patch Token; \(T=196\) |
| \(K\) | Số kết quả đầu dùng trong Recall@\(K\) |
| \(N\) | Số candidate đầu được rerank: 10, 20, 50 hoặc 100 |

Không dùng \(K\) cho số lớp vì \(K\) đã có nghĩa quen thuộc trong Recall@\(K\). Vì vậy công thức uncertainty dùng \(C/S_i\), không dùng \(K/S_i\).

## Embedding và cosine

- \(\mathbf z_i\): embedding thô của ảnh \(x_i\).
- \(\widehat{\mathbf z}_i=\mathbf z_i/\|\mathbf z_i\|_2\): embedding đã chuẩn hóa L2.
- \(s^{\mathrm{cos}}_{ij}=\widehat{\mathbf z}_i^\top\widehat{\mathbf z}_j\): cosine similarity giữa query \(i\) và candidate \(j\).
- \(\mathbf h_i^{\mathrm{cls}}\): final-layer CLS token.
- \(\mathbf h_{it}^{\mathrm{patch}}\): Patch Token thứ \(t\) của ảnh \(i\).
- \(\mathbf z_i^{(m)}\): embedding do phương pháp E2A thứ \(m\) tạo ra.

Chữ đậm luôn chỉ vector hoặc ma trận. Dấu mũ “hat” luôn chỉ vector đã chuẩn hóa, không phải dự đoán lớp.

## Evidential Deep Learning

Với lớp \(c\) của ảnh \(i\):

\[
e_{ic}=\operatorname{softplus}([f_\theta(x_i)]_c),\qquad
\alpha_{ic}=e_{ic}+1.
\]

- \(e_{ic}\): evidence không âm.
- \(\alpha_{ic}\): tham số Dirichlet.
- \(S_i=\sum_{c=1}^{C}\alpha_{ic}\): Dirichlet strength, tức tổng các tham số Dirichlet.
- \(u_i=C/S_i\): image uncertainty; giá trị lớn hơn nghĩa là mô hình ít chắc chắn hơn.
- \(\boldsymbol\alpha_i\): toàn bộ vector \(C\) tham số Dirichlet của ảnh \(i\).
- \(d^\alpha_{ij}=\|\boldsymbol\alpha_i-\boldsymbol\alpha_j\|_2\): khoảng cách dùng cho Evidential Embedding; nhỏ hơn nghĩa là gần hơn.

## Pairwise Confidence và Fusion

\[
\boldsymbol\phi_{ij}=[\widehat{\mathbf z}_i;\widehat{\mathbf z}_j;
|\widehat{\mathbf z}_i-\widehat{\mathbf z}_j|],\quad
\ell_{ij}=g_\omega(\boldsymbol\phi_{ij}),\quad
c_{ij}=\sigma(\ell_{ij}).
\]

\(\ell_{ij}\) là logit của MLP; \(c_{ij}\in(0,1)\) là pairwise confidence score. Báo cáo gọi đây là “score”, không khẳng định là xác suất đã được hiệu chỉnh.

Điểm kết hợp là

\[
s^{\mathrm{final}}_{ij}(\lambda)
=\lambda s^{\mathrm{cos}}_{ij}+(1-\lambda)c_{ij},\qquad \lambda\in[0,1].
\]

Ký hiệu \(s^{\mathrm{final}}\) được dùng thay cho \(S_\lambda\) để không nhầm với Dirichlet strength \(S_i\). \(\lambda\) là hệ số kết hợp, không phải xác suất.

## Recall và Hits

Đặt \(h_i^{(K)}=1\) nếu Top-\(K\) của query \(i\) có ít nhất một ảnh cùng lớp, ngược lại bằng 0. Khi đó:

\[
\operatorname{Recall@}K=\frac{1}{|\mathcal Q|}\sum_{x_i\in\mathcal Q}h_i^{(K)},
\qquad H@K=\sum_{x_i\in\mathcal Q}h_i^{(K)}.
\]

Recall@\(K\) là tỷ lệ; Hits@\(K\) là số query thành công tương ứng. Hits chỉ được dùng để so sánh chính xác khi các giá trị Recall làm tròn trông giống nhau.

## Cách đọc nhanh các chữ cái

- \(C\): classes.
- \(T\): tokens.
- \(K\): retrieval cutoff.
- \(N\): reranking depth.
- \(S_i\): Dirichlet strength.
- \(u_i\): uncertainty của một ảnh.
- \(c_{ij}\): confidence của một cặp ảnh.
- \(s^{\mathrm{cos}}_{ij}\), \(s^{\mathrm{final}}_{ij}\): điểm cosine và điểm cuối.

Hệ ký hiệu này phù hợp với cách trình bày phổ biến trong EDL, deep metric learning và image retrieval, đồng thời giữ riêng các đại lượng có vai trò khác nhau.

## Tài liệu đối chiếu

- Sensoy, Kaplan và Kandemir, *Evidential Deep Learning to Quantify Classification Uncertainty*: định nghĩa evidence, Dirichlet parameter, strength và \(u=C/S\). <https://arxiv.org/abs/1806.01768v3>
- Kim và cộng sự, *Proxy Anchor Loss for Deep Metric Learning*: cách ký hiệu embedding, proxy, cosine similarity và Recall@\(K\). <https://arxiv.org/abs/2003.13911v1>
- Đorđević và Kumar, *Evidential Transformers for Improved Image Retrieval*: dùng vector \(\boldsymbol\alpha\) làm image embedding và khoảng cách L2. <https://arxiv.org/abs/2409.01082v2>
