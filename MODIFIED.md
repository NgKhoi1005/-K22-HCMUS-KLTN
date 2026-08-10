# Phân tích `T826_KL_KHMT24_SourceCode.ipynb`: đối chiếu với `ORGINAL_ColabNAS.ipynb` và bài báo gốc ColabNAS

## 0. Bối cảnh

- `ORGINAL_ColabNAS.ipynb` (notebook đi kèm bài báo, public trên GitHub) chỉ là **companion code tối giản**: class `ColabNAS` (đúng Section 3 của bài báo), hàm `test_tflite_model`, và **một** ví dụ sử dụng (1 dataset, 1 cấu hình phần cứng duy nhất — STM32L412KBU3).
- Bài báo (*"ColabNAS: Obtaining lightweight task-specific convolutional neural networks following Occam's razor"*, Garavagno et al., 2024) ở các **Section 4–7** mô tả bằng lời + bảng số liệu **4 thí nghiệm đầy đủ**: (4) 5 bộ dữ liệu, (5) so sánh với Transfer Learning, (6) đánh giá trên 3 phần cứng hạn chế, (7) so sánh SOTA (MCUNet, Micronets) trên VWW — nhưng notebook gốc **không hề chứa code** để chạy các thí nghiệm này.
- `T826_KL_KHMT24_SourceCode.ipynb` tự viết lại toàn bộ hạ tầng thí nghiệm còn thiếu đó, đồng thời bổ sung một số ý tưởng của riêng sinh viên.

Dưới đây tách rõ hai loại khác biệt mà `T826` mang lại: **(A)** code mới so với notebook gốc nhưng bản chất là tái hiện lại đúng phương pháp bài báo đã mô tả, và **(B)** phần thực sự vượt ra ngoài những gì bài báo trình bày.

---

## Phần A — Code mới so với notebook gốc, nhưng ý tưởng đã có sẵn trong bài báo

Đây là phần chiếm khối lượng lớn nhất của `T826`. Bài báo có mô tả bằng lời/bằng bảng (Table 1–10) nhưng notebook gốc không có code tương ứng, nên các con số cấu hình dưới đây được đối chiếu và khớp **chính xác** với bài báo:

| Nội dung trong T826 | Vị trí trong bài báo | Đối chiếu số liệu |
|---|---|---|
| Mô tả 5 dataset: Melanoma, Flowers-4, Animals-3, MNIST, VWW | Section 4.1–4.5 | Số lượng ảnh T826 ghi khớp với bài báo: Melanoma 9 605 train / 1 000 test; Flowers-4 1 052 dandelion + 1 054 iris + 1 048 magnolia + 1 048 tulip (3 360 train / 842 test); Animals-3 ghi theo số ảnh của tập train (1 689 bướm / 2 478 gà / 2 098 ngựa), tổng 6 265 train + 1 568 test = 7 833 ảnh — khớp 2 112 / 3 098 / 2 623 ảnh mỗi lớp mà bài báo báo cáo; MNIST 60 000 / 10 000; VWW 115 228 / 8 059 |
| `hardware_configs` (L0, L1, L4 = STM32L010RBT6 / L151UCY6DTR / L412KBU3) | Table 4 (Section 6) | RAM/Flash tính bằng KiB×1024, MACC = CoreMark×10⁴: L0 = 20480/131072/750000, L1 = 32768/262144/930000, L4 = 40960/131072/2730000 — khớp tuyệt đối. Notebook gốc chỉ hard-code sẵn 1 target (L4) trong ví dụ dùng. |
| `searchOnHardware`, `testModel`, vòng lặp `for target in hardware_configs` | Section 6 (đánh giá hardware-aware) | Tái hiện đúng quy trình "chạy ColabNAS trên từng target, so sánh model kết quả" mà bài báo mô tả cho cả 5 dataset. |
| Class `TLModel` (MobileNetV2 backbone đóng băng, GAP + 1 Dense, train 20 epoch @ lr 1e-3, fine-tune 10 epoch @ lr 1e-5, input 224×224×3) | Section 5 | Khớp chính xác cấu hình epoch/lr/batch mà bài báo mô tả cho quy trình TL. |
| Ràng buộc RAM/Flash/MACC dùng cho ColabNAS khi so sánh với TL (2 583 552 B / 2 713 592 B / 300 000 000) | Table 1 (kết quả TL trên Melanoma) | Khớp với 2523 KiB / 2650 KiB / 300 MM mà bài báo báo cáo. |
| Chỉ chọn Melanoma/Flowers-4/Animals-3 cho phần TL (không có MNIST, VWW) | Section 5, đoạn giải thích loại VWW & MNIST | Bài báo giải thích rõ 2 dataset này bị loại "vì số lượng ảnh lớn khiến ColabNAS vượt quá thời gian miễn phí của Colab" — T826 tuân theo đúng giới hạn này. |
| Cấu hình so sánh SOTA trên VWW: `sota_max_RAM=131072`, `sota_max_Flash=524288`, `sota_max_MACC=6080000`, hardware STM32F446RE | Table 10 / Section 7 | Khớp đúng "128 kiB RAM, 512 kiB Flash, CoreMark 608" mà bài báo nêu cho MCU dùng để so với MCUNet/Micronets. |
| Đo latency bằng `%timeit interpreter.invoke()` sau khi `stm32tflm` chạy trên model kết quả | Section 7 | Bài báo mô tả nguyên văn: *"a latency value is measured using the IPython magic command '%timeit' alongside the TFLite interpreter invocation"* — T826 hiện thực đúng câu lệnh này. Notebook gốc không đo latency ở bất kỳ đâu. |

Ngoài ra là các refactor thuần kỹ thuật không liên quan ý tưởng bài báo: dùng `pathlib.Path` thay chuỗi nối, `shutil.rmtree` thay `os.system("rm -rf")`, thêm `verbose=0` và tắt `model.summary()` cho gọn log, đưa `epochs` thành tham số khởi tạo của class (bản gốc hard-code 100), cho `quantize_model_uint8` / `evaluate_flash_and_peak_RAM_occupancy` nhận đường dẫn nguồn–đích làm tham số thay vì đọc `self.model_name`, lưu song song cả base model (`base_architecture_*.tflite`) cạnh model kết quả để có thể đánh giá cả hai, và setup đường dẫn cho môi trường Kaggle thay vì Colab.

**Kết luận Phần A:** đây không phải là ý tưởng "vượt ra ngoài bài báo" — là phần **mã hoá lại trung thực** các thí nghiệm mà bài báo đã mô tả bằng lời/bảng số liệu nhưng notebook gốc (chỉ chứa thuật toán lõi) chưa từng hiện thực. Nói ngắn gọn: *code mới, ý tưởng cũ (lấy từ paper)*.

---

## Phần B — Điểm thực sự vượt ra ngoài bài báo gốc (ý tưởng không có trong paper)

### 1. Đưa augmentation ra khỏi vòng lặp tìm kiếm, chỉ áp dụng cho kiến trúc thắng cuộc (two-stage)
Trong bản gốc, hai lớp `RandomFlip('horizontal')` và `RandomRotation(0.2, ...)` nằm ngay trong preprocessing pipeline của `Model()` (`ORGINAL_ColabNAS.ipynb`), nên **mọi** kiến trúc thử trong quá trình search đều được huấn luyện kèm augmentation, và augmentation trở thành một phần cố định của model kết quả.

T826 tách augmentation khỏi graph của model, chuyển thành một `tf.keras.Sequential` áp lên tập dữ liệu riêng `train_ds_aug`, và chỉ dùng nó cho **kiến trúc tốt nhất tìm được** sau khi `search` kết thúc (`two_stage=True`): nạp lại trọng số của candidate, huấn luyện thêm `self.epochs` epoch trên `train_ds_aug`, rồi **chỉ giữ model augmentation nếu val_accuracy của nó cao hơn base model** (`model_type` trả về `"augmentation"` hoặc `"base model"`). Cả hai model đều được lưu (`base_architecture_*.tflite` và `resulting_architecture_*.tflite`) và đều được đánh giá trên tập test.

Nhờ đó quá trình search không còn phải chịu thêm nhiễu và chi phí của augmentation ở từng bước, còn augmentation vẫn được kiểm chứng có thực sự giúp ích hay không trên đúng kiến trúc cuối cùng. Cấu hình hiện tại: bật ở Thí nghiệm 1 (`hw_two_stage = True`), tắt ở Thí nghiệm 3 (`sota_two_stage = False`). Một biến thể two-stage khác — fine-tune từng model với augmentation ở learning rate giảm 100 lần, trong `train(train_mode="two-stage")` — hiện để dạng comment.

### 2. Cơ chế huấn luyện lại (`max_atempt`) trong `explore_num_cells`
Bài báo (Algorithm 2, EXPLORE_NUM_CELLS) chỉ dùng **đúng một** chiến lược: hill-climbing thuần túy, dừng ngay lập tức khi accuracy không tăng so với bước trước — nghĩa là một lần huấn luyện "kém may mắn" là đủ để chấm dứt việc thêm cell.

T826 giữ nguyên vòng hill-climbing đó nhưng cho phép huấn luyện lại **đúng cấu hình `(k, c)` vừa bị giảm accuracy** tối đa `max_atempt` lần cho mỗi giá trị `k`, với điều kiện mức sụt giảm so với bước trước đạt ngưỡng `1e-3` và kiến trúc đó vẫn khả thi về phần cứng. Nếu lần huấn luyện lại cho val_accuracy vượt bước trước, vòng lặp tiếp tục tăng `c` như bình thường. Mục đích là tránh dừng sớm vì nhiễu ngẫu nhiên của một lần huấn luyện.

Mặc định `search(max_atempt=1)` — Thí nghiệm 1 dùng mặc định này; Thí nghiệm 2 và Thí nghiệm 3 gọi với `max_atempt=0`, tức chạy đúng như bản gốc.

### 3. Cơ chế fixed-seed / determinism để tăng khả năng tái lập kết quả
Hằng số `FIXED_SEED = 11` cùng loạt dòng đánh dấu `# [Fix Randomness]` (hiện để dạng comment, chưa bật mặc định):
- `tf.keras.utils.set_random_seed(FIXED_SEED)`
- `tf.config.experimental.enable_op_determinism()`
- `tf.keras.initializers.GlorotUniform(seed=FIXED_SEED)` cho khởi tạo trọng số
- `tf.keras.backend.clear_session()` + reset seed ở đầu mỗi vòng lặp NAS (`evaluate_model_process`) và ở cell chạy Transfer Learning

Bài báo **không hề đề cập** đến việc kiểm soát seed khởi tạo trọng số mạng hay bật deterministic ops trên GPU; bài báo (và notebook gốc) chỉ dùng `seed=11` để chia tập train/validation — đúng vai trò mà `FIXED_SEED` đang thực sự đảm nhiệm trong `image_dataset_from_directory` của T826. Đây là mối quan tâm về **tính lặp lại thực nghiệm (reproducibility)** mà sinh viên tự đưa thêm vào, không xuất phát từ paper.

### 4. Gộp chạy đồng thời cả 3 mục tiêu phần cứng cho một dataset đã chọn
Bài báo trình bày kết quả Section 6 dưới dạng bảng cho từng target riêng lẻ, không đưa ra đoạn mã tổ chức thực nghiệm. Vòng lặp `for target in hardware_configs: ...` trong T826 là cách tổ chức lại (không phải ý tưởng thuật toán mới) để tái hiện nhanh toàn bộ Section 6 chỉ trong một lần chạy.

---

## Bảng tóm tắt

| Nhóm nội dung | Có trong bài báo? | Có trong notebook gốc? | Trong T826 |
|---|---|---|---|
| Thuật toán lõi ColabNAS (`Model`, `search`, hill-climbing trong `explore_num_cells`) | Có (Section 3) | Có | Giữ nguyên logic tìm kiếm, refactor path/log; các thay đổi khác nêu ở Phần B |
| Mô tả 5 dataset | Có (Section 4) | Không | Thêm mới — mã hoá lại mô tả của bài báo |
| Thí nghiệm hardware-aware trên 3 target (L0/L1/L4) | Có (Section 6) | Không (chỉ 1 target) | Thêm mới, khớp đúng số liệu Table 4 |
| So sánh Transfer Learning (`TLModel`) | Có (Section 5) | Không | Thêm mới, khớp đúng cấu hình Section 5 |
| So sánh SOTA (MCUNet/Micronets) + đo latency | Có (Section 7) | Không | Thêm mới, khớp đúng Table 10 |
| Augmentation chỉ áp dụng cho kiến trúc thắng cuộc (two-stage) | **Không** (bài báo augmentation mọi model trong lúc search) | Không (augmentation nằm trong `Model()`) | **Ý tưởng mới của sinh viên** (đang bật ở Thí nghiệm 1) |
| Huấn luyện lại `max_atempt` lần trong `explore_num_cells` | **Không** | Không | **Ý tưởng mới của sinh viên** (đang bật, mặc định 1 lần) |
| Fixed-seed / determinism cho weight init | **Không** | Không | **Ý tưởng mới của sinh viên** (đang để tắt) |

**Tóm lại:** phần lớn khối lượng công việc "thêm mới" trong T826 (thí nghiệm 1, 2, 3 — hardware-aware, transfer learning, so sánh SOTA) thực chất là việc **tái hiện trung thực** các thí nghiệm mà bài báo đã mô tả nhưng notebook gốc chưa hiện thực. Ba điểm **thực sự là ý tưởng mới, không có trong bài báo gốc** là: (1) đưa augmentation ra khỏi vòng lặp tìm kiếm và chỉ huấn luyện lại kiến trúc thắng cuộc với dữ liệu augmentation, chỉ giữ lại nếu tốt hơn base model; (2) cho huấn luyện lại tối đa `max_atempt` lần khi accuracy sụt giảm, để không dừng sớm vì nhiễu của một lần huấn luyện; và (3) cơ chế cố định seed/determinism nhằm tăng khả năng tái lập kết quả. Hai ý tưởng đầu đang được bật ở Thí nghiệm 1, ý tưởng thứ ba hiện vẫn để ở dạng code comment.
