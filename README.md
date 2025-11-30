# HR Analytics: Job Change of Data Scientists

## Mục lục

1. [Giới thiệu](#giới-thiệu)
2. [Dataset](#dataset)
3. [Phương pháp (Method)](#phương-pháp-method)
4. [Installation & Setup](#installation--setup)
5. [Usage](#usage)
6. [Results](#results)
7. [Project Structure](#project-structure)
8. [Challenges & Solutions](#challenges--solutions)
9. [Future Improvements](#future-improvements)
10. [Contributors](#contributors)
11. [License](#license)

---

## Giới thiệu

### Mô tả bài toán

Một công ty chuyên về Big Data và Data Science muốn dự đoán **khả năng ứng viên sẽ đổi việc** sau khi hoàn thành các khoá học/đào tạo của công ty.  
Nhiệm vụ là:

- Phân tích các yếu tố liên quan đến quyết định **có/không muốn đổi việc**.
- Xây dựng mô hình dự đoán nhị phân:  
  - `target = 0`: Không tìm việc mới  
  - `target = 1`: Đang hoặc sẽ tìm việc mới

### Động lực và ứng dụng thực tế

- Hỗ trợ phòng nhân sự (HR) nhận diện các ứng viên **dễ rời bỏ** để:
  - Điều chỉnh chính sách lương thưởng, lộ trình phát triển.
  - Tối ưu chi phí tuyển dụng, đào tạo.
- Giúp ứng viên hiểu rõ **profile của những người có xu hướng đổi việc**.

### Mục tiêu cụ thể

- Làm sạch và tiền xử lý dữ liệu **chỉ với NumPy**.
- Khai thác dữ liệu bằng trực quan hoá (Matplotlib, Seaborn).
- Xây dựng Logistic Regression implement **từ đầu bằng NumPy**.
- Đánh giá mô hình bằng các độ đo: Accuracy, Precision, Recall, F1-score.

---

## Dataset

### Nguồn dữ liệu

Dataset: **HR Analytics: Job Change of Data Scientists** (Kaggle)

### Mô tả các features chính

- `enrollee_id`: ID ứng viên (không dùng cho mô hình).
- `city`: Mã thành phố làm việc.
- `city_development_index`: Chỉ số phát triển của thành phố (đã scale).
- `gender`: Giới tính.
- `relevent_experience`: Có/không có kinh nghiệm liên quan.
- `enrolled_university`: Tình trạng đang học đại học.
- `education_level`: Trình độ học vấn.
- `major_discipline`: Chuyên ngành.
- `experience`: Số năm kinh nghiệm
- `company_size`: Quy mô công ty hiện tại.
- `company_type`: Loại công ty.
- `last_new_job`: Thời gian từ lần đổi công việc trước.
- `training_hours`: Số giờ training hoàn thành.
- `target`: Nhãn mục tiêu (0/1).

### Kích thước và đặc điểm

- Số dòng train: ~19k
- Số cột: 14
- Dữ liệu gồm cả:
  - Numeric: `city_development_index`, `training_hours`
  - Ordinal: `experience`, `company_size`, `last_new_job`
  - Categorical: `city`, `gender`, `major_discipline`, ...

---

## Phương pháp (Method)

### Quy trình xử lý dữ liệu

1. **Đọc dữ liệu**  
   - Sử dụng `np.genfromtxt` với `dtype=str`
2. **Chuẩn hoá missing values**:
   - Coi các giá trị `"", Unknown, Other, ...` như **missing**.
3. **Xử lý kiểu dữ liệu**:
   - Chuyển các cột numeric về `float`.
   - Mã hoá ordinal cho:
     - `experience` → số năm gần đúng.
     - `company_size` → số lượng nhân viên gần đúng.
     - `last_new_job` → số năm kể từ lần đổi việc.
4. **Mã hoá categorical**:
   - Dùng `np.unique(..., return_inverse=True)` để map category → code.
5. **Điền missing values**:
   - Numeric: dùng **mean** hoặc **median**.
6. **Chuẩn hoá (Normalization / Standardization)**:
   - `training_hours` → log transform `log1p` → z-score.
   - Các numeric khác → z-score
7. **Lưu dữ liệu đã xử lý**:
   - `data/processed/processed_train.csv` — gồm cột `target` và các features numeric.
   - `data/processed/processed_test.csv` — gồm `enrollee_id` và các features numeric.

### Thuật toán sử dụng

#### Logistic Regression (NumPy)

- Mô hình:

$$
\hat{y} = \sigma(W^T x + b), \quad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

- Hàm mất mát (Binary Cross Entropy):

$$
L = -\frac{1}{m}\sum_{i=1}^m \big[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\big]
$$

- Gradient:

$$
\frac{\partial L}{\partial W} = \frac{1}{m} X^T(\hat{y}-y), \quad
\frac{\partial L}{\partial b} = \frac{1}{m}\sum_{i=1}^m (\hat{y}_i-y_i)
$$

- Cập nhật tham số (Gradient Descent):

$$
W := W - \alpha \frac{\partial L}{\partial W},\quad
b := b - \alpha \frac{\partial L}{\partial b}
$$


### Độ đo đánh giá

- Accuracy  
- Precision  
- Recall  
- F1-score  

Được implement lại bằng NumPy trong `src/models.py`.

### Cross-validation

- Cài đặt **K-Fold** thủ công bằng NumPy.
- Đánh giá trung bình các độ đo cho 5 folds.

---

## Installation & Setup

```bash
git clone https://github.com/CrisNam77/jobchange-datascientists-analysis.git
cd jobchange-datascientists-analysis

python -m venv .venv
MacOS: source .venv/bin/activate 
Window: .venv\Scripts\activate

pip install -r requirements.txt
```
## Usage

Chạy các notebook theo thứ tự để tái hiện quá trình phân tích:

1.  **Bước 1: Khám phá dữ liệu (EDA)**
    - Mở `notebooks/01_data_exploration.ipynb`.
    - Chạy toàn bộ để xem phân tích thống kê, phân bố target và các biểu đồ trực quan hóa.

2.  **Bước 2: Tiền xử lý**
    - Mở `notebooks/02_preprocessing.ipynb`.
    - Notebook này gọi các hàm từ `src/data_processing.py` để làm sạch và lưu file đã xử lý vào `data/processed/`.

3.  **Bước 3: Huấn luyện mô hình**
    - Mở `notebooks/03_modeling.ipynb`.
    - Thực hiện Feature Engineering, huấn luyện `LogisticRegressionNumpy`, và đánh giá kết quả trên tập Validation.


## 6. Results

Kết quả đánh giá trên tập Validation (20% split) sử dụng mô hình tự code:

- **Accuracy:** ~74.5%
- **Precision:** ~46.3%
- **Recall:** ~16.7%
- **F1-Score:** ~0.24

**Phân tích:**
- Mô hình đạt độ chính xác khá (74.5%) nhưng Recall thấp đối với nhãn 1 (Người muốn đổi việc).
- Nguyên nhân chính là do dữ liệu mất cân bằng (Imbalanced Data - Nhãn 0 chiếm đa số), khiến mô hình thiên về dự đoán nhãn 0.
- Confusion Matrix và ROC Curve (xem trong notebook 03) cho thấy mô hình cần cải thiện thêm khả năng phát hiện nhóm thiểu số.


## 7. Project Structure

```bash
project-name/
├── README.md               # Tài liệu dự án
├── requirements.txt        # Các thư viện cần thiết
├── data/
│   ├── raw/                # Dữ liệu gốc (aug_train.csv, aug_test.csv)
│   └── processed/          # Dữ liệu sau khi xử lý (csv)
├── notebooks/
│   ├── 01_data_exploration.ipynb  # EDA: Khám phá dữ liệu
│   ├── 02_preprocessing.ipynb     # Pipeline tiền xử lý
│   └── 03_modeling.ipynb          # Feature Eng, Model Training & Eval
└── src/
    ├── __init__.py
    ├── data_processing.py  # Module xử lý dữ liệu bằng NumPy
    ├── visualization.py    # Module vẽ biểu đồ (Matplotlib/Seaborn)
    └── models.py           # Module chứa LogisticRegressionNumpy
```

## Challenges & Solutions
1. Thao tác với dữ liệu hỗn hợp (Mixed Data Types) mà không có Pandas:
- Thách thức: `np.genfromtxt` khi đọc file CSV chứa cả chuỗi và số thường trả về mảng có `dtype=object` hoặc `string`, khiến việc tính toán toán học ngay lập tức là không thể. Ngoài ra, việc truy cập cột bằng tên (như `df['column']`) không được hỗ trợ sẵn.
- Giải pháp:
   - Viết hàm tiện ích `load_csv_numpy` để tách riêng header và data.
   - Viết hàm col_idx để ánh xạ tên cột sang chỉ số index (int).
   - Sử dụng phương pháp ép kiểu dữ liệu (astype) linh hoạt cho từng cột cụ thể khi cần tính toán.
2. Xử lý giá trị thiếu (Missing Values) thủ công:
- Thách thức: Không có các hàm tiện lợi như `fillna()` hay `dropna()`.
- Giải pháp:
   - Chuẩn hóa tất cả các giá trị lạ về np.nan.
   - Sử dụng kỹ thuật Boolean Masking của NumPy (ví dụ: arr[np.isnan(arr)] = value) để điền giá trị trung bình (Mean) hoặc trung vị (Median) được tính toán từ các phần tử không bị thiếu (np.nanmean, np.nanmedian).

## Future Improvements
Để cải thiện hiệu suất mô hình và mở rộng dự án, các hướng phát triển tiếp theo bao gồm:
- Xử lý mất cân bằng dữ liệu (Imbalanced Data): Tự cài đặt thuật toán SMOTE (Synthetic Minority Over-sampling Technique) hoặc Random Undersampling bằng NumPy để cân bằng tỷ lệ giữa nhãn 0 và 1.
- Feature Selection: Cài đặt thuật toán chọn lọc đặc trưng (như Recursive Feature Elimination) dựa trên trọng số $W$ của mô hình Logistic Regression.

## Contributors
Phạm Thành Nam

## License
Dự án này chỉ được sử dụng cho mục đích học tập và nghiên cứu
