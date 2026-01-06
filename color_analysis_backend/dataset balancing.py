import pandas as pd
import os

csv_in = r"D:\NMAI\data\processed\all_data.csv"
csv_out = r"D:\NMAI\data\processed\balanced_data.csv"

# đọc dữ liệu từ CSV gốc
df = pd.read_csv(csv_in)

# in thống kê số ảnh từng lớp trước khi cân bằng
print("Phân bố ban đầu:")
print(df['label'].value_counts())

# lấy số ảnh ít nhất trong các lớp để cân bằng
min_count = df['label'].value_counts().min()
print("\nSố ảnh mỗi lớp sau cân bằng:", min_count)

# under-sampling: lấy min_count ảnh cho mỗi lớp
balanced_df = (
    df.groupby("label", group_keys=False)
      .apply(lambda x: x.sample(min_count, random_state=42))
)

# lưu dữ liệu đã cân bằng ra CSV mới
balanced_df.to_csv(csv_out, index=False)

# in thống kê sau khi cân bằng
print("\nPhân bố sau cân bằng:")
print(balanced_df['label'].value_counts())
print("\nFile lưu tại:", csv_out)
