ĐỀ TÀI: XÂY DỰNG MÔ HÌNH DỰ ĐOÁN DOANH SỐ BÁN HÀNG ĐIỆN TỬ

Kết quả đạt được: Biểu đồ doanh số theo tháng

Biểu đồ doanh số theo thành phố

Biểu đồ doanh số theo giờ

Biểu đồ phân phối số lượng sản phẩm

Rplot01

🚩 Cài đặt thư viện cần thiết
Để bắt đầu, bạn cần cài đặt các thư viện cần thiết

library(tidyr)
library(dplyr)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(sparklyr)
library(Metrics)
🚩 Kết nối đến Spark
Kết nối đến Spark với lệnh sau:
sc <- spark_connect(master = "local")

🚩 Thiết lập đường dẫn và đọc dữ liệu
Thiết lập đường dẫn đến file CSV và đọc dữ liệu:
file_path <- "sales2019clean.csv"
df_combined <- read.csv(file_path, stringsAsFactors = FALSE)

🚩 Chuyển đổi dữ liệu sang Spark DataFrame
Chuyển đổi dữ liệu sang Spark DataFrame:

df_spark <- copy_to(sc, df_combined, "df_combined", overwrite = TRUE)

🚩 Xử lý dữ liệu thiếu
Lọc ra các hàng có dữ liệu thiếu:
df_spark <- df_spark %>% filter(!is.na('Quantity.Ordered') & !is.na('Price.Each'))

🚩 Lưu trữ DataFrame trong bộ nhớ
Lưu trữ DataFrame trong bộ nhớ Spark:

df_spark <- df_spark %>% sparklyr::sdf_persist()

🚩 Kiểm tra dữ liệu trong Spark
df_spark %>%
  summarize(count = n()) %>%
  collect()

🚩 Chuyển đổi kiểu dữ liệu và tính toán doanh số


df_combined <- df_combined %>%
  filter(!is.na(Quantity.Ordered) & !is.na(Price.Each)) %>%
  mutate(
    Quantity.Ordered = as.integer(Quantity.Ordered),
    Price.Each = as.numeric(Price.Each),
    Sales = Quantity.Ordered * Price.Each
  )

🚩 Trích xuất tháng và giờ đặt hàng
Trích xuất tháng và giờ từ cột Order.Date:

df_combined$Order.Date <- as.POSIXct(df_combined$Order.Date, format="%m/%d/%Y %H:%M")
df_combined$Month <- format(df_combined$Order.Date, "%m")
df_combined$Hours <- format(df_combined$Order.Date, "%H")

🚩 Trích xuất thành phố từ địa chỉ
Trích xuất tên thành phố từ Purchase.Address:

df_combined$City <- sapply(strsplit(df_combined$Purchase.Address, ","), 
                           function(x) ifelse(length(x) >= 2, trimws(x[2]), NA))
🚩 Tổng hợp doanh số theo tháng, thành phố và giờ
Tổng hợp doanh số:

sales_value_month <- aggregate(Sales ~ Month, data = df_combined, sum)
sales_value_city <- aggregate(Sales ~ City, data = df_combined, sum)
sales_value_hours <- aggregate(Sales ~ Hours, data = df_combined, sum)

🚩 Vẽ biểu đồ trực quan hóa
Vẽ biểu đồ doanh số theo tháng, thành phố, và giờ:


par(mfrow=c(2, 2))
# Biểu đồ doanh số theo tháng
barplot(sales_value_month$Sales, names.arg = sales_value_month$Month, 
        xlab = "Months", ylab = "Sales in USD", col = "blue", main = "Sales by Month")


# Biểu đồ doanh số theo thành phố
barplot(sales_value_city$Sales, names.arg = sales_value_city$City, las = 2, 
        xlab = "Cities", ylab = "Sales in USD", col = "red", main = "Sales by City")
# Biểu đồ doanh số theo giờ
plot(as.numeric(sales_value_hours$Hours), sales_value_hours$Sales, type = "o", 
     xlab = "Hours", ylab = "Sales in USD", xaxt='n', main = "Sales by Hour")
axis(1, at = as.numeric(sales_value_hours$Hours), labels = sales_value_hours$Hours)
# Biểu đồ phân phối số lượng sản phẩm
all_products <- aggregate(Quantity.Ordered ~ Product, data = df_combined, sum)
barplot(all_products$Quantity.Ordered, names.arg = all_products$Product, las = 2,col = "green", xlab = "Products", ylab = "Quantity Ordered", main = "Product Demand")

🚩 Xây dựng mô hình dự đoán doanh số
Chia dữ liệu thành tập train/test:

set.seed(42)
trainIndex <- createDataPartition(df_combined$Quantity.Ordered, p = 0.7, list = FALSE)
train_data <- df_combined[trainIndex, ]
test_data <- df_combined[-trainIndex, ]

🚩 Huấn luyện mô hình cây quyết định
model <- rpart(Quantity.Ordered ~ Month + Hours + City + Price.Each, 
               data = train_data, method = "anova")
rpart.plot(model)

🚩 Dự đoán và đánh giá mô hình
y_pred <- predict(model, test_data)
accuracy <- cor(y_pred, test_data$Quantity.Ordered)
print(paste("Độ chính xác của mô hình (tương quan Pearson):", round(accuracy, 1)))
