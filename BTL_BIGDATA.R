# 🚩 Cài đặt thư viện cần thiết
library(tidyr)
library(dplyr)
library(ggplot2)
library(caret)
library(rpart)
library(rpart.plot)
library(sparklyr)
library(Metrics)

# 🚩 Kết nối đến Spark
sc <- spark_connect(master = "local")

# 🚩 Thiết lập đường dẫn và đọc dữ liệu
file_path <- "sales2019clean.csv"
df_combined <- read.csv(file_path, stringsAsFactors = FALSE)

# 🚩 Chuyển đổi dữ liệu sang Spark DataFrame
df_spark <- copy_to(sc, df_combined, "df_combined", overwrite = TRUE)

# 🚩 Xử lý dữ liệu thiếu
df_spark <- df_spark %>% filter(!is.na(Quantity.Ordered) & !is.na(Price.Each))

# 🚩 Chuyển đổi kiểu dữ liệu và tính toán doanh số
df_spark <- df_spark %>%
  mutate(
    Quantity.Ordered = as.integer(Quantity.Ordered),
    Price.Each = as.numeric(Price.Each),
    Sales = Quantity.Ordered * Price.Each
  )

# 🚩 Trích xuất tháng và giờ đặt hàng
df_spark <- df_spark %>%
  mutate(Order.Date = as.POSIXct(Order.Date, format="%m/%d/%Y %H:%M"),
         Month = format(Order.Date, "%m"),
         Hours = format(Order.Date, "%H"))

# 🚩 Trích xuất thành phố từ địa chỉ
df_spark <- df_spark %>%
  mutate(City = sapply(strsplit(Purchase.Address, ","), 
                       function(x) ifelse(length(x) >= 2, trimws(x[2]), NA)))

# 🚩 Tổng hợp doanh số theo tháng
sales_value_month <- df_spark %>%
  group_by(Month) %>%
  summarise(Sales = sum(Sales, na.rm = TRUE)) %>%
  collect()

# 🚩 Tổng hợp doanh số theo thành phố
sales_value_city <- df_spark %>%
  group_by(City) %>%
  summarise(Sales = sum(Sales, na.rm = TRUE)) %>%
  collect()

# 🚩 Tổng hợp doanh số theo giờ
sales_value_hours <- df_spark %>%
  group_by(Hours) %>%
  summarise(Sales = sum(Sales, na.rm = TRUE)) %>%
  collect()

# 🚩 Vẽ biểu đồ trực quan hóa
par(mfrow=c(2, 2))

# 1. Biểu đồ doanh số theo tháng
barplot(sales_value_month$Sales, names.arg = sales_value_month$Month, 
        xlab = "Months", ylab = "Sales in USD", col = "blue", main = "Sales by Month")

# 2. Biểu đồ doanh số theo thành phố
barplot(sales_value_city$Sales, names.arg = sales_value_city$City, las = 2, 
        xlab = "Cities", ylab = "Sales in USD", col = "red", main = "Sales by City")

# 3. Biểu đồ doanh số theo giờ
plot(as.numeric(sales_value_hours$Hours), sales_value_hours$Sales, type = "o", 
     xlab = "Hours", ylab = "Sales in USD", xaxt='n', main = "Sales by Hour")
axis(1, at = as.numeric(sales_value_hours$Hours), labels = sales_value_hours$Hours)

# 4. Phân phối số lượng sản phẩm theo đơn hàng
all_products <- df_spark %>%
  group_by(Product) %>%
  summarise(Quantity.Ordered = sum(Quantity.Ordered, na.rm = TRUE)) %>%
  collect()

barplot(all_products$Quantity.Ordered, names.arg = all_products$Product, las = 2, 
        col = "green", xlab = "Products", ylab = "Quantity Ordered", main = "Product Demand")

# 🚩 Chuyển đổi df_spark thành R DataFrame để huấn luyện mô hình
df_combined_clean <- df_spark %>%
  filter(!is.na(Quantity.Ordered)) %>%
  collect()

# 🚩 Chia dữ liệu thành tập train/test
set.seed(42)
trainIndex <- createDataPartition(df_combined_clean$Quantity.Ordered, p = 0.7, list = FALSE)
train_data <- df_combined_clean[trainIndex, ]
test_data <- df_combined_clean[-trainIndex, ]

# 🚩 Mã hóa dữ liệu phân loại thành factor
train_data <- train_data %>% mutate(across(where(is.character), as.factor))
test_data <- test_data %>% mutate(across(where(is.character), as.factor))

# 🚩 Huấn luyện mô hình cây quyết định
model <- rpart(Quantity.Ordered ~ Month + Hours + City + Price.Each, 
               data = train_data, method = "anova")

# 🚩 Vẽ cây quyết định
rpart.plot(model)

# 🚩 Dự đoán trên tập test
y_pred <- predict(model, test_data)

# 🚩 Tính toán độ chính xác
accuracy <- cor(y_pred, test_data$Quantity.Ordered)
print(paste("Độ chính xác của mô hình (tương quan Pearson):", round(accuracy, 2)))

# 🚩 Đánh giá hiệu suất mô hình
rmse_value <- rmse(test_data$Quantity.Ordered, y_pred)
print(paste("Giá trị RMSE của mô hình:", round(rmse_value, 2)))

# 🚩 Xuất dữ liệu đã xử lý
write.csv(df_combined_clean, "sales2019final_clean.csv", row.names = FALSE)

# 🚩 Ngắt kết nối Spark
spark_disconnect(sc)