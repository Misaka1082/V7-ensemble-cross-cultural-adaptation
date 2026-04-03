# ============================================
# 跨文化适应量表分析脚本 - 第一部分：数据准备和描述性统计
# ============================================

# 加载必要的包
library(psych)      # 用于因子分析、信度分析
library(lavaan)     # 用于验证性因子分析
library(semTools)   # 用于测量等值性检验
library(ggplot2)    # 用于绘图
library(dplyr)      # 用于数据处理

# ============================================
# 1. 读取数据
# ============================================

# 读取法国数据
france_data <- read.csv("FranchN=249.csv", encoding = "GBK")

# 读取香港数据
hongkong_data <- readxl::read_excel("HKN=75.xlsx")

# 查看数据结构
cat("法国数据维度:", dim(france_data), "\n")
cat("香港数据维度:", dim(hongkong_data), "\n")

# 查看列名
cat("\n法国数据列名:\n")
print(names(france_data))

cat("\n香港数据列名:\n")
print(names(hongkong_data))

# ============================================
# 2. 数据预处理
# ============================================

# 重命名列名以便统一分析
# 法国数据列名
france_cols <- c(
  "duration",                    # 来法国生活时长
  "adapt1", "adapt2", "adapt3", "adapt4", "adapt5", "adapt6", "adapt7", "adapt8",  # 跨文化适应程度 8个条目
  "culture_maintain",           # 文化保持
  "social_maintain",            # 社会保持
  "culture_contact",            # 文化接触
  "social_contact",             # 社会接触
  "family1", "family2", "family3", "family4", "family5", "family6", "family7", "family8",  # 家庭支持 8个条目
  "comm_freq",                  # 家庭沟通频率
  "comm_openness",              # 沟通坦诚度
  "autonomy1", "autonomy2",     # 自主性 2个条目
  "connection1", "connection2", "connection3", "connection4", "connection5",  # 社会联结感 5个条目
  "openness"                    # 开放性
)

# 香港数据列名
hongkong_cols <- c(
  "duration",                    # 来港时长
  "adapt1", "adapt2", "adapt3", "adapt4", "adapt5", "adapt6", "adapt7", "adapt8",  # 跨文化适应程度 8个条目
  "culture_maintain",           # 文化保持
  "social_maintain",            # 社会保持
  "culture_contact",            # 文化接触
  "social_contact",             # 社会接触
  "family1", "family2", "family3", "family4", "family5", "family6", "family7", "family8",  # 家庭支持 8个条目
  "comm_freq",                  # 沟通频率
  "comm_openness",              # 沟通的坦诚度
  "autonomy1", "autonomy2",     # 自主性 2个条目
  "connection1", "connection2", "connection3", "connection4", "connection5",  # 社会联结感 5个条目
  "openness"                    # 开放性
)

# 重命名列
names(france_data) <- france_cols
names(hongkong_data) <- hongkong_cols

# 添加组别标识
france_data$group <- "France"
hongkong_data$group <- "HongKong"

# 合并数据用于跨组分析
combined_data <- rbind(
  france_data[, c("group", france_cols)],
  hongkong_data[, c("group", hongkong_cols)]
)

# ============================================
# 3. 描述性统计
# ============================================

cat("\n========== 描述性统计 ==========\n")

# 法国数据描述性统计
cat("\n法国数据描述性统计:\n")
print(psych::describe(france_data[, -which(names(france_data) == "group")]))

# 香港数据描述性统计
cat("\n香港数据描述性统计:\n")
print(psych::describe(hongkong_data[, -which(names(hongkong_data) == "group")]))

# ============================================
# 4. 定义量表结构
# ============================================

# 定义各个量表的条目
scales <- list(
  adaptation = c("adapt1", "adapt2", "adapt3", "adapt4", "adapt5", "adapt6", "adapt7", "adapt8"),  # 跨文化适应程度
  maintenance = c("culture_maintain", "social_maintain"),  # 保持
  contact = c("culture_contact", "social_contact"),  # 接触
  family_support = c("family1", "family2", "family3", "family4", "family5", "family6", "family7", "family8"),  # 家庭支持
  communication = c("comm_freq", "comm_openness"),  # 沟通
  autonomy = c("autonomy1", "autonomy2"),  # 自主性
  connection = c("connection1", "connection2", "connection3", "connection4", "connection5"),  # 社会联结感
  openness = "openness"  # 开放性
)

# 保存数据
save(france_data, hongkong_data, combined_data, scales, 
     file = "analysis_data.RData")

cat("\n数据准备完成！\n")