# ============================================
# 跨文化适应量表分析脚本 - 第二部分：条目分析和信度分析
# ============================================

# 加载第一部分保存的数据
load("analysis_data.RData")

# ============================================
# 5. 条目分析 (CITC和删除条目后的α值)
# ============================================

cat("\n========== 条目分析 (CITC) ==========\n")

# 定义一个函数进行条目分析
item_analysis <- function(data, scale_name, items) {
  cat("\n量表:", scale_name, "\n")
  cat("条目:", paste(items, collapse = ", "), "\n")
  
  if (length(items) < 2) {
    cat("条目数不足，无法计算CITC和α值\n")
    return(NULL)
  }
  
  # 计算总分
  total_score <- rowSums(data[, items], na.rm = TRUE)
  
  # 计算CITC
  citc_results <- data.frame(
    item = items,
    citc = sapply(items, function(item) {
      cor(data[, item], total_score - data[, item], use = "complete.obs")
    })
  )
  
  # 计算整体α值
  alpha_total <- psych::alpha(data[, items])
  
  # 计算删除每个条目后的α值
  alpha_if_deleted <- sapply(items, function(item) {
    remaining_items <- setdiff(items, item)
    if (length(remaining_items) >= 2) {
      psych::alpha(data[, remaining_items])$total$raw_alpha
    } else {
      NA
    }
  })
  
  citc_results$alpha_if_deleted <- alpha_if_deleted
  citc_results$alpha_total <- alpha_total$total$raw_alpha
  
  # 打印结果
  print(citc_results)
  
  # 判断标准：CITC > 0.3，删除条目后α值不增加
  citc_results$acceptable <- citc_results$citc > 0.3
  
  return(citc_results)
}

# 对每个量表进行条目分析（法国数据）
cat("\n--- 法国数据条目分析 ---\n")
france_item_results <- list()

for (scale_name in names(scales)) {
  items <- scales[[scale_name]]
  if (length(items) >= 2) {
    france_item_results[[scale_name]] <- item_analysis(france_data, scale_name, items)
  }
}

# 对每个量表进行条目分析（香港数据）
cat("\n--- 香港数据条目分析 ---\n")
hongkong_item_results <- list()

for (scale_name in names(scales)) {
  items <- scales[[scale_name]]
  if (length(items) >= 2) {
    hongkong_item_results[[scale_name]] <- item_analysis(hongkong_data, scale_name, items)
  }
}

# ============================================
# 6. 信度分析 (Cronbach's α)
# ============================================

cat("\n========== 信度分析 (Cronbach's α) ==========\n")

# 定义一个函数计算信度
calculate_reliability <- function(data, scale_name, items) {
  if (length(items) < 2) {
    return(list(alpha = NA, items = length(items)))
  }
  
  alpha_result <- psych::alpha(data[, items])
  return(list(
    alpha = alpha_result$total$raw_alpha,
    items = length(items),
    result = alpha_result
  ))
}

# 法国数据信度分析
cat("\n--- 法国数据信度分析 ---\n")
france_reliability <- list()

for (scale_name in names(scales)) {
  items <- scales[[scale_name]]
  reliability <- calculate_reliability(france_data, scale_name, items)
  france_reliability[[scale_name]] <- reliability
  
  cat(scale_name, ": α =", round(reliability$alpha, 3), 
      "(n =", reliability$items, "items)\n")
}

# 香港数据信度分析
cat("\n--- 香港数据信度分析 ---\n")
hongkong_reliability <- list()

for (scale_name in names(scales)) {
  items <- scales[[scale_name]]
  reliability <- calculate_reliability(hongkong_data, scale_name, items)
  hongkong_reliability[[scale_name]] <- reliability
  
  cat(scale_name, ": α =", round(reliability$alpha, 3), 
      "(n =", reliability$items, "items)\n")
}

# ============================================
# 7. 创建信度分析汇总表
# ============================================

# 创建汇总表格
reliability_summary <- data.frame(
  scale = names(scales),
  france_alpha = sapply(names(scales), function(x) {
    if (!is.null(france_reliability[[x]]$alpha)) {
      round(france_reliability[[x]]$alpha, 3)
    } else {
      NA
    }
  }),
  france_items = sapply(names(scales), function(x) {
    if (!is.null(france_reliability[[x]]$items)) {
      france_reliability[[x]]$items
    } else {
      NA
    }
  }),
  hongkong_alpha = sapply(names(scales), function(x) {
    if (!is.null(hongkong_reliability[[x]]$alpha)) {
      round(hongkong_reliability[[x]]$alpha, 3)
    } else {
      NA
    }
  }),
  hongkong_items = sapply(names(scales), function(x) {
    if (!is.null(hongkong_reliability[[x]]$items)) {
      hongkong_reliability[[x]]$items
    } else {
      NA
    }
  })
)

cat("\n========== 信度分析汇总表 ==========\n")
print(reliability_summary)

# 保存结果
save(france_item_results, hongkong_item_results,
     france_reliability, hongkong_reliability, reliability_summary,
     file = "reliability_analysis.RData")

cat("\n条目分析和信度分析完成！\n")