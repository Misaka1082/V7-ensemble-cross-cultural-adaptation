# ============================================
# 跨文化适应量表分析脚本 - 第三部分：结构效度检验
# ============================================

# 加载之前保存的数据
load("analysis_data.RData")

# ============================================
# 8. KMO和Bartlett球形检验
# ============================================

cat("\n========== KMO和Bartlett球形检验 ==========\n")

# 定义一个函数进行KMO和Bartlett检验
check_factorability <- function(data, items, group_name = "") {
  cat("\n---", group_name, "---\n")
  
  if (length(items) < 3) {
    cat("条目数不足，无法进行因子分析\n")
    return(NULL)
  }
  
  # 提取相关数据
  scale_data <- data[, items]
  
  # 检查是否有缺失值
  if (any(is.na(scale_data))) {
    cat("数据中存在缺失值，将使用成对删除\n")
  }
  
  # KMO检验
  kmo_result <- psych::KMO(scale_data)
  cat("KMO值:", round(kmo_result$MSA, 3), "\n")
  
  # Bartlett球形检验
  bartlett_result <- psych::cortest.bartlett(scale_data)
  cat("Bartlett球形检验:\n")
  cat("  χ² =", round(bartlett_result$chisq, 2), "\n")
  cat("  df =", bartlett_result$df, "\n")
  cat("  p =", format.pval(bartlett_result$p.value, digits = 3), "\n")
  
  # 判断标准
  cat("\n判断标准:\n")
  cat("  KMO > 0.6: 适合因子分析\n")
  cat("  Bartlett p < 0.05: 适合因子分析\n")
  
  return(list(kmo = kmo_result, bartlett = bartlett_result))
}

# 对主要量表进行检验（跨文化适应程度量表）
cat("\n主要量表：跨文化适应程度 (8个条目)\n")

# 法国数据
france_factorability <- check_factorability(
  france_data, 
  scales$adaptation, 
  "法国数据"
)

# 香港数据
hongkong_factorability <- check_factorability(
  hongkong_data, 
  scales$adaptation, 
  "香港数据"
)

# ============================================
# 9. 探索性因子分析 (EFA)
# ============================================

cat("\n========== 探索性因子分析 (EFA) ==========\n")

# 定义一个函数进行EFA
perform_efa <- function(data, items, group_name = "", nfactors = NULL) {
  cat("\n---", group_name, "---\n")
  
  if (length(items) < 3) {
    cat("条目数不足，无法进行EFA\n")
    return(NULL)
  }
  
  scale_data <- data[, items]
  
  # 确定因子数（如果未指定）
  if (is.null(nfactors)) {
    # 使用平行分析确定因子数
    parallel <- psych::fa.parallel(scale_data, fa = "fa", plot = FALSE)
    nfactors <- parallel$nfact
    cat("平行分析建议的因子数:", nfactors, "\n")
  }
  
  # 进行EFA（使用主轴因子法）
  efa_result <- psych::fa(
    scale_data,
    nfactors = nfactors,
    rotate = "promax",  # 使用斜交旋转
    fm = "pa"  # 主轴因子法
  )
  
  # 打印结果
  cat("\n因子载荷矩阵:\n")
  print(round(efa_result$loadings, 3))
  
  cat("\n因子相关性:\n")
  if (!is.null(efa_result$Phi)) {
    print(round(efa_result$Phi, 3))
  }
  
  cat("\n模型拟合指标:\n")
  cat("  TLI (NNFI):", round(efa_result$TLI, 3), "\n")
  cat("  RMSEA:", round(efa_result$RMSEA[1], 3), "\n")
  cat("  RMSR:", round(efa_result$rms, 3), "\n")
  
  return(efa_result)
}

# 对跨文化适应程度量表进行EFA
cat("\n量表：跨文化适应程度\n")

# 法国数据EFA
france_efa <- perform_efa(
  france_data,
  scales$adaptation,
  "法国数据"
)

# 香港数据EFA（样本量较小，可能需要谨慎解释）
hongkong_efa <- perform_efa(
  hongkong_data,
  scales$adaptation,
  "香港数据"
)

# ============================================
# 10. 验证性因子分析 (CFA)
# ============================================

cat("\n========== 验证性因子分析 (CFA) ==========\n")

# 定义CFA模型（以跨文化适应程度为例）
# 假设为单因子模型
adaptation_model <- '
  # 因子定义
  adaptation =~ adapt1 + adapt2 + adapt3 + adapt4 + adapt5 + adapt6 + adapt7 + adapt8
'

# 定义一个函数进行CFA
perform_cfa <- function(data, model, group_name = "") {
  cat("\n---", group_name, "---\n")
  
  # 进行CFA
  cfa_result <- lavaan::cfa(
    model,
    data = data,
    estimator = "ML",  # 最大似然估计
    missing = "ml"     # 使用全信息最大似然处理缺失值
  )
  
  # 打印拟合指标
  fit_measures <- lavaan::fitMeasures(cfa_result)
  
  cat("\n模型拟合指标:\n")
  cat("  χ²/df:", round(fit_measures["chisq"] / fit_measures["df"], 2), "\n")
  cat("  CFI:", round(fit_measures["cfi"], 3), "\n")
  cat("  TLI:", round(fit_measures["tli"], 3), "\n")
  cat("  RMSEA:", round(fit_measures["rmsea"], 3), "\n")
  cat("  SRMR:", round(fit_measures["srmr"], 3), "\n")
  
  # 判断标准
  cat("\n判断标准:\n")
  cat("  χ²/df < 3: 良好拟合\n")
  cat("  CFI > 0.90: 可接受拟合; > 0.95: 良好拟合\n")
  cat("  TLI > 0.90: 可接受拟合; > 0.95: 良好拟合\n")
  cat("  RMSEA < 0.08: 可接受拟合; < 0.06: 良好拟合\n")
  cat("  SRMR < 0.08: 良好拟合\n")
  
  # 打印标准化因子载荷
  cat("\n标准化因子载荷:\n")
  standardized_solution <- lavaan::standardizedSolution(cfa_result)
  factor_loadings <- standardized_solution[standardized_solution$op == "=~", ]
  print(round(factor_loadings[, c("lhs", "rhs", "est.std")], 3))
  
  return(cfa_result)
}

# 对跨文化适应程度量表进行CFA
cat("\n量表：跨文化适应程度 (单因子模型)\n")

# 法国数据CFA
france_cfa <- perform_cfa(
  france_data,
  adaptation_model,
  "法国数据"
)

# 香港数据CFA
hongkong_cfa <- perform_cfa(
  hongkong_data,
  adaptation_model,
  "香港数据"
)

# ============================================
# 11. 保存结果
# ============================================

save(france_factorability, hongkong_factorability,
     france_efa, hongkong_efa,
     france_cfa, hongkong_cfa,
     file = "validity_analysis.RData")

cat("\n结构效度检验完成！\n")