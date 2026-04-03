# ============================================
# 跨文化适应量表分析脚本 - 第四部分：跨文化等值性检验
# ============================================

# 加载之前保存的数据
load("analysis_data.RData")

# ============================================
# 12. 跨文化等值性检验
# ============================================

cat("\n========== 跨文化等值性检验 ==========\n")

# 使用跨文化适应程度量表作为示例
# 定义测量模型
measurement_model <- '
  # 因子定义
  adaptation =~ adapt1 + adapt2 + adapt3 + adapt4 + adapt5 + adapt6 + adapt7 + adapt8
'

# ============================================
# 12.1 配置等值性 (Configural Invariance)
# ============================================

cat("\n--- 配置等值性检验 ---\n")

configural_model <- measurement_model

configural_fit <- lavaan::cfa(
  configural_model,
  data = combined_data,
  group = "group",
  estimator = "ML",
  missing = "ml"
)

configural_fit_measures <- lavaan::fitMeasures(configural_fit)

cat("配置等值性模型拟合指标:\n")
cat("  χ²/df:", round(configural_fit_measures["chisq"] / configural_fit_measures["df"], 2), "\n")
cat("  CFI:", round(configural_fit_measures["cfi"], 3), "\n")
cat("  TLI:", round(configural_fit_measures["tli"], 3), "\n")
cat("  RMSEA:", round(configural_fit_measures["rmsea"], 3), "\n")
cat("  SRMR:", round(configural_fit_means["srmr"], 3), "\n")

# ============================================
# 12.2 度量等值性 (Metric Invariance)
# ============================================

cat("\n--- 度量等值性检验 ---\n")

metric_fit <- lavaan::cfa(
  measurement_model,
  data = combined_data,
  group = "group",
  group.equal = "loadings",  # 约束因子载荷相等
  estimator = "ML",
  missing = "ml"
)

metric_fit_measures <- lavaan::fitMeasures(metric_fit)

cat("度量等值性模型拟合指标:\n")
cat("  χ²/df:", round(metric_fit_measures["chisq"] / metric_fit_measures["df"], 2), "\n")
cat("  CFI:", round(metric_fit_measures["cfi"], 3), "\n")
cat("  TLI:", round(metric_fit_measures["tli"], 3), "\n")
cat("  RMSEA:", round(metric_fit_measures["rmsea"], 3), "\n")
cat("  SRMR:", round(metric_fit_measures["srmr"], 3), "\n")

# ============================================
# 12.3 标量等值性 (Scalar Invariance)
# ============================================

cat("\n--- 标量等值性检验 ---\n")

scalar_fit <- lavaan::cfa(
  measurement_model,
  data = combined_data,
  group = "group",
  group.equal = c("loadings", "intercepts"),  # 约束因子载荷和截距相等
  estimator = "ML",
  missing = "ml"
)

scalar_fit_measures <- lavaan::fitMeasures(scalar_fit)

cat("标量等值性模型拟合指标:\n")
cat("  χ²/df:", round(scalar_fit_measures["chisq"] / scalar_fit_measures["df"], 2), "\n")
cat("  CFI:", round(scalar_fit_measures["cfi"], 3), "\n")
cat("  TLI:", round(scalar_fit_measures["tli"], 3), "\n")
cat("  RMSEA:", round(scalar_fit_measures["rmsea"], 3), "\n")
cat("  SRMR:", round(scalar_fit_measures["srmr"], 3), "\n")

# ============================================
# 12.4 模型比较
# ============================================

cat("\n--- 模型比较 ---\n")

# 使用anova进行嵌套模型比较
cat("\n1. 配置等值性 vs 度量等值性:\n")
compare_config_metric <- lavaan::anova(configural_fit, metric_fit)
print(compare_config_metric)

cat("\n2. 度量等值性 vs 标量等值性:\n")
compare_metric_scalar <- lavaan::anova(metric_fit, scalar_fit)
print(compare_metric_scalar)

# ============================================
# 12.5 使用semTools进行更详细的等值性检验
# ============================================

cat("\n--- 使用semTools进行等值性检验 ---\n")

if (require(semTools)) {
  # 进行多组测量等值性检验
  invariance_test <- semTools::measurementInvariance(
    model = measurement_model,
    data = combined_data,
    group = "group",
    estimator = "ML",
    missing = "ml"
  )
  
  # 打印结果
  print(invariance_test)
  
  # 提取拟合指标比较
  cat("\n拟合指标比较:\n")
  fit_comparison <- semTools::compareFit(
    configural_fit,
    metric_fit,
    scalar_fit
  )
  print(fit_comparison)
}

# ============================================
# 12.6 等值性判断标准
# ============================================

cat("\n========== 等值性判断标准 ==========\n")
cat("\n1. 配置等值性:\n")
cat("   - 模型在两组中都能拟合\n")
cat("   - 因子结构相同\n")
cat("   - 是后续检验的基础\n")

cat("\n2. 度量等值性 (弱等值性):\n")
cat("   - 因子载荷相等\n")
cat("   - 允许进行跨组相关和回归分析\n")
cat("   - ΔCFI < 0.01, ΔRMSEA < 0.015\n")

cat("\n3. 标量等值性 (强等值性):\n")
cat("   - 因子载荷和条目截距都相等\n")
cat("   - 允许进行跨组均值比较\n")
cat("   - ΔCFI < 0.01, ΔRMSEA < 0.015\n")

# ============================================
# 13. 保存结果
# ============================================

save(configural_fit, metric_fit, scalar_fit,
     compare_config_metric, compare_metric_scalar,
     file = "invariance_analysis.RData")

cat("\n跨文化等值性检验完成！\n")

# ============================================
# 14. 生成报告摘要
# ============================================

cat("\n========== 分析报告摘要 ==========\n")
cat("\n数据概况:\n")
cat("  法国样本: n =", nrow(france_data), "\n")
cat("  香港样本: n =", nrow(hongkong_data), "\n")
cat("  总样本: n =", nrow(combined_data), "\n")

cat("\n主要发现:\n")
cat("  1. 条目分析: 检查了各量表的CITC和删除条目后的α值\n")
cat("  2. 信度分析: 计算了各量表的Cronbach's α系数\n")
cat("  3. 结构效度: 进行了KMO、Bartlett、EFA和CFA分析\n")
cat("  4. 跨文化等值性: 检验了配置、度量和标量等值性\n")

cat("\n建议:\n")
cat("  1. 检查条目分析结果，删除CITC < 0.3的条目\n")
cat("  2. 根据EFA结果调整量表结构\n")
cat("  3. 根据CFA结果评估模型拟合\n")
cat("  4. 根据等值性检验结果判断量表是否适合跨文化比较\n")