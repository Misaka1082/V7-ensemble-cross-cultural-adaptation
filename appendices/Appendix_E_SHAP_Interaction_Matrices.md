# Appendix E

## SHAP Interaction Value Matrices (Full)

SHAP interaction values decompose each prediction into pairwise feature contributions. Tables E1 and E3 present mean absolute SHAP interaction strengths; Tables E2 and E4 present permutation-test *p*-values (1,000 shuffles). Only the upper triangle is shown; the lower triangle is symmetric.

*Table E1*

*Mean Absolute SHAP Interaction Strength: Hong Kong Sample (N = 75)*

| Variable | CC | SC | SCn | FS | Op | CM | MHK | CF | CH | Au | SM |
|----------|------|------|------|------|------|------|------|------|------|------|------|
| CC | — | 0.007 | 0.131 | 0.128 | 0.085 | 0.023 | 0.009 | 0.003 | 0.158 | 0.060 | 0.021 |
| SC |  | — | 0.025 | 0.024 | 0.014 | 0.004 | 0.001 | 0.000 | 0.027 | 0.012 | 0.004 |
| SCn |  |  | — | 0.683 | 0.301 | 0.156 | 0.043 | 0.014 | 0.687 | 0.290 | 0.095 |
| FS |  |  |  | — | 0.337 | 0.123 | 0.039 | 0.015 | 0.697 | 0.308 | 0.104 |
| Op |  |  |  |  | — | 0.059 | 0.020 | 0.008 | 0.361 | 0.143 | 0.051 |
| CM |  |  |  |  |  | — | 0.009 | 0.003 | 0.125 | 0.048 | 0.015 |
| MHK |  |  |  |  |  |  | — | 0.001 | 0.040 | 0.015 | 0.004 |
| CF |  |  |  |  |  |  |  | — | 0.014 | 0.005 | 0.002 |
| CH |  |  |  |  |  |  |  |  | — | 0.306 | 0.109 |
| Au |  |  |  |  |  |  |  |  |  | — | 0.061 |
| SM |  |  |  |  |  |  |  |  |  |  | — |

*Note.* Values represent the mean absolute joint SHAP contribution of each feature pair, normalized to [0, 1]. The upper triangle shows interaction strength; the lower triangle is empty (symmetric). Permutation test *p*-values are presented in Table E2. CC = Cultural Contact; SC = Social Contact; SCn = Social Connectedness; FS = Family Support; Op = Openness; CM = Cultural Maintenance; MHK = Months in HK; CF = Communication Frequency; CH = Communication Honesty; Au = Autonomy; SM = Social Maintenance.

*Table E2*

*Permutation Test p-Values for SHAP Interaction Strength: Hong Kong Sample (N = 75)*

| Variable | CC | SC | SCn | FS | Op | CM | MHK | CF | CH | Au | SM |
|----------|------|------|------|------|------|------|------|------|------|------|------|
| CC | — | 0.000* | 0.072 | 0.168 | 0.000* | 0.066 | 0.005* | 0.006* | 0.005* | 0.584 | 0.804 |
| SC |  | — | 0.021* | 0.178 | 0.011* | 0.329 | 0.272 | 0.029* | 0.029* | 0.499 | 0.859 |
| SCn |  |  | — | 0.000* | 0.088 | 0.000* | 0.002* | 0.005* | 0.008* | 0.085 | 0.783 |
| FS |  |  |  | — | 0.001* | 0.000* | 0.002* | 0.000* | 0.006* | 0.018* | 0.662 |
| Op |  |  |  |  | — | 0.012* | 0.011* | 0.000* | 0.003* | 0.445 | 0.681 |
| CM |  |  |  |  |  | — | 0.000* | 0.000* | 0.006* | 0.222 | 0.913 |
| MHK |  |  |  |  |  |  | — | 0.005* | 0.025* | 0.672 | 0.998 |
| CF |  |  |  |  |  |  |  | — | 0.006* | 0.091 | 0.673 |
| CH |  |  |  |  |  |  |  |  | — | 0.584 | 0.828 |
| Au |  |  |  |  |  |  |  |  |  | — | 0.002* |
| SM |  |  |  |  |  |  |  |  |  |  | — |

*Note.* *p*-values derived from 1,000 permutation shuffles. \* *p* < .05 indicates a statistically significant interaction. The upper triangle shows *p*-values; the lower triangle is empty.

---

*Table E3*

*Mean Absolute SHAP Interaction Strength: France Sample (N = 249)*

| Variable | CC | SC | SCn | FS | Op | CM | MHK | CF | CH | Au | SM |
|----------|------|------|------|------|------|------|------|------|------|------|------|
| CC | — | 0.048 | 0.211 | 0.156 | 0.307 | 0.100 | 0.122 | 0.093 | 0.246 | 0.085 | 0.123 |
| SC |  | — | 0.103 | 0.074 | 0.095 | 0.039 | 0.060 | 0.041 | 0.088 | 0.042 | 0.074 |
| SCn |  |  | — | 0.507 | 0.450 | 0.216 | 0.369 | 0.223 | 0.489 | 0.242 | 0.455 |
| FS |  |  |  | — | 0.379 | 0.144 | 0.297 | 0.166 | 0.367 | 0.199 | 0.303 |
| Op |  |  |  |  | — | 0.192 | 0.333 | 0.200 | 0.506 | 0.204 | 0.253 |
| CM |  |  |  |  |  | — | 0.123 | 0.076 | 0.173 | 0.079 | 0.137 |
| MHK |  |  |  |  |  |  | — | 0.139 | 0.300 | 0.192 | 0.277 |
| CF |  |  |  |  |  |  |  | — | 0.188 | 0.097 | 0.161 |
| CH |  |  |  |  |  |  |  |  | — | 0.195 | 0.306 |
| Au |  |  |  |  |  |  |  |  |  | — | 0.189 |
| SM |  |  |  |  |  |  |  |  |  |  | — |

*Note.* Values represent the mean absolute joint SHAP contribution of each feature pair, normalized to [0, 1]. The upper triangle shows interaction strength; the lower triangle is empty (symmetric). Permutation test *p*-values are presented in Table E4. CC = Cultural Contact; SC = Social Contact; SCn = Social Connectedness; FS = Family Support; Op = Openness; CM = Cultural Maintenance; MHK = Months in HK; CF = Communication Frequency; CH = Communication Honesty; Au = Autonomy; SM = Social Maintenance.

*Table E4*

*Permutation Test p-Values for SHAP Interaction Strength: France Sample (N = 249)*

| Variable | CC | SC | SCn | FS | Op | CM | MHK | CF | CH | Au | SM |
|----------|------|------|------|------|------|------|------|------|------|------|------|
| CC | — | 0.006* | 0.979 | 0.937 | 0.000* | 0.001* | 0.943 | 0.097 | 0.000* | 0.911 | 1.000 |
| SC |  | — | 0.970 | 0.929 | 0.077 | 0.033* | 0.815 | 0.188 | 0.152 | 0.649 | 0.339 |
| SCn |  |  | — | 0.015* | 1.000 | 0.417 | 0.771 | 0.983 | 0.950 | 0.969 | 0.094 |
| FS |  |  |  | — | 0.388 | 0.872 | 0.005* | 0.750 | 0.353 | 0.030* | 0.753 |
| Op |  |  |  |  | — | 0.032* | 0.040* | 0.120 | 0.000* | 0.570 | 1.000 |
| CM |  |  |  |  |  | — | 0.457 | 0.555 | 0.087 | 0.820 | 0.645 |
| MHK |  |  |  |  |  |  | — | 0.314 | 0.218 | 0.000* | 0.003* |
| CF |  |  |  |  |  |  |  | — | 0.226 | 0.215 | 0.214 |
| CH |  |  |  |  |  |  |  |  | — | 0.745 | 0.999 |
| Au |  |  |  |  |  |  |  |  |  | — | 0.006* |
| SM |  |  |  |  |  |  |  |  |  |  | — |

*Note.* *p*-values derived from 1,000 permutation shuffles. \* *p* < .05 indicates a statistically significant interaction. The upper triangle shows *p*-values; the lower triangle is empty.
