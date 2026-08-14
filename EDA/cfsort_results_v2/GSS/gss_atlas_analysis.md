We would like to share the results of our sensitivity analysis for the selected regions. 

In our rebuttal, we already mentioned that using an atlas of the same size selected with specificity in mind improves the UXM Tissue Concordance Score.   
We suggested that Syto should benefit from the more specific regions, following an observation that UXM and Syto per cell-type $R^2$ are correlated with Proxy Gap Specificity Score, which indirectly measures how well the on-target class is separated from the second most prevalent class. We now have experimental evidence confirming our assumption. We summarize our main findings in three observations:   

1. **Switching to GSS atlas improves results accross different labeling schemes.** The improvement magnitude differs but direction is concordant with UXM (see Table A).       

2. **The TCS gains are reinforced by calibration.** The range of TCS improvements comparing to equivalent Classifier-Deconvolver-Calibration grows 
from $[+0.015,  +0.037]$ for uncalibrated results to $[+0.040,  +0.060]$ for the best deconvolver under vector calibration scheme.     

3. **The UXM improvement stems mostly from eliminating Adipocytes and Fallopian Epithelial bias.** As can be seen from the Table A, change in TCS score 
is not as dramatic for Syto as for UXM. Table C reveals that this is not only due to the higher baseline for Syto, but also because Syto doesn't have the same bias to begin with, which takes us back to the argument that Syto is more robust to the choice of less specific regions.         

Additionally, we observed higher correlation between UXM and Syto under the GSS atlas ($0.894-0.907$ vs $0.775-0.809$ under U25 for the matched SWN deconvolver; $0.837-0.910$ vs $0.778-0.829$ across tested 15 Labeling-Classifier-Deconvolver combinations). We see two possible explanations: either both solutions are closer to the ground truth, or GSS-atlas increased specificity prompts both methods towards a common solution more aggressively.   

Our interpretation is subject to the limitations of TCS, which we briefly highlighted in our rebuttal and committed to explain in greater detail in our paper. 
One notable example: under the GSS atlas, UXM increases the fraction of Smooth Muscle allocated to Uterus and Ovary tissues from 0.185 to 0.624 on average, significantly 
boosting TCS for these two tissues. Syto increase is in the range from 0.103-0.435 to 0.192-0.792 depending on the labeling scheme and deconvolver. 
We see from examining each tissue in greater detail that the Soft Labeling scheme with pooling tends to allocate less mass to the non-dominant but sensible cell types despite lower TCS. 

**All TCS values below are macro-averaged over tissues**: the score is averaged within each of the 16 mapped tissues first, then over the 16 tissues, so that no tissue 
dominates by sample count (the tissues range from 6 to 29 samples out of 260). Confidence intervals are accordingly a cluster bootstrap that resamples tissues 
(4000 draws), and "improved" counts tissues rather than samples. 


| method | TCS U25 | TCS GSS | Δ | 95% CI | tissues improved |
|---|---|---|---|---|---|
| **UXM** | 0.603 | 0.694 | **+0.092** | [+0.030, +0.160] | 69% |
| Syto — Hard | 0.712 | 0.745 | +0.032 | [−0.026, +0.108] | 50% |
| Syto — Soft | 0.682 | 0.718 | +0.037 | [−0.016, +0.100] | 56% |
| Syto — Soft-NoPool | 0.713 | 0.728 | +0.015 | [−0.038, +0.072] | 44% |
> Table A: Comparison of TCS gains for UXM and best **uncalibrated** result for each labeling scheme 
> using Dismir classifier. UXM improves in 11 of the 16 tissues and is the only arm whose interval 
> excludes zero; Syto trades bigger gains in roughly half of the tissues over smaller losses in the rest. 
> With only 16 clusters the intervals are necessarily wide, so the Syto gains are directional rather 
> than individually significant.   


| scheme | uncalibrated | clip0 | simplex | vector |
|---|---|---|---|---|
| Hard | +0.032 | +0.040 | +0.023 | **+0.045** |
| Soft | +0.037 | +0.048 | +0.049† | **+0.060†** |
| Soft-NoPool | +0.015 | +0.025 | +0.021 | **+0.040** |
> Table B: Relative TCS gains per calibration scheme for Dismir classifier + SWN deconvolver. 
> The reported number is computed as $\overline{TCS}_{GSS}-\overline{TCS}_{U25}$, each term macro-averaged over tissues. 
> † marks the two cells whose tissue-cluster bootstrap interval excludes zero.       


| recipient | UXM | Hard | Soft | Soft-NoPool | direction |
|---|---|---|---|---|---|
| Adipocytes | **−0.049** | −0.014 | −0.026 | +0.012 | mixed |
| Colon-Fibro | **+0.030** | −0.014 | −0.009 | −0.025 | mixed |
| Breast-Basal-Ep | −0.021 | −0.013 | −0.013 | −0.013 | **all −** |
| Fallopian-Ep | **−0.050** | +0.002 | +0.000 | +0.003 | mixed |
| Pancreas-Delta | +0.015 | +0.011 | +0.012 | +0.014 | **all +** |
| Smooth-Musc | −0.002 | +0.011 | +0.008 | +0.015 | mixed |
| Pancreas-Duct | −0.016 | −0.004 | −0.004 | −0.003 | **all −** |
| Lung-Ep-Bron | −0.001 | −0.006 | −0.013 | −0.005 | **all −** |
| Liver-Hep | +0.003 | +0.004 | +0.007 | +0.008 | **all +** |
| Kidney-Ep | +0.004 | +0.004 | +0.005 | +0.006 | **all +** |
| Oligodend | −0.004 | −0.004 | −0.005 | −0.003 | **all −** |
| Thyroid-Ep | −0.006 | −0.001 | +0.007 | −0.002 | mixed |
| Breast-Luminal-Ep | +0.005 | −0.002 | −0.003 | −0.003 | mixed |
| Endothel | +0.001 | +0.003 | +0.001 | −0.006 | mixed |
> Table C: Per-recipient Δ (GSS − U25) of the macro-averaged off-target mass, ranked by pooled |Δ|. 
> Negative = the GSS atlas leaks less mass to that cell type. Each column sums to the total off-target 
> mass, i.e. to $1-\overline{TCS}$ of the corresponding arm (UXM 0.397 → 0.306; Hard 0.288 → 0.255; 
> Soft 0.318 → 0.282; Soft-NoPool 0.287 → 0.272).
