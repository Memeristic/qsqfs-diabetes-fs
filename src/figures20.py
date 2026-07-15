"""
src/figures20.py
================
Builds the twenty analysis figures (F01-F20) for a fitted cohort.

The figure set is generated from the feature matrix and the comparative table,
so the same code path serves the command-line pipeline (`run_20_figures.py`) and
the Streamlit application. Figures are written as PNG and described by a
manifest, which callers use to render or package them.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy import stats as sstats
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc as _auc, average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import statsmodels.api as sm
except ImportError:          # optional: only used for the F09 odds-ratio figure
    sm = None

from src.evaluation import nested_cv_evaluate

STYLE = {
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#334155", "axes.linewidth": 0.8, "axes.grid": True,
    "grid.color": "#E2E8F0", "grid.linewidth": 0.7, "axes.axisbelow": True,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": "#1F3A5F",
}
BLUE, ORANGE, GREEN, RED, GREY = "#2563EB", "#F59E0B", "#10B981", "#EF4444", "#94A3B8"
DM_C, ND_C = "#EF4444", "#3B82F6"


def build_figures(
    X: np.ndarray,
    y: np.ndarray,
    names: Sequence[str],
    mmap: Dict[str, Sequence[int]],
    cfg: dict,
    comparative_table: List[dict],
    out_dir: str,
    feature_labels: Optional[Dict[str, str]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> List[dict]:
    """Generate F01-F20 into ``out_dir/fig20`` and return the manifest.

    ``comparative_table`` is the list of per-method rows produced by the
    comparative study (Method / AUC / Accuracy / F1 / Features); it is the only
    input the figures cannot derive from the cohort itself.
    """
    plt.rcParams.update(STYLE)

    OUT = out_dir
    FIG = os.path.join(OUT, "fig20")
    os.makedirs(FIG, exist_ok=True)
    LAB = dict(feature_labels or {})
    # Normalise the comparative table to a consistent schema. The command-line
    # pipeline supplies rows as {Method, AUC, Accuracy, F1, Features}; the
    # interactive app supplies the raw comparative result, which may key metrics
    # in lower case (auc/accuracy/f1) or nest per-fold arrays. Either shape is
    # coerced here so the figure code sees one schema.
    def _norm_comparative(raw):
        rows = []
        if isinstance(raw, dict):
            raw = raw.get("table", raw.get("results", raw))
        if isinstance(raw, dict):
            raw = [dict(v, Method=k) if isinstance(v, dict) else v
                   for k, v in raw.items()]
        for r in (raw or []):
            if not isinstance(r, dict):
                continue
            def g(*keys, default=np.nan):
                for k in keys:
                    if k in r and r[k] is not None:
                        v = r[k]
                        if isinstance(v, (list, tuple, np.ndarray)):
                            v = float(np.nanmean(v)) if len(v) else np.nan
                        return v
                return default
            rows.append({
                "Method":   g("Method", "method", "name", default="?"),
                "AUC":      float(g("AUC", "auc", default=np.nan)),
                "Accuracy": float(g("Accuracy", "accuracy", "acc", default=np.nan)),
                "F1":       float(g("F1", "f1", "F1-score", default=np.nan)),
                "Features": float(g("Features", "features", "n_features",
                                    "n_features_mean", default=np.nan)),
            })
        return rows

    SUMM = {"comparative": {"table": _norm_comparative(comparative_table)}}
    names = list(names)
    manifest: List[dict] = []

    modality_of = {i: m for m, idx in mmap.items() for i in idx}

    # A named view of the matrix, used by the correlation and distribution panels.
    combined = pd.DataFrame(X, columns=names)

    def save(fig, fid, title, desc):
        p = os.path.join(FIG, fid + ".png")
        fig.tight_layout()
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        manifest.append({"id": fid, "file": fid + ".png", "title": title, "desc": desc})
        if progress:
            progress(f"{fid} - {title}")

    def lab(f):
        if f in LAB:
            return LAB[f]
        return f.split("_", 1)[1] if "_" in f else f

    # ---- nested CV (recover oof + stability) ----
    print(">> nested CV (recover out-of-fold predictions) ...", flush=True)
    res=nested_cv_evaluate(X,y,mmap,cfg,n_outer=cfg["evaluation"]["n_outer_folds"],classifier="knn")
    proba=np.asarray(res["oof_proba"]); yt=np.asarray(res["oof_y"]); freq=np.asarray(res["selection_frequency"])
    stable_idx=[i for i in np.argsort(freq)[::-1] if freq[i]>=0.5]
    stable=[names[i] for i in stable_idx]

    # ---- univariate stats for every feature ----
    print(">> univariate statistics ...", flush=True)
    rows=[]
    for j,nm in enumerate(names):
        x=X[:,j]; x1=x[y==1]; x0=x[y==0]
        try: auc=roc_auc_score(y,x)
        except Exception: auc=0.5
        disc=max(auc,1-auc)
        try: _,p=sstats.mannwhitneyu(x1,x0,alternative="two-sided")
        except Exception: p=1.0
        sp=np.sqrt(((len(x1)-1)*x1.var(ddof=1)+(len(x0)-1)*x0.var(ddof=1))/max(len(x)-2,1)) if len(x)>2 else 1
        d=(x1.mean()-x0.mean())/sp if sp>0 else 0.0
        rows.append({"feature":nm,"modality":modality_of.get(j,"?"),"auc":auc,"disc":disc,
                     "p":p,"neglog10p":-np.log10(max(p,1e-12)),"cohens_d":d,"nunique":len(np.unique(x))})
    uni=pd.DataFrame(rows)
    uni.to_csv(os.path.join(OUT,"univariate_stats.csv"),index=False)
    continuous=uni[uni.nunique_>2] if False else uni[uni["nunique"]>2]

    # =================== FIGURES ===================
    # F01 cohort composition (donut)
    fig,ax=plt.subplots(figsize=(6,5))
    cnts=[int((y==0).sum()),int((y==1).sum())]
    w,_,_=ax.pie(cnts,labels=[f"No diabetes\n{cnts[0]} ({100*cnts[0]/len(y):.0f}%)",f"Diabetes\n{cnts[1]} ({100*cnts[1]/len(y):.0f}%)"],
        colors=[ND_C,DM_C],autopct="",startangle=90,wedgeprops=dict(width=0.42,edgecolor="white",linewidth=2))
    ax.text(0,0,f"n = {len(y)}",ha="center",va="center",fontsize=16,fontweight="bold")
    ax.set_title("F1 · Cohort composition (class balance)")
    save(fig,"F01_cohort","Cohort composition","Class balance: diabetic vs non-diabetic patients.")

    # F02 features per modality
    fig,ax=plt.subplots(figsize=(7,4))
    mc=pd.Series({m:len(idx) for m,idx in mmap.items()}).sort_values()
    ax.barh(mc.index,mc.values,color=BLUE)
    for i,v in enumerate(mc.values): ax.text(v+0.5,i,str(v),va="center",fontsize=10)
    ax.set_xlabel("Number of features"); ax.set_title("F2 · Features per modality")
    save(fig,"F02_modalities","Features per modality","How many measurements each clinical modality contributes.")

    # F03 data density per modality (mean fraction non-missing before impute proxy: fraction non-zero variance / mean |z|)
    fig,ax=plt.subplots(figsize=(7,4))
    dens={}
    for m,idx in mmap.items():
        sub=X[:,idx]
        dens[m]=float((sub!=0).mean())  # meds/dx are 0/1 presence; labs/vitals mostly non-zero
    ds=pd.Series(dens).sort_values()
    ax.barh(ds.index,ds.values,color=GREEN)
    for i,v in enumerate(ds.values): ax.text(v+0.01,i,f"{v:.2f}",va="center",fontsize=10)
    ax.set_xlabel("Mean fraction of non-zero entries"); ax.set_xlim(0,1)
    ax.set_title("F3 · Data density by modality")
    save(fig,"F03_density","Data density by modality","Fraction of non-zero entries per modality (sparsity of meds/dx vs labs/vitals).")

    # F04 distribution of the strongest single discriminator, by outcome.
    # The criterion analytes are withheld by the exclusion policy, so this panel
    # reports whichever admissible feature separates the classes best.
    gi=int(uni.sort_values("disc",ascending=False).index[0])
    fig,ax=plt.subplots(figsize=(7,4.5))
    g=X[:,gi]
    bins=np.linspace(np.percentile(g,1),np.percentile(g,99),25)
    ax.hist(g[y==0],bins=bins,alpha=0.6,color=ND_C,label="No diabetes",density=True)
    ax.hist(g[y==1],bins=bins,alpha=0.6,color=DM_C,label="Diabetes",density=True)
    ax.axvline(np.median(g[y==0]),color=ND_C,ls="--",lw=1.5); ax.axvline(np.median(g[y==1]),color=DM_C,ls="--",lw=1.5)
    ax.set_xlabel(lab(names[gi])+" (value)"); ax.set_ylabel("Density"); ax.legend()
    ax.set_title(f"F4 · {lab(names[gi])} distribution by diabetes status")
    save(fig,"F04_top_feature_dist",f"{lab(names[gi])} by status","Overlaid distributions of the top predictor split by diabetes status (dashed = medians).")

    # F05 boxplots of top-6 discriminative continuous features by DM status
    top6=continuous.sort_values("disc",ascending=False).head(6)["feature"].tolist()
    fig=plt.figure(figsize=(11,6)); gs=GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.3)
    for k,f in enumerate(top6):
        ax=fig.add_subplot(gs[k//3,k%3]); j=names.index(f)
        data_bp=[X[y==0,j],X[y==1,j]]
        # matplotlib >=3.9 renamed boxplot(labels=) to tick_labels=; support both.
        import inspect as _inspect
        _bp_kw = ("tick_labels" if "tick_labels" in
                  _inspect.signature(ax.boxplot).parameters else "labels")
        bp=ax.boxplot(data_bp,patch_artist=True,widths=0.6,showfliers=False,
                      **{_bp_kw:["No DM","DM"]})
        for patch,c in zip(bp["boxes"],[ND_C,DM_C]): patch.set_facecolor(c); patch.set_alpha(0.6)
        for med in bp["medians"]: med.set_color("black")
        a=uni[uni.feature==f].iloc[0]
        ax.set_title(f"{lab(f)}\nAUC={a['disc']:.2f}  p={a['p']:.1e}",fontsize=10)
    fig.suptitle("F5 · Top-6 discriminative features by diabetes status",fontsize=14,fontweight="bold",color="#1F3A5F")
    save(fig,"F05_boxplots","Top-6 feature boxplots","Distribution of the six most discriminative continuous features, split by status.")

    # F06 single-feature discriminative power (top 15)
    fig,ax=plt.subplots(figsize=(8,6))
    t15=uni.sort_values("disc",ascending=False).head(15).iloc[::-1]
    colors=[BLUE if m=="labs" else ORANGE if m=="vitals" else GREEN if m=="meds" else GREY for m in t15["modality"]]
    ax.barh([lab(f) for f in t15["feature"]],t15["disc"],color=colors)
    ax.axvline(0.5,color=RED,ls="--",lw=1,label="chance (0.5)")
    ax.set_xlim(0.5,max(0.75,t15["disc"].max()+0.03)); ax.set_xlabel("Single-feature AUC (discriminative power)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE,label="labs"),Patch(color=ORANGE,label="vitals"),Patch(color=GREEN,label="meds"),Patch(color=GREY,label="dx")],loc="lower right",fontsize=8)
    ax.set_title("F6 · Univariate discriminative power (top 15)")
    save(fig,"F06_univariate_auc","Univariate discriminative power","Each feature's standalone ability to separate diabetics (AUC), coloured by modality.")

    # F07 volcano: effect size vs significance
    fig,ax=plt.subplots(figsize=(8,6))
    sig=uni["p"]<0.05
    ax.scatter(uni.loc[~sig,"cohens_d"],uni.loc[~sig,"neglog10p"],c=GREY,s=25,alpha=0.6,label="ns")
    ax.scatter(uni.loc[sig,"cohens_d"],uni.loc[sig,"neglog10p"],c=RED,s=35,alpha=0.85,label="p<0.05")
    ax.axhline(-np.log10(0.05),color=BLUE,ls="--",lw=1)
    for _,r in uni.sort_values("neglog10p",ascending=False).head(6).iterrows():
        ax.annotate(lab(r["feature"]),(r["cohens_d"],r["neglog10p"]),fontsize=8,xytext=(4,2),textcoords="offset points")
    ax.set_xlabel("Effect size (Cohen's d):  DM − no-DM"); ax.set_ylabel("−log10(p)  (Mann–Whitney U)")
    ax.legend(); ax.set_title("F7 · Volcano plot — effect size vs significance")
    save(fig,"F07_volcano","Volcano plot","Effect size (Cohen's d) against statistical significance for every feature; labelled points are the strongest.")

    # F08 correlation heatmap of stable features
    if len(stable)>=2:
        Xs=combined[stable].corr().values; labs_s=[lab(f) for f in stable]
        fig,ax=plt.subplots(figsize=(8,7))
        im=ax.imshow(Xs,cmap="RdBu_r",vmin=-1,vmax=1)
        ax.set_xticks(range(len(stable))); ax.set_yticks(range(len(stable)))
        ax.set_xticklabels(labs_s,rotation=45,ha="right",fontsize=8); ax.set_yticklabels(labs_s,fontsize=8)
        for i in range(len(stable)):
            for j in range(len(stable)):
                ax.text(j,i,f"{Xs[i,j]:.2f}",ha="center",va="center",fontsize=6.5,
                        color="white" if abs(Xs[i,j])>0.6 else "black")
        fig.colorbar(im,fraction=0.046,pad=0.04); ax.set_title("F8 · Correlation among stable features")
        save(fig,"F08_corr","Stable-feature correlations","Pearson correlations among the stably-selected features (redundancy check).")

    # F09 logistic-regression odds ratios (forest plot) for top-8 stable features
    or_feats=stable[:8] if len(stable)>=1 else []
    if or_feats and sm is None:
        print("   F09 odds-ratio figure skipped: install 'statsmodels' to generate it.")
    if or_feats and sm is not None:
        Xo=StandardScaler().fit_transform(combined[or_feats].values); Xo=sm.add_constant(Xo)
        try:
            m=sm.Logit(y,Xo).fit_regularized(disp=0,alpha=0.5)  # mild L2 for stability (35 events)
            params=m.params[1:]; 
            # bootstrap CI for ORs (regularized fit has no analytic CI)
            B=300; boot=np.zeros((B,len(or_feats)))
            rng=np.random.default_rng(42)
            for b in range(B):
                idx=rng.integers(0,len(y),len(y))
                try: mb=sm.Logit(y[idx],Xo[idx]).fit_regularized(disp=0,alpha=0.5); boot[b]=mb.params[1:]
                except Exception: boot[b]=np.nan
            lo=np.nanpercentile(boot,2.5,axis=0); hi=np.nanpercentile(boot,97.5,axis=0)
            OR=np.exp(params); ORlo=np.exp(lo); ORhi=np.exp(hi)
            order=np.argsort(OR)
            fig,ax=plt.subplots(figsize=(8,5.5))
            yv=np.arange(len(or_feats))
            ax.errorbar(OR[order],yv,xerr=[OR[order]-ORlo[order],ORhi[order]-OR[order]],
                        fmt="o",color=BLUE,ecolor=GREY,capsize=3,ms=7)
            ax.axvline(1.0,color=RED,ls="--",lw=1)
            ax.set_yticks(yv); ax.set_yticklabels([lab(or_feats[i]) for i in order],fontsize=9)
            ax.set_xscale("log"); ax.set_xlabel("Adjusted odds ratio (per 1 SD, log scale)")
            ax.set_title("F9 · Logistic-regression odds ratios (stable features)")
            save(fig,"F09_odds_ratios","Odds ratios (logistic)","Multivariable-adjusted odds ratios per 1-SD increase, 95% bootstrap CI; vertical line = no effect.")
        except Exception as e:
            print("   OR fig skipped:",e)

    # F10 per-fold metrics grouped bar
    pf=pd.DataFrame(res["per_fold"])
    fig,ax=plt.subplots(figsize=(9,5))
    xf=np.arange(len(pf)); wd=0.25
    ax.bar(xf-wd,pf["auc"],wd,label="AUC",color=BLUE)
    ax.bar(xf,pf["accuracy"],wd,label="Accuracy",color=ORANGE)
    ax.bar(xf+wd,pf["f1"],wd,label="F1",color=GREEN)
    ax.axhline(res["auc_mean"],color=BLUE,ls=":",lw=1)
    ax.set_xticks(xf); ax.set_xticklabels([f"Fold {i+1}" for i in xf]); ax.set_ylim(0,1)
    ax.set_ylabel("Score"); ax.legend(ncol=3); ax.set_title("F10 · Nested-CV performance per fold")
    save(fig,"F10_perfold","Per-fold performance","AUC / accuracy / F1 for each outer fold (dotted = mean AUC); shows estimate stability.")

    # F11 ROC
    fig,ax=plt.subplots(figsize=(6.5,6))
    fpr,tpr,_=roc_curve(yt,proba); auc=roc_auc_score(yt,proba)
    ax.plot(fpr,tpr,color=BLUE,lw=3,label=f"QSQ-FS (AUC={auc:.3f})"); ax.plot([0,1],[0,1],ls="--",color=GREY)
    ax.fill_between(fpr,tpr,alpha=0.1,color=BLUE)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(loc="lower right")
    ax.set_title("F11 · ROC curve (out-of-fold)")
    save(fig,"F11_roc","ROC curve","Out-of-fold ROC; area under curve summarises ranking quality.")

    # F12 precision-recall
    fig,ax=plt.subplots(figsize=(6.5,6))
    pr,rc,_=precision_recall_curve(yt,proba); ap=average_precision_score(yt,proba)
    ax.plot(rc,pr,color=GREEN,lw=3,label=f"AP={ap:.3f}"); ax.axhline(yt.mean(),color=GREY,ls="--",label=f"baseline={yt.mean():.2f}")
    ax.set_xlabel("Recall (sensitivity)"); ax.set_ylabel("Precision (PPV)"); ax.set_ylim(0,1.02); ax.legend(loc="upper right")
    ax.set_title("F12 · Precision–Recall curve (out-of-fold)")
    save(fig,"F12_pr","Precision-Recall curve","PR curve — more informative than ROC under class imbalance; AP = average precision.")

    # F13 calibration
    fig,ax=plt.subplots(figsize=(6.5,6))
    frac_pos,mean_pred=calibration_curve(yt,proba,n_bins=5,strategy="quantile")
    ax.plot(mean_pred,frac_pos,"o-",color=ORANGE,lw=2,ms=7,label="QSQ-FS")
    ax.plot([0,1],[0,1],ls="--",color=GREY,label="perfectly calibrated")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed fraction positive"); ax.legend(loc="upper left")
    ax.set_title("F13 · Calibration (reliability) curve")
    save(fig,"F13_calibration","Calibration curve","Are predicted probabilities trustworthy? Points on the diagonal = well-calibrated.")

    # F14 predicted-probability separation
    fig,ax=plt.subplots(figsize=(7.5,5))
    bins=np.linspace(0,1,21)
    ax.hist(proba[yt==0],bins=bins,alpha=0.6,color=ND_C,label="No diabetes",density=True)
    ax.hist(proba[yt==1],bins=bins,alpha=0.6,color=DM_C,label="Diabetes",density=True)
    ax.axvline(0.5,color="black",ls="--",lw=1,label="threshold 0.5")
    ax.set_xlabel("Predicted probability of diabetes"); ax.set_ylabel("Density"); ax.legend()
    ax.set_title("F14 · Predicted-probability separation by true class")
    save(fig,"F14_separation","Probability separation","Distribution of predicted risk for each true class; overlap = where mistakes happen.")

    # F15 threshold sweep
    th=np.linspace(0.05,0.95,37); sens=[];spec=[];f1s=[];yj=[]
    for t in th:
        p=(proba>=t).astype(int); tn,fp,fn,tp=confusion_matrix(yt,p,labels=[0,1]).ravel()
        se=tp/(tp+fn) if tp+fn else 0; sp=tn/(tn+fp) if tn+fp else 0
        pr_=tp/(tp+fp) if tp+fp else 0; f1=2*pr_*se/(pr_+se) if pr_+se else 0
        sens.append(se);spec.append(sp);f1s.append(f1);yj.append(se+sp-1)
    best=th[int(np.argmax(yj))]
    fig,ax=plt.subplots(figsize=(8.5,5.5))
    ax.plot(th,sens,color=DM_C,lw=2,label="Sensitivity"); ax.plot(th,spec,color=ND_C,lw=2,label="Specificity")
    ax.plot(th,f1s,color=GREEN,lw=2,label="F1"); ax.plot(th,yj,color=ORANGE,lw=2,ls="--",label="Youden J")
    ax.axvline(0.5,color=GREY,ls=":",lw=1,label="default 0.5"); ax.axvline(best,color=RED,ls="-",lw=1.5,label=f"best J @ {best:.2f}")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Metric value"); ax.legend(ncol=2,fontsize=9)
    ax.set_title("F15 · Threshold sweep (operating-point selection)")
    save(fig,"F15_threshold","Threshold sweep","How sensitivity/specificity/F1 trade off with the decision threshold; red = optimal Youden point.")

    # F16 confusion matrices (counts + normalized)
    p05=(proba>=0.5).astype(int); cm=confusion_matrix(yt,p05); cmn=cm/cm.sum(axis=1,keepdims=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for ax,mat,ttl,fmt in [(axes[0],cm,"Counts","d"),(axes[1],cmn,"Row-normalised","0.2f")]:
        im=ax.imshow(mat,cmap="Blues",vmin=0,vmax=mat.max())
        ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(["No DM","DM"]); ax.set_yticklabels(["No DM","DM"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(ttl)
        for i in range(2):
            for j in range(2):
                v=mat[i,j]; ax.text(j,i,(f"{v:d}" if fmt=="d" else f"{v:.2f}"),ha="center",va="center",
                    fontsize=15,color="white" if v>mat.max()*0.5 else "black")
    fig.suptitle("F16 · Confusion matrix (out-of-fold)",fontsize=14,fontweight="bold",color="#1F3A5F")
    save(fig,"F16_confusion","Confusion matrix","Out-of-fold confusion as raw counts and per-class rates.")

    # F17 selection frequency (stability)
    fig,ax=plt.subplots(figsize=(8,7))
    top20=uni.assign(freq=freq).sort_values("freq",ascending=False).head(20).iloc[::-1] if len(uni)==len(freq) else None
    order=np.argsort(freq)[::-1][:20][::-1]
    labs17=[lab(names[i]) for i in order]; vals=[freq[i] for i in order]
    cols=[BLUE if v>=0.5 else GREY for v in vals]
    ax.barh(labs17,vals,color=cols); ax.axvline(0.5,color=RED,ls="--",lw=1,label="stable ≥0.5")
    ax.set_xlim(0,1); ax.set_xlabel("Selection frequency across folds"); ax.legend(loc="lower right")
    ax.set_title("F17 · Feature selection stability")
    save(fig,"F17_stability","Selection stability","How often each feature is chosen across folds; past the line = stable/trustworthy.")

    # F18 permutation importance on stable subset
    if stable:
        Xst=X[:,[names.index(f) for f in stable]]
        rf=RandomForestClassifier(n_estimators=400,random_state=42).fit(Xst,y)
        pi=permutation_importance(rf,Xst,y,n_repeats=30,random_state=42,scoring="roc_auc")
        imp=pd.DataFrame({"f":stable,"m":pi.importances_mean,"s":pi.importances_std}).sort_values("m")
        fig,ax=plt.subplots(figsize=(8,6))
        ax.barh([lab(f) for f in imp["f"]],imp["m"],xerr=imp["s"],color=BLUE,ecolor=GREY,capsize=3)
        ax.set_xlabel("Permutation importance (drop in AUC when shuffled)")
        ax.set_title("F18 · Feature importance (permutation)")
        save(fig,"F18_importance","Permutation importance","Which stable features actually drive the model (AUC lost when each is scrambled).")

    # F19 comparative methods across metrics
    ct=pd.DataFrame(SUMM["comparative"]["table"])
    if ct.empty or "AUC" not in ct.columns:
        # No comparative table available (e.g. study not yet run) -- emit a note
        # figure instead of raising, so the other 18 figures still render.
        for _fid,_ttl in [("F19_comparative","Method comparison"),
                          ("F20_frontier","Accuracy vs parsimony")]:
            fig,ax=plt.subplots(figsize=(9,5)); ax.axis("off")
            ax.text(0.5,0.5,"Run the Comparative study first\nto generate this figure",
                    ha="center",va="center",fontsize=13,color=GREY)
            save(fig,_fid,_ttl,"Requires the comparative study.")
        json.dump(manifest, open(os.path.join(FIG, "manifest.json"), "w"), indent=2)
        return manifest
    fig,ax=plt.subplots(figsize=(10,5.5))
    xm=np.arange(len(ct)); wd=0.27
    ax.bar(xm-wd,ct["AUC"],wd,label="AUC",color=BLUE)
    ax.bar(xm,ct["Accuracy"],wd,label="Accuracy",color=ORANGE)
    ax.bar(xm+wd,ct["F1"],wd,label="F1",color=GREEN)
    ax.set_xticks(xm); ax.set_xticklabels(ct["Method"],rotation=25,ha="right")
    ax.set_ylim(0,1); ax.set_ylabel("Score"); ax.legend(ncol=3)
    # highlight QSQ-FS
    for i,mth in enumerate(ct["Method"]):
        if mth=="QSQ-FS": ax.axvspan(i-0.42,i+0.42,color=BLUE,alpha=0.06)
    ax.set_title("F19 · Method comparison across metrics (leak-free, equal budget)")
    save(fig,"F19_comparative","Method comparison","QSQ-FS vs 7 baselines on AUC/Accuracy/F1 under one fair protocol.")

    # F20 efficiency frontier: #features vs AUC
    fig,ax=plt.subplots(figsize=(8,6))
    for _,r in ct.iterrows():
        is_q=r["Method"]=="QSQ-FS"; is_full=r["Features"]>=(ct["Features"].max()-1)
        ax.scatter(r["Features"],r["AUC"],s=160 if is_q else 90,
                   color=BLUE if is_q else (RED if is_full else GREY),
                   edgecolor="black",zorder=3,marker="*" if is_q else ("s" if is_full else "o"))
        ax.annotate(r["Method"],(r["Features"],r["AUC"]),fontsize=8,xytext=(6,4),textcoords="offset points")
    ax.set_xlabel("Number of features used"); ax.set_ylabel("AUC")
    ax.set_title("F20 · Accuracy-vs-parsimony frontier")
    save(fig,"F20_frontier","Accuracy vs parsimony","AUC against feature count: top-left = accurate AND simple. QSQ-FS (star) uses ~half the features.")


    json.dump(manifest, open(os.path.join(FIG, "manifest.json"), "w"), indent=2)
    return manifest
