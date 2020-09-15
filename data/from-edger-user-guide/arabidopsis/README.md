# RNA-Seq of pathogen inoculated arabidopsis with batch effects

This dataset contains the data from EdgeR user guide (3.30.3) section 4.2.
The data originally comes from (Cumbie et al., 2011).

EdgeR authors have made it convenient for us and have created a 
[rds file](http://bioinf.wehi.edu.au/edgeR/UserGuideData/arab.rds) 
with the read counts for us.

Following the EdgeR guide, they load the file as described below:

```R
> arab <- readRDS("arab.rds")
> head(arab)
mock1 mock2 mock3 hrcc1 hrcc2 hrcc3
AT1G01010 35 77 40 46 64 60
AT1G01020 43 45 32 43 39 49
AT1G01030 16 24 26 27 35 20
AT1G01040 72 43 64 66 25 90
AT1G01050 49 78 90 67 45 60
AT1G01060 0 15 2 0 21 8
``` 

The factors are described and file is then converted to DGEList item:

```R
> Treat <- factor(substring(colnames(arab),1,4))
> Treat <- relevel(Treat, ref="mock")
> Time <- factor(substring(colnames(arab),5,5))
```

```R
y <- DGEList(counts=arab, group=Treat)
```

Before normalisation, the reads are further filtered:

```R
> keep <- filterByExpr(y)
> table(keep)
keep
FALSE TRUE
12292 13930
> y <- y[keep, , keep.lib.sizes=FALSE]
```

The CSV file [`arab.csv`](arab.csv) contains the `y$counts` 
after this filtering step (see [`toCsv.R`](toCSV.R)).

## Expected answers

The `edgeR` guide describes the following expected answers on this dataset:

```R
> y <- calcNormFactors(y)
> y$samples
group lib.size norm.factors
mock1 mock 1882391 0.977
mock2 mock 1870625 1.023
mock3 mock 3227243 0.914
hrcc1 hrcc 2101449 1.058
hrcc2 hrcc 1243266 1.083
hrcc3 hrcc 3494821 0.955
```

Reproducing the results, gives these answers (which are equivalent):

```R
> calcNormFactors(y)$samples
      group lib.size norm.factors
mock1  mock  1882391    0.9771684
mock2  mock  1870625    1.0228448
mock3  mock  3227243    0.9142136
hrcc1  hrcc  2101449    1.0584492
hrcc2  hrcc  1243266    1.0828426
hrcc3  hrcc  3494821    0.9548559
> calcNormFactors(as.data.frame(y$counts))
    mock1     mock2     mock3     hrcc1     hrcc2     hrcc3 
0.9771684 1.0228448 0.9142136 1.0584492 1.0828426 0.9548559
```

## References

* [EdgeR user guide](https://bioconductor.org/packages/release/bioc/html/edgeR.html)
* Cumbie, J., Kimbrel, J., Di, Y., Schafer, D., Wilhelm, L., Fox, S., Sullivan, C., Curzon, A., Carrington, J., Mockler, T., Chang, J. (2011). GENE-Counter: A Computational Pipeline for the Analysis of RNA-Seq Data for Gene Expression Differences PLoS ONE  6(10), e25279. https://dx.doi.org/10.1371/journal.pone.0025279
