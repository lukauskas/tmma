# tmma - MA plots and TMM for python

Based on the implementation in [`EdgeR` Bioconductor package](https://bioconductor.org/packages/release/bioc/html/edgeR.html).

## Installation (development)

Clone the repository:

```
git clone https://github.com/lukauskas/tmma.git
```

Install package with (`-e`) flag set and optional `test` dependencies.

```
pip install -e .[test]
```

## References

* Robinson, M., McCarthy, D., Smyth, G. (2009). edgeR: a Bioconductor package for differential expression analysis of digital gene expression data Bioinformatics  26(1), 139-140. https://dx.doi.org/10.1093/bioinformatics/btp616
* Robinson, M., Oshlack, A. (2010). A scaling normalization method for differential expression analysis of RNA-seq data Genome Biology  11(3), R25. https://dx.doi.org/10.1186/gb-2010-11-3-r25
* McCarthy, D., Chen, Y., Smyth, G. (2012). Differential expression analysis of multifactor RNA-Seq experiments with respect to biological variation Nucleic Acids Research  40(10), 4288-4297. https://dx.doi.org/10.1093/nar/gks042