---
license: cc-by-sa-4.0
language:
- en
tags:
- urban
- perception
- computervision
size_categories:
- n<1K
---
# Street Perception Evaluation Considering Socioeconomics (SPECS)
Repository for the Street Perception Evaluation Considering Socioeconomics (SPECS) dataset developed in the [It's not you, it's me: Global urban visual perception varies across demographics and personalities](https://github.com/matqr/specs) project.

This dataset is also used in related work titled [It is not always greener on the other side: Greenery perception across demographics and personalities in multiple cities](https://doi.org/10.1016/j.landurbplan.2026.105618), [codebase](https://github.com/matqr/greenery-perception) and [additional dataset](https://huggingface.co/datasets/matiasqr/greenery-perception) is also available.

These projects were developed at the Future Cities Lab Global in the Singapore-ETH Centre in close collaboration with the [Urban Analytics Lab (UAL)](https://ual.sg/) at the National University of Singapore (NUS).


# Content Breakdown
```
SPECS
├── abuja/
    ├── 10 CSV files with contextual information for this city' images
├── global-streetscapes/ (download contextual data from original [Global Streetscapes](https://huggingface.co/datasets/NUS-UAL/global-streetscapes))
├── labels/
    ├── final/ (6 XLSX files with final pairwise comparisons)
    ├── inferences/ (6 CSV files with inferred perception scores using the perception model used in [Global Streetscapes](https://huggingface.co/datasets/NUS-UAL/global-streetscapes))
    ├── processed/ (10 CSV files with computed perception Q scores)
├── svi/ (SVIs should be downloaded following the [wiki](https://github.com/matqr/specs/wiki) or with the metadata file)
    ├── img_paths.csv (path location )
    ├── metadata.csv (file with imagery metadata)
    ├── visual_complexity_all.csv
```

# Read More
Read more about this project on its [website](), which includes an overview of this effort together with the background and the [paper](https://www.nature.com/articles/s44284-025-00330-x).

A free version (postprint / author-accepted manuscript) can be downloaded [here](https://arxiv.org/abs/2505.12758).

# Citation
To cite this work, please refer to the [paper](https://www.nature.com/articles/s44284-025-00330-x):

Quintana M, Gu Y, Liang X, Hou Y, Ito K, Zhu Y, Abdelrahman M, Biljecki F (2025): Global urban visual perception varies across demographics and personalities. Nature Cities 2(11): 1092-1106. doi: 10.1038/s44284-025-00330-x

BibTeX:
```
@article{quintana2025,
  title = {Global Urban Visual Perception Varies across Demographics and Personalities},
  author = {Quintana, Matias and Gu, Youlong and Liang, Xiucheng and Hou, Yujun and Ito, Koichi and Zhu, Yihan and Abdelrahman, Mahmoud and Biljecki, Filip},
  year = {2025},
  journal = {Nature Cities},
  volume = {2},
  number = {11},
  pages = {1092--1106},
  issn = {2731-9997},
  doi = {10.1038/s44284-025-00330-x},
  url = {https://www.nature.com/articles/s44284-025-00330-x},
}
```

# Related projects

Quintana, M., Liu, F., Torkko, J., Gu, Y., Liang, X., Hou, Y., Ito, K., Zhu, Y., Abdelrahman, M., Toivonen, T., Lu, Y., & Biljecki, F. (2026). It is not always greener on the other side: Greenery perception across demographics and personalities in multiple cities. Landscape and Urban Planning, 271, 105618. https://doi.org/10.1016/j.landurbplan.2026.105618

BibTeX:
```
@article{quintana2026,
  title = {It Is Not Always Greener on the Other Side: Greenery Perception across Demographics and Personalities in Multiple Cities},
  author = {Quintana, Matias and Liu, Fangqi and Torkko, Jussi and Gu, Youlong and Liang, Xiucheng and Hou, Yujun and Ito, Koichi and Zhu, Yihan and Abdelrahman, Mahmoud and Toivonen, Tuuli and Lu, Yi and Biljecki, Filip},
  year = {2026},
  journal= {Landscape and Urban Planning},
  volume = {271},
  pages = {105618},
  issn = {0169-2046},
  doi = {10.1016/j.landurbplan.2026.105618},
  url = {https://www.sciencedirect.com/science/article/pii/S0169204626000423},
}
```