# rihcharts

Copyright &copy; 2023 Darren Erik Vengroff

[![Hippocratic License HL3-CL-ECO-EXTR-FFD-LAW-MIL-SV](https://img.shields.io/static/v1?label=Hippocratic%20License&message=HL3-CL-ECO-EXTR-FFD-LAW-MIL-SV&labelColor=5e2751&color=bc8c3d)](https://firstdonoharm.dev/version/3/0/cl-eco-extr-ffd-law-mil-sv.html)

This project generates impact charts from the dataset
constructed by the `rihdata` project. It does so using
the `impactcharts` project, which it turn relies on
`SHAP`.

## Instructions

Before running the analysis is this project, it is
necessary to download the data using the `rihdata`
project. Normally, `rihdata` and `rihcharts` are
installed locally under the same parent directory.
If they are not, a small local modification may be 
needed for things to work.

All of the analysis this project does is
orchestrated by a single `Makefile`. It requires
[GNU make](https://www.gnu.org/software/make/) 
version 4.3 or higher.

Before you run GNU make, you will have to set up a
Python virtual environment with all the project's
dependencies. I use [poetry](https://python-poetry.org/) 
to do this. All top-level dependencies are listed in 
`pyproject.toml` and the full set of recursive 
dependencies and version is stored in `poetry.lock`.

Once you have your virtual environment and the right
version of GNU Make installed, the single command

```shell
gmake [-j 8]
```

should build the entire project. The `-j` argument
is optional. If given, it tells make that it can
parallelize the operation eight ways across multiple
CPU cores. If you have multiple cores, you can choose
an appropriate `-j` number to use them all.

The steps the Makefile will execute are:

1. Fit and hyperparameter tune an XGBoost model 
   for each CBSA that predicts median housing
   prices based on median income and racial demographics.
2. Fit a series of linear models for the same problem.
3. Generate SHAP values and plots for a large number 
   of individual models and their ensemble for each CBSA and demographic feature.

How long this will take depends on how fast your 
machine is, how many cores you have and choose to use,
your internet connection are. For me, running on 8 
eight cores on an 
M1 MacBook Pro from a clean  state it takes less than 
an hour to complete.

For more details and options, please consult the 
`Makefile`.

The results will be a series of plots in a newly
created directory called `plots-2021/shap/xgb`. 
There is a subdirectory for each CBSA, for example
for Atlanta it is 
`Atlanta-Sandy_Springs-Alpharetta,_GA_Metro_Area/12060`.
The number `12060` is the FIPS code for the CBSA. 
It will be different for each one.

Within this directory are the impact charts. Each
chart
shows the impact of the fraction of the population
belonging to one racial or ethnic group on median 
home prices. This is the impact of that variable
alone, corrected for the impact of median household
income. For example the impact on the fraction of
Black residents looks like this 

![atl-nh-black.png](docs%2Fimages%2Fatl-nh-black.png)

The basic idea of this chart is that neighborhoods
that have almost no Black residents at all have 
higher median home prices than those that have large
numbers of Black residents. We can see from the shape
of the curve just how much Blackness in a neighborhood
creates what level of effect.

Median household income's impact is in a 
final chart in the subdirectory. For Atlanta, it 
looks like this

![atl-median-income.png](docs%2Fimages%2Fatl-median-income.png)

In this case we see that higher income areas tend to
have higher median home prices, and that except for 
very low income areas this affect looks almost linear.

It is important to realize that impact charts are 
built using Machine Learning models along with an
approach called SHAP to specifically isolate the 
effects of one variable from another. 

Taken together, these two charts show us that whether
there is any correlation between Blackness and income
or not. 

This project produces hundreds of charts like this, and they
are all a little different, and show different effects
in different areas. 

Like in `rihdata`, we can use choose to do the analysis 
for the top `N` for `N` other than the default of 50.
For example, for 10, we could run

```shell
gmake -j 8 N=10
```

Before running this, we must be sure that we explicity
ran 

```shell
gmake N=10
```

in the `rihdata` project first to ensure we have all
the data and metadata we need.
