## 0.1.7 (2021-06-25)
* Avoid out-of-bounds memory read (issue [#165](https://github.com/atomashpolskiy/rustface/issues/165)) (thanks @mashedcode!)

## 0.1.6 (2020-11-02)
* Made model struct immutable and `Clone`-able so it can be re-used (issue [#12](https://github.com/atomashpolskiy/rustface/issues/12)) (thanks @kornelski!)
* Made `Rectangle` public (thanks @kornelski!).

## 0.1.5 (2020-10-28)
* Reduced use of `unsafe` (thanks @kornelski!).
* Performance improvements (thanks @kornelski!).

## 0.1.4 (2020-10-01)
* Made Rayon an optional dependency using a feature flag. It is enabled by default. Parallel processing can now be disabled by providing the `--no-default-features` flag at build time.

## 0.1.3 (2020-09-08)
* Edition 2018
* Updated dependencies
* Various code quality improvements, such as reduced unsafety (thanks @kornelski!)
