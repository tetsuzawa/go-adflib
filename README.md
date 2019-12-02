go-adflib
====

[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)
![test](https://github.com/tetsuzawa/go-adflib/workflows/test/badge.svg)
[![GoDoc](https://godoc.org/github.com/tetsuzawa/go-adflib?status.svg)](https://godoc.org/github.com/tetsuzawa/go-adflib)

go-adflib was created to implement adaptive signal processing task with golang. This library uses gonum/floats and gonum/mat for matrix operations. 

This library is created with reference to [padasip](https://github.com/matousc89/padasip) (writen in python).

## Demo
By using an adaptive filter, you can predict the value of the next sample.

![demo](https://raw.github.com/wiki/tetsuzawa/go-adflib/img/run_example.gif)

## Usage

See [godoc](https://godoc.org/github.com/tetsuzawa/go-adflib)

## Install

```shell
go get github.com/tetsuzawa/go-adflib/adf
```

## Author
See here [github](https://github.com/tetsuzawa)
