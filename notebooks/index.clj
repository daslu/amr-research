;; # Preface
;;

^{:kindly/hide-code true}
(ns index
  (:require
   ;; Clojure string utilities:
   [clojure.string :as str]
   ;; Annotating kinds of visualizations (https://scicloj.github.io/kindly-noted/):
   [scicloj.kindly.v4.kind :as kind]))

^{:kindly/hide-code true
  :kind/md true}
(->> "README.md"
     slurp
     str/split-lines
     (drop 1)
     (str/join "\n"))

;; ## Chapters in this book
