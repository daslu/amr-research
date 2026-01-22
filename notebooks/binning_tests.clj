(ns binning-tests
  "Tests for binning implementation.
   
   Since MALDIquant doesn't have an equivalent binning function
   (their binPeaks is for aligning detected peaks across samples),
   we create synthetic test cases to verify correctness against
   the DRIAMS paper specification."
  (:require [maldi.data.binning :as binning]
            [tablecloth.api :as tc]
            [scicloj.kindly.v4.kind :as kind]))

;; ## Binning Tests

;; The Weis et al. DRIAMS paper specifies:
;; "partition the m/z axis in the range of 2,000 to 20,000 Da into disjoint, 
;; equal-sized bins and sum the intensity of all measurements in the sample 
;; falling into the same bin"

;; ### Test Case 1: Simple case - single point per bin

(def test-spectrum-1
  (tc/dataset {:mass [2001.5 2004.5 2007.5]
               :intensity [100.0 200.0 300.0]}))

test-spectrum-1

(def binning-params-1
  {:range [2000 2010]
   :step 3})

binning-params-1

;; Expected bins:
;; bin 0: [2000, 2003) -> mass 2001.5, intensity 100.0
;; bin 1: [2003, 2006) -> mass 2004.5, intensity 200.0  
;; bin 2: [2006, 2009) -> mass 2007.5, intensity 300.0
;; bin 3: [2009, 2012) -> empty
;; Total bins: (2010-2000)/3 + 1 = 4

(def binned-1 (binning/bin-spectrum test-spectrum-1 binning-params-1))

binned-1

(kind/table
 {:column-names ["Bin" "Expected" "Actual" "Match?"]
  :row-vectors [[0 100.0 (aget binned-1 0) (= 100.0 (aget binned-1 0))]
                [1 200.0 (aget binned-1 1) (= 200.0 (aget binned-1 1))]
                [2 300.0 (aget binned-1 2) (= 300.0 (aget binned-1 2))]
                [3 0.0 (aget binned-1 3) (= 0.0 (aget binned-1 3))]]})

(= [100.0 200.0 300.0 0.0] (vec binned-1))
(kind/test-last [true?])

;; ### Test Case 2: Multiple points in same bin (should sum)

(def test-spectrum-2
  (tc/dataset {:mass [2001.0 2002.0 2004.0 2005.0]
               :intensity [100.0 50.0 200.0 75.0]}))

test-spectrum-2

;; Expected:
;; bin 0: [2000, 2003) -> masses 2001.0, 2002.0 -> sum = 150.0
;; bin 1: [2003, 2006) -> masses 2004.0, 2005.0 -> sum = 275.0
;; bin 2: [2006, 2009) -> empty -> 0.0
;; bin 3: [2009, 2012) -> empty -> 0.0

(def binned-2 (binning/bin-spectrum test-spectrum-2 binning-params-1))

binned-2

(kind/table
 {:column-names ["Bin" "Expected" "Actual" "Match?"]
  :row-vectors [[0 150.0 (aget binned-2 0) (= 150.0 (aget binned-2 0))]
                [1 275.0 (aget binned-2 1) (= 275.0 (aget binned-2 1))]
                [2 0.0 (aget binned-2 2) (= 0.0 (aget binned-2 2))]
                [3 0.0 (aget binned-2 3) (= 0.0 (aget binned-2 3))]]})

(= [150.0 275.0 0.0 0.0] (vec binned-2))
(kind/test-last [true?])

;; ### Test Case 3: Out of range values (should be filtered)

(def test-spectrum-3
  (tc/dataset {:mass [1999.0 2001.0 2005.0 2011.0]
               :intensity [999.0 100.0 200.0 888.0]}))

test-spectrum-3

;; Expected:
;; 1999.0 -> out of range (< 2000), filtered
;; 2001.0 -> bin 0, intensity 100.0
;; 2005.0 -> bin 1, intensity 200.0
;; 2011.0 -> out of range (> 2010), filtered

(def binned-3 (binning/bin-spectrum test-spectrum-3 binning-params-1))

binned-3

(kind/table
 {:column-names ["Bin" "Expected" "Actual" "Match?"]
  :row-vectors [[0 100.0 (aget binned-3 0) (= 100.0 (aget binned-3 0))]
                [1 200.0 (aget binned-3 1) (= 200.0 (aget binned-3 1))]
                [2 0.0 (aget binned-3 2) (= 0.0 (aget binned-3 2))]
                [3 0.0 (aget binned-3 3) (= 0.0 (aget binned-3 3))]]})

(= [100.0 200.0 0.0 0.0] (vec binned-3))
(kind/test-last [true?])

;; ### Test Case 4: DRIAMS parameters (2000-20000, step=3)

(def driams-params
  {:range [2000 20000]
   :step 3})

driams-params

;; Number of bins should be (20000-2000)/3 + 1 = 6001
(def n-bins-driams (binning/calculate-n-bins driams-params))

n-bins-driams

(kind/md (format "**DRIAMS binning creates %d bins**" n-bins-driams))

(= 6001 n-bins-driams)
(kind/test-last [true?])

;; Test with synthetic data
(def test-spectrum-driams
  (tc/dataset {:mass [2000.0 2003.0 2006.0 10000.0 19999.0]
               :intensity [10.0 20.0 30.0 500.0 999.0]}))

test-spectrum-driams

(def binned-driams (binning/bin-spectrum test-spectrum-driams driams-params))

binned-driams

;; Verify array length
(= 6001 (alength binned-driams))
(kind/test-last [true?])

;; Verify specific bins
;; 2000.0 -> bin 0
;; 2003.0 -> bin 1
;; 2006.0 -> bin 2
;; 10000.0 -> bin (10000-2000)/3 = 2666.67 floored = 2666
;; 19999.0 -> bin (19999-2000)/3 = 5999.67 floored = 5999

(kind/table
 {:column-names ["Mass" "Expected Bin" "Intensity" "Actual Value" "Match?"]
  :row-vectors [[2000.0 0 10.0 (aget binned-driams 0) (= 10.0 (aget binned-driams 0))]
                [2003.0 1 20.0 (aget binned-driams 1) (= 20.0 (aget binned-driams 1))]
                [2006.0 2 30.0 (aget binned-driams 2) (= 30.0 (aget binned-driams 2))]
                [10000.0 2666 500.0 (aget binned-driams 2666) (= 500.0 (aget binned-driams 2666))]
                [19999.0 5999 999.0 (aget binned-driams 5999) (= 999.0 (aget binned-driams 5999))]]})

(and (= 10.0 (aget binned-driams 0))
     (= 20.0 (aget binned-driams 1))
     (= 30.0 (aget binned-driams 2))
     (= 500.0 (aget binned-driams 2666))
     (= 999.0 (aget binned-driams 5999)))
(kind/test-last [true?])

;; ### Test Case 5: Empty spectrum

(def test-spectrum-empty
  (tc/dataset {:mass []
               :intensity []}))

test-spectrum-empty

(def binned-empty (binning/bin-spectrum test-spectrum-empty binning-params-1))

binned-empty

;; Should create array of zeros
(= [0.0 0.0 0.0 0.0] (vec binned-empty))
(kind/test-last [true?])
