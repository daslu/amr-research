(ns scicloj.amr.data.bacteria
  "Bacterial species and antibiotic definitions for AMR prediction.

  The species list includes the three from the Weis et al. paper
  (E. coli, S. aureus, K. pneumoniae) plus five additional species
  found by scanning DRIAMS site A (2018) metadata for combinations
  with ≥100 labeled samples and both R ≥ 10 and S ≥ 10.

  Antibiotic lists per species were selected using the same
  data-availability criterion — not clinical guidelines.")

(def E-coli "Escherichia coli")
(def S-aureus "Staphylococcus aureus")
(def S-epidermidis "Staphylococcus epidermidis")
(def P-aeruginosa "Pseudomonas aeruginosa")
(def K-pneumoniae "Klebsiella pneumoniae")
(def E-cloacae "Enterobacter cloacae")
(def E-faecium "Enterococcus faecium")
(def P-mirabilis "Proteus mirabilis")

(def important-bacteria
  [E-coli
   S-aureus
   K-pneumoniae
   P-aeruginosa
   S-epidermidis
   E-cloacae
   E-faecium
   P-mirabilis])

(def species->antibiotics
  "Species–antibiotic combinations for AMR prediction.
  Includes the Weis et al. paper combinations plus additional
  clinically relevant species with sufficient data in DRIAMS
  (≥100 samples with both R and S labels at site A, 2018)."
  {E-coli
   [:Ceftriaxone :Cefepime :Ciprofloxacin :Cotrimoxazole
    :Piperacillin-Tazobactam :Nitrofurantoin
    :Meropenem :Ertapenem]

   S-aureus
   [:Oxacillin :Ciprofloxacin :Cotrimoxazole :Clindamycin
    :Erythromycin :Penicillin :Tetracycline
    (keyword "Fusidic acid") :Rifampicin :Gentamicin
    (keyword "Amoxicillin-Clavulanic acid")
    (keyword "Ampicillin-Amoxicillin")
    :Vancomycin :Linezolid]

   K-pneumoniae
   [:Ceftriaxone :Cefepime :Ciprofloxacin :Cotrimoxazole
    (keyword "Amoxicillin-Clavulanic acid")
    :Ceftazidime :Tobramycin :Piperacillin-Tazobactam]

   P-aeruginosa
   [:Piperacillin-Tazobactam :Cefepime :Ceftazidime
    :Meropenem :Amikacin :Ciprofloxacin
    :Colistin :Tobramycin]

   S-epidermidis
   [:Oxacillin :Ciprofloxacin :Cotrimoxazole :Gentamicin
    :Clindamycin :Erythromycin :Tetracycline
    (keyword "Fusidic acid") :Rifampicin]

   E-cloacae
   [:Ceftriaxone :Ceftazidime :Piperacillin-Tazobactam
    :Cotrimoxazole :Ertapenem :Ciprofloxacin
    :Tobramycin :Cefepime]

   E-faecium
   [:Vancomycin]

   P-mirabilis
   [:Cotrimoxazole (keyword "Ampicillin-Amoxicillin")
    :Ciprofloxacin (keyword "Amoxicillin-Clavulanic acid")
    :Tobramycin]})

