(ns scicloj.amr.data.bacteria)

(def E-coli "Escherichia coli")
(def S-aureus "Staphylococcus aureus")
(def S-epidermidis "Staphylococcus epidermidis")
(def P-aeruginosa "Pseudomonas aeruginosa")
(def K-pneumoniae "Klebsiella pneumoniae")

(def important-bacteria
  [E-coli
   S-aureus
   P-aeruginosa
   K-pneumoniae])

(def species->antibiotics
  "The 23 species–antibiotic combinations from the Weis et al. paper."
  {E-coli
   [:Meropenem :Ertapenem :Ceftriaxone :Cefepime
    :Piperacillin-Tazobactam :Nitrofurantoin
    :Ciprofloxacin :Cotrimoxazole]

   S-aureus
   [:Cotrimoxazole :Clindamycin :Vancomycin :Linezolid
    (keyword "Amoxicillin-Clavulanic acid")
    (keyword "Ampicillin-Amoxicillin")
    :Oxacillin]

   P-aeruginosa
   [:Piperacillin-Tazobactam :Cefepime :Ceftazidime
    :Meropenem :Amikacin :Ciprofloxacin
    :Colistin :Tobramycin]})
