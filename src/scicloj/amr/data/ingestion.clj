(ns scicloj.amr.data.ingestion
  (:require [tablecloth.api :as tc]
            [babashka.fs :as fs]
            [clojure.string :as str]))

(def ^:dynamic *base-dir*
  (System/getenv "DRIAMS_BASE_DIR"))

(defn find-data-files
  "Find all files with given extension in data directory"
  [extension]
  (->> (fs/glob *base-dir* (str "**/*." extension))
       (concat (fs/glob *base-dir* (str "*." extension))) ; Also check base directory
       (map str)
       (filter #(fs/regular-file? %))
       distinct
       sort))

(comment
  (find-data-files "txt.gz"))

(defn parse-raw-file-path
  "Extract metadata from DRIAMS file path"
  [path]
  (-> path
      (str/split #"\.")
      first
      (str/replace *base-dir* "")
      (str/replace #"DRIAMS-" "")
      (str/replace #"raw/" "")
      (str/split #"/")))

(comment
  (-> (find-data-files "txt.gz")
      first
      parse-raw-file-path))


(def raw-files-dataset
  (memoize
   (fn []
     (-> (find-data-files "txt.gz")
         (->> (map (fn [path]
                     (conj (parse-raw-file-path path)
                           path))))
         tc/dataset
         (tc/rename-columns [:site :year :code :path])
         (tc/map-columns :year :year #(Integer/parseInt %))
         (tc/map-columns :site :site keyword)))))

(comment
  (raw-files-dataset))


(defn load-raw-spectrum
  "Load raw spectrum data from file"
  [path]
  (-> path
      (tc/dataset {:separator " "})
      (tc/rename-columns [:mass :intensity])))

(comment
  (-> (raw-files-dataset)
      :path
      first
      load-raw-spectrum))

(defn available-cases
  "Get all available cases from data directory"
  []
  (-> (raw-files-dataset)
      (tc/unique-by [:site :year :code])))

(comment
  (available-cases))

(defn available-site-years
  []
  (-> (raw-files-dataset)
      (tc/select-columns [:site :year])
      (tc/unique-by [:site :year])))

(comment
  (available-site-years))

(defn load-metadata
  "Load metadata/labels for cases"
  [{:keys [year site]}]
  (let [path (format "%sDRIAMS-%s/id/%d/%d_clean.csv.gz"
                     *base-dir*
                     (name site)
                     year
                     year)]
    (when (fs/exists? path)
      (tc/dataset path {:key-fn keyword}))))

(comment
  (load-metadata {:site :A
                  :year 2018}))

(defn example-path []
  (-> (raw-files-dataset)
      (tc/select-rows #(and (= (:year %) 2018)
                            (= (:site %) :A)))
      :path
      second))

(comment
  (load-raw-spectrum (example-path)))


(def all-antibiotics
  (memoize
   (fn []
     (->> (tc/rows (available-site-years) :as-maps)
          (mapcat (fn [acase]
                    (-> acase
                        load-metadata
                        (tc/drop-columns [:code :species :laboratory_species
                                          :$error :$value
                                          :column-0 :combined_code
                                          (keyword "Unnamed: 0")])
                        keys)))
          distinct
          sort))))


(comment
  (all-antibiotics))


(def all-bacteria
  (memoize
   (fn []
     (->> (tc/rows (available-site-years) :as-maps)
          (mapcat (fn [acase]
                    (-> acase
                        load-metadata
                        :species)))
          distinct
          sort))))


(comment
  (all-bacteria))
