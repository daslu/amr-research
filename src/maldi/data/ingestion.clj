(ns maldi.data.ingestion
  (:require [tablecloth.api :as tc]
            [babashka.fs :as fs]
            [clojure.string :as str]
            [maldi.errors :as errors]))

(defn base-dir-from-env []
  (System/getenv "DRIAMS_BASE_DIR"))

(defn find-data-files
  "Find all files with given extension in data directory"
  [extension & {:keys [base-dir]
                :or {base-dir (base-dir-from-env)}}]
  (->> (fs/glob base-dir (str "**/*." extension))
       (concat (fs/glob base-dir (str "*." extension))) ; Also check base directory
       (map str)
       (filter #(fs/regular-file? %))
       distinct
       sort))

(comment
  (find-data-files "txt.gz" {}))

(defn parse-raw-file-path
  "Extract metadata from DRIAMS file path"
  [path {:keys [base-dir]
         :or {base-dir (base-dir-from-env)}}]
  (-> path
      (str/split #"\.")
      first
      (str/replace base-dir "")
      (str/replace #"DRIAMS-" "")
      (str/replace #"raw/" "")
      (str/split #"/")))

(comment
  (-> (find-data-files "txt.gz" {})
      first
      (parse-raw-file-path {})))


(def raw-files-dataset
  (memoize
   (fn [{:keys [base-dir]
         :or {base-dir (base-dir-from-env)}}]
     (-> (find-data-files "txt.gz" {:base-dir base-dir})
         (->> (map (fn [path]
                     (conj (parse-raw-file-path path {:base-dir base-dir})
                           path))))
         tc/dataset
         (tc/rename-columns [:site :year :code :path])
         (tc/map-columns :year :year #(Integer/parseInt %))
         (tc/map-columns :site :site keyword)))))

(comment
  (raw-files-dataset {}))


(defn load-raw-spectrum
  "Load raw spectrum data from file"
  [path]
  (errors/with-error-handling
    errors/data-error
    {:path path :operation :load-raw}
    (-> path
        (tc/dataset {:separator " "})
        (tc/rename-columns [:mass :intensity]))))

(comment
  (-> (raw-files-dataset {})
      :path
      first
      load-raw-spectrum))

(defn available-cases
  "Get all available cases from data directory"
  [{:keys [base-dir]
    :or {base-dir (base-dir-from-env)}}]
  (-> base-dir
      raw-files-dataset
      (tc/unique-by [:site :year :code])))

(comment
  (available-cases {}))

(defn load-metadata
  "Load metadata/labels for cases"
  [{:keys [base-dir site year]
    :or {base-dir (base-dir-from-env)}}]
  (let [path (format "%sDRIAMS-%s/id/%d/%d_clean.csv.gz"
                     base-dir
                     (name site)
                     year
                     year)]
    (when (fs/exists? path)
      (tc/dataset path {:key-fn keyword}))))

(comment
  (load-metadata {:site :A
                  :year 2018}))

(defn example-path []
  (-> (raw-files-dataset {})
      (tc/select-rows #(and (= (:year %) 2018)
                            (= (:site %) :A)))
      :path
      second))


(comment
  (load-raw-spectrum (example-path)))
