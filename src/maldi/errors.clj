(ns maldi.errors
  (:require [clojure.tools.logging :as log]))

(defn error-context
  "Create an error with context information"
  [message context & {:keys [cause]}]
  (ex-info message
           (merge {:type :maldi/error
                   :timestamp (java.time.Instant/now)}
                  context)
           cause))

(defn data-error
  "Create a data processing error"
  [message data & {:keys [cause]}]
  (error-context message
                 {:error-type :data-processing
                  :data data}
                 :cause cause))

(defn model-error
  "Create a model training/prediction error"
  [message model-info & {:keys [cause]}]
  (error-context message
                 {:error-type :model
                  :model model-info}
                 :cause cause))

(defn validation-error
  "Create a validation error"
  [message validation-data & {:keys [cause]}]
  (error-context message
                 {:error-type :validation
                  :validation validation-data}
                 :cause cause))

(defn pipeline-error
  "Create a pipeline execution error"
  [message stage data & {:keys [cause]}]
  (error-context message
                 {:error-type :pipeline
                  :stage stage
                  :data data}
                 :cause cause))

(defmacro with-error-handling
  "Execute body with error handling and logging"
  [error-fn context & body]
  `(try
     ~@body
     (catch Exception e#
       (let [error# (~error-fn
                     (str "Error: " (.getMessage e#))
                     ~context
                     :cause e#)]
         (log/error error# "Processing failed")
         (throw error#)))))

(defmacro safe-execute
  "Execute body and return nil on error with logging"
  [context & body]
  `(try
     ~@body
     (catch Exception e#
       (log/warn e# "Safe execution failed" ~context)
       nil)))

(defn handle-error
  "Generic error handler with logging"
  [error context]
  (log/error error "Error occurred" context)
  (throw (error-context "Processing failed" context :cause error)))