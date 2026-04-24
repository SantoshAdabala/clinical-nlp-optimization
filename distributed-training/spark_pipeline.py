import argparse
import json
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType, ArrayType
)
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    HashingTF,
    IDF,
    NGram,
    CountVectorizer,
)
from pyspark.ml import Pipeline

DEFAULT_INPUT_PATH = "data/amazon_reviews/"
DEFAULT_OUTPUT_PATH = "output/features/"
NUM_TF_FEATURES = 4096
NUM_NGRAM = 2


def create_spark_session(app_name="ClinicalNLP-DistributedPipeline", local=False):
    builder = SparkSession.builder.appName(app_name)

    if local:
        builder = (
            builder
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.sql.shuffle.partitions", "8")
        )
    else:
        builder = (
            builder
            .config("spark.sql.shuffle.partitions", "200")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.sql.parquet.compression.codec", "snappy")
        )

    return builder.getOrCreate()


def ingest_data(spark, input_path, sample_fraction=None):
    print("STAGE 1: DATA INGESTION")

    df = spark.read.parquet(input_path)

    if sample_fraction:
        df = df.sample(fraction=sample_fraction, seed=42)

    columns = df.columns
    if "review_body" in columns:
        df = df.select(
            F.col("review_id").alias("doc_id"),
            F.col("review_body").alias("text"),
            F.col("star_rating").alias("rating"),
            F.col("product_title").alias("source"),
        )
    elif "content" in columns:
        df = df.withColumn("doc_id", F.monotonically_increasing_id().cast("string"))
        df = df.select(
            F.col("doc_id"),
            F.col("content").alias("text"),
            F.col("label").alias("rating"),
            F.col("title").alias("source"),
        )
    else:
        if "text" not in columns:
            raise ValueError(f"Unknown schema. Columns: {columns}")

    df = df.filter(
        F.col("text").isNotNull() & (F.length(F.col("text")) > 10)
    )

    num_partitions = max(df.rdd.getNumPartitions(), 100)
    df = df.repartition(num_partitions)

    count = df.count()
    print(f"  Records loaded: {count:,}")
    print(f"  Partitions: {df.rdd.getNumPartitions()}")

    return df, count


def clean_text(df):
    print("\nSTAGE 2: TEXT CLEANING")

    df = df.withColumn(
        "text_clean",
        F.lower(F.col("text"))
    ).withColumn(
        "text_clean",
        F.regexp_replace("text_clean", "<[^>]+>", " ")
    ).withColumn(
        "text_clean",
        F.regexp_replace("text_clean", "[^a-z0-9\\s\\-]", " ")  # keep hyphens for medical terms
    ).withColumn(
        "text_clean",
        F.regexp_replace("text_clean", "\\s+", " ")
    ).withColumn(
        "text_clean",
        F.trim("text_clean")
    ).withColumn(
        "text_length",
        F.length("text_clean")
    ).withColumn(
        "word_count",
        F.size(F.split("text_clean", " "))
    )

    df = df.filter(
        (F.col("word_count") >= 5) & (F.col("word_count") <= 2000)
    )

    stats = df.agg(
        F.count("*").alias("total"),
        F.avg("word_count").alias("avg_words"),
        F.max("word_count").alias("max_words"),
    ).collect()[0]

    print(f"  Records after cleaning: {stats['total']:,}")
    print(f"  Avg word count: {stats['avg_words']:.1f}")
    print(f"  Max word count: {stats['max_words']}")

    return df


def build_feature_pipeline():
    tokenizer = Tokenizer(inputCol="text_clean", outputCol="words")

    clinical_stop_words = StopWordsRemover.loadDefaultStopWords("english") + [
        "patient", "also", "use", "used", "using", "would", "could",
        "one", "two", "get", "got", "like", "really", "much",
    ]
    stopwords_remover = StopWordsRemover(
        inputCol="words",
        outputCol="filtered_words",
        stopWords=clinical_stop_words,
    )

    ngram = NGram(n=NUM_NGRAM, inputCol="filtered_words", outputCol="bigrams")

    hashing_tf = HashingTF(
        inputCol="filtered_words",
        outputCol="raw_features",
        numFeatures=NUM_TF_FEATURES,
    )
    idf = IDF(inputCol="raw_features", outputCol="tfidf_features")

    bigram_tf = HashingTF(
        inputCol="bigrams",
        outputCol="bigram_raw_features",
        numFeatures=NUM_TF_FEATURES,
    )
    bigram_idf = IDF(inputCol="bigram_raw_features", outputCol="bigram_tfidf_features")

    pipeline = Pipeline(stages=[
        tokenizer,
        stopwords_remover,
        ngram,
        hashing_tf,
        idf,
        bigram_tf,
        bigram_idf,
    ])

    return pipeline


def extract_features(df, pipeline):
    print("\nSTAGE 3: FEATURE EXTRACTION (TF-IDF + N-grams)")

    start = time.time()
    model = pipeline.fit(df)
    features_df = model.transform(df)
    elapsed = time.time() - start

    print(f"  Pipeline fit + transform: {elapsed:.1f}s")
    print(f"  TF-IDF features: {NUM_TF_FEATURES} dimensions")
    print(f"  Bigram features: {NUM_TF_FEATURES} dimensions")

    return features_df, model


def compute_corpus_statistics(df):
    print("\nSTAGE 4: CORPUS STATISTICS")

    words_df = df.select(F.explode("filtered_words").alias("word"))
    word_counts = (
        words_df
        .groupBy("word")
        .count()
        .orderBy(F.desc("count"))
    )

    top_words = word_counts.limit(50).collect()

    length_stats = df.agg(
        F.count("*").alias("total_docs"),
        F.avg("word_count").alias("avg_doc_length"),
        F.stddev("word_count").alias("std_doc_length"),
        F.percentile_approx("word_count", 0.5).alias("median_doc_length"),
        F.percentile_approx("word_count", 0.95).alias("p95_doc_length"),
        F.sum("word_count").alias("total_words"),
    ).collect()[0]

    rating_dist = (
        df.groupBy("rating")
        .count()
        .orderBy("rating")
        .collect()
    )

    stats = {
        "total_documents": length_stats["total_docs"],
        "total_words": length_stats["total_words"],
        "avg_document_length": round(length_stats["avg_doc_length"], 1),
        "std_document_length": round(length_stats["std_doc_length"], 1),
        "median_document_length": length_stats["median_doc_length"],
        "p95_document_length": length_stats["p95_doc_length"],
        "top_20_words": [
            {"word": row["word"], "count": row["count"]}
            for row in top_words[:20]
        ],
        "rating_distribution": {
            str(row["rating"]): row["count"] for row in rating_dist
        },
    }

    print(f"  Total documents: {stats['total_documents']:,}")
    print(f"  Total words: {stats['total_words']:,}")
    print(f"  Avg doc length: {stats['avg_document_length']} words")
    print(f"  Vocabulary (top 5): {[w['word'] for w in stats['top_20_words'][:5]]}")

    return stats


def write_output(features_df, stats, output_path, local=False):
    print("\nSTAGE 5: WRITING OUTPUT")

    output_df = features_df.select(
        "doc_id",
        "text_clean",
        "rating",
        "source",
        "word_count",
        "filtered_words",
        "tfidf_features",
        "bigram_tfidf_features",
    )

    features_path = f"{output_path}/tfidf_features/"
    output_df.write.mode("overwrite").partitionBy("rating").parquet(features_path)
    print(f"  Features written to: {features_path}")

    metadata_path = f"{output_path}/metadata/"
    if local:
        import os
        os.makedirs(metadata_path, exist_ok=True)
        with open(f"{metadata_path}/corpus_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
    else:
        stats_json = json.dumps(stats)
        stats_df = features_df.sparkSession.createDataFrame(
            [(stats_json,)], ["stats_json"]
        )
        stats_df.write.mode("overwrite").text(f"{metadata_path}/corpus_stats")

    print(f"  Metadata written to: {metadata_path}")

    num_output_partitions = output_df.rdd.getNumPartitions()
    print(f"  Output partitions: {num_output_partitions}")

    return features_path


def run_pipeline(input_path, output_path, sample_fraction=None, local=False):
    print(f"Distributed NLP Pipeline ({'LOCAL' if local else 'EMR CLUSTER'})\n")

    pipeline_start = time.time()
    spark = create_spark_session(local=local)
    spark.sparkContext.setLogLevel("WARN")

    try:
        df, raw_count = ingest_data(spark, input_path, sample_fraction)
        df = clean_text(df)
        df.cache()

        feature_pipeline = build_feature_pipeline()
        features_df, model = extract_features(df, feature_pipeline)
        stats = compute_corpus_statistics(features_df)
        write_output(features_df, stats, output_path, local=local)

        elapsed = time.time() - pipeline_start
        throughput = raw_count / elapsed if elapsed > 0 else 0

        stats["pipeline_metrics"] = {
            "total_runtime_seconds": round(elapsed, 1),
            "throughput_docs_per_second": round(throughput, 1),
            "input_records": raw_count,
            "spark_partitions": df.rdd.getNumPartitions(),
            "mode": "local" if local else "emr",
        }

        print(f"\n  Runtime: {elapsed:.1f}s")
        print(f"  Throughput: {throughput:.1f} docs/sec")
        print(f"  Input: {raw_count:,} records")
        print(f"  Output: {output_path}")

        return stats

    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed NLP Pipeline")
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, help="S3 input path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="S3 output path")
    parser.add_argument("--sample", type=float, default=None, help="Sample fraction (0-1)")
    parser.add_argument("--local", action="store_true", help="Run in local mode")
    args = parser.parse_args()

    stats = run_pipeline(args.input, args.output, args.sample, args.local)

    if args.local:
        import os
        os.makedirs("results", exist_ok=True)
        with open("results/pipeline_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
