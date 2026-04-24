import json
import os
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from spark_pipeline import (
    create_spark_session,
    clean_text,
    build_feature_pipeline,
    extract_features,
    compute_corpus_statistics,
    write_output,
)

RESULTS_DIR = Path("results")
LOCAL_OUTPUT = "output/local_test"

SAMPLE_DATA = [
    ("doc_001", "Patient reported improvement after taking ibuprofen 400mg twice daily for chronic lower back pain. No adverse effects noted during follow-up visit.", 4, "Clinical Note", "2024-01-15"),
    ("doc_002", "This vitamin D supplement has been great for my bone health. Doctor recommended 2000 IU daily and my levels improved significantly after three months.", 5, "Health Review", "2024-02-01"),
    ("doc_003", "Blood pressure medication caused dizziness and fatigue. Switched from lisinopril to losartan with better tolerance. Monitor renal function quarterly.", 3, "Clinical Note", "2024-01-20"),
    ("doc_004", "The glucometer is easy to use and gives consistent readings. Helps me track my blood sugar levels throughout the day. Battery life could be better.", 4, "Device Review", "2024-03-01"),
    ("doc_005", "Diagnosed with type 2 diabetes mellitus. Started metformin 500mg BID. Counseled on diet modification and exercise. Follow-up in 3 months with HbA1c.", 3, "Clinical Note", "2024-02-15"),
    ("doc_006", "Allergic reaction to amoxicillin - developed rash and mild swelling. Documented penicillin allergy in chart. Prescribed azithromycin as alternative.", 2, "Clinical Note", "2024-01-25"),
    ("doc_007", "This heating pad provides excellent relief for my arthritis pain. The auto-shutoff feature is a nice safety addition. Highly recommend for joint pain.", 5, "Health Review", "2024-03-10"),
    ("doc_008", "Post-operative recovery progressing well after knee replacement surgery. Physical therapy sessions three times weekly. Range of motion improving steadily.", 4, "Clinical Note", "2024-02-20"),
    ("doc_009", "The blood pressure monitor is accurate when compared to readings at the doctor office. Easy to use cuff and large display. Stores previous readings.", 5, "Device Review", "2024-01-30"),
    ("doc_010", "Chronic migraine management with sumatriptan PRN and topiramate daily prophylaxis. Headache diary shows reduction from 15 to 6 episodes per month.", 4, "Clinical Note", "2024-03-05"),
    ("doc_011", "Probiotic supplement helped with digestive issues after antibiotic course. Noticed improvement in bloating and regularity within two weeks of starting.", 4, "Health Review", "2024-02-10"),
    ("doc_012", "Annual wellness exam completed. BMI 28.5 - discussed weight management strategies. Lipid panel ordered. Up to date on immunizations. Colonoscopy due next year.", 3, "Clinical Note", "2024-01-10"),
    ("doc_013", "This pulse oximeter gives quick readings and the display is bright and easy to read. Useful for monitoring oxygen levels during respiratory illness.", 4, "Device Review", "2024-03-15"),
    ("doc_014", "Asthma exacerbation triggered by seasonal allergies. Increased inhaled corticosteroid dose. Added montelukast. Peak flow monitoring at home recommended.", 3, "Clinical Note", "2024-02-25"),
    ("doc_015", "Fish oil supplement for cardiovascular health. Taking 1000mg EPA DHA daily as recommended by cardiologist. No fishy aftertaste with this brand.", 4, "Health Review", "2024-01-05"),
    ("doc_016", "Thyroid function tests show subclinical hypothyroidism. TSH elevated at 7.2. Started levothyroxine 25mcg daily. Recheck labs in 6 weeks.", 3, "Clinical Note", "2024-03-20"),
    ("doc_017", "The TENS unit provides good pain relief for my shoulder injury. Multiple intensity settings and the pads stick well. Portable enough for travel.", 4, "Device Review", "2024-02-05"),
    ("doc_018", "Depression screening positive PHQ-9 score 14. Started sertraline 50mg daily. Referred to behavioral health for cognitive behavioral therapy. Safety plan reviewed.", 2, "Clinical Note", "2024-01-18"),
    ("doc_019", "Melatonin supplement helps me fall asleep faster. Taking 3mg about 30 minutes before bed. No grogginess in the morning unlike prescription sleep aids.", 4, "Health Review", "2024-03-25"),
    ("doc_020", "Diabetic foot exam completed. Monofilament testing normal bilaterally. No ulcers or calluses noted. Pedal pulses palpable. Continue daily foot inspections.", 4, "Clinical Note", "2024-02-28"),
]


def generate_larger_dataset(spark, base_data, multiplier=50):
    schema = StructType([
        StructField("doc_id", StringType(), False),
        StructField("text", StringType(), False),
        StructField("rating", IntegerType(), False),
        StructField("source", StringType(), False),
        StructField("date", StringType(), False),
    ])

    expanded_data = []
    for i in range(multiplier):
        for doc_id, text, rating, source, date in base_data:
            new_id = f"{doc_id}_v{i:03d}"
            if i % 3 == 0:
                text = text + " Follow-up appointment scheduled."
            elif i % 3 == 1:
                text = "Additional notes: " + text
            expanded_data.append((new_id, text, rating, source, date))

    df = spark.createDataFrame(expanded_data, schema)
    return df


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    os.makedirs(LOCAL_OUTPUT, exist_ok=True)

    print("Local pipeline test on synthetic data\n")

    spark = create_spark_session(local=True)
    spark.sparkContext.setLogLevel("WARN")

    try:
        print("  Generating test dataset...")
        df = generate_larger_dataset(spark, SAMPLE_DATA, multiplier=50)
        count = df.count()
        print(f"  Generated {count:,} records")

        df = clean_text(df)
        df.cache()

        feature_pipeline = build_feature_pipeline()
        features_df, model = extract_features(df, feature_pipeline)

        stats = compute_corpus_statistics(features_df)

        write_output(features_df, stats, LOCAL_OUTPUT, local=True)

        stats["pipeline_metrics"] = {
            "mode": "local_test",
            "input_records": count,
            "note": "Synthetic data — multiply by 1000x for EMR scale estimate",
        }

        with open(RESULTS_DIR / "pipeline_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n  Stats saved: {RESULTS_DIR / 'pipeline_stats.json'}")

        print("\n  Sample output:")
        features_df.select("doc_id", "word_count", "rating").show(5, truncate=False)

    finally:
        spark.stop()

    print("Done.")


if __name__ == "__main__":
    main()
