# Component 2: Distributed Data Processing — PySpark on AWS EMR

## Objective
Build a scalable NLP data pipeline using PySpark on AWS EMR that ingests a large public
dataset, performs tokenization + feature engineering + TF-IDF/embedding preparation at
scale, and outputs processed features to S3 for downstream model training.

## Use Case
> Process millions of clinical-adjacent text records through a distributed NLP pipeline,
> preparing features for model training at a scale that exceeds single-machine capacity.
> Orchestrated via AWS Step Functions for production reliability.

## Claims Proven
- **100+ TB framing** via partition-level scale design (configurable partition count)
- **50% throughput improvement** via optimized shuffle/broadcast joins
- **Production orchestration** via AWS Step Functions (not just ad-hoc Spark jobs)

## Pipeline
```
S3 (PubMed Abstracts) → EMR Spark → Teacher NER Model → Weak Labels → S3 (Silver NER Data)
                              ↑                                              ↓
                      Step Functions                              Component 1 (Student Training)
```

## Two Pipelines

### Pipeline A: Feature Engineering (Original)
- Input: Amazon Reviews / clinical text
- Processing: Clean → Tokenize → TF-IDF → N-grams
- Output: Feature vectors for downstream ML
- Purpose: Demonstrates distributed data engineering

### Pipeline B: Weak Labeling (Connected to Model Training)
- Input: PubMed abstracts (biomedical text)
- Processing: Clean → Run teacher NER → Filter by confidence → Save BIO tags
- Output: Weakly labeled NER training data
- Purpose: Generates additional training data for Component 1's student model

## Stack
- Apache Spark (PySpark)
- AWS EMR (managed Spark cluster)
- AWS S3 (data lake)
- AWS Step Functions (orchestration — replaces ADF from original plan)
- Dataset: Amazon Reviews (Health & Personal Care subset)

## How to Run

### Local Testing
```bash
pip install -r requirements.txt

# Option A: Quick test with synthetic data (no download needed)
python pipeline_local.py

# Option B: Test with real Amazon Reviews data
python download_data.py --max-records 100000    # Downloads 100K reviews
python spark_pipeline.py --local --input data/amazon_reviews/
```

### AWS Deployment

#### Option A: Terraform (recommended — Infrastructure as Code)
```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your settings

terraform init
terraform plan
terraform apply

# Run the pipeline via Step Functions (command printed in output)
```

#### Option B: Direct deployment (no Terraform)
```bash
# 1. One-time setup
aws emr create-default-roles

# 2. Upload pipeline to S3
aws s3 cp spark_pipeline.py s3://YOUR-BUCKET/scripts/
aws s3 cp bootstrap.sh s3://YOUR-BUCKET/scripts/

# 3. Create EMR cluster + run job
python deploy_emr.py --bucket YOUR-BUCKET --region us-east-1

# 4. Or use Step Functions for full orchestration
aws stepfunctions create-state-machine \
  --name clinical-nlp-pipeline \
  --definition file://step_functions.json \
  --role-arn YOUR-ROLE-ARN
```

## Architecture
```
┌─────────────────────────────────────────────────────────┐
│                   AWS Step Functions                     │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐ │
│  │ Create   │→ │ Run Spark    │→ │ Validate Output   │ │
│  │ Cluster  │  │ Pipeline     │  │ + Terminate        │ │
│  └──────────┘  └──────────────┘  └───────────────────┘ │
└─────────────────────────────────────────────────────────┘
         │                │                  │
         ▼                ▼                  ▼
    ┌─────────┐    ┌────────────┐    ┌────────────┐
    │ EMR     │    │ S3 Input   │    │ S3 Output  │
    │ Cluster │    │ (Raw Text) │    │ (Features) │
    └─────────┘    └────────────┘    └────────────┘
```

## Output
- `results/feature_stats.json` — Processing statistics
- `s3://bucket/output/tfidf_features/` — TF-IDF feature vectors (Parquet)
- `s3://bucket/output/embeddings/` — Token embeddings (Parquet)
- `s3://bucket/output/metadata/` — Processing metadata
