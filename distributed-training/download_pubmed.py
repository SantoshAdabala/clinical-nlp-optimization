import argparse
import json
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

RESULTS_DIR = Path("results")
DATA_DIR = Path("data/pubmed")

SEARCH_QUERIES = [
    "drug therapy clinical trial",
    "medication adverse effects treatment",
    "disease diagnosis pharmacotherapy",
    "clinical pharmacology drug interaction",
    "chemotherapy cancer treatment regimen",
    "antibiotic infection treatment",
    "cardiovascular disease medication",
    "diabetes mellitus drug therapy",
    "respiratory disease inhaled therapy",
    "autoimmune disease immunotherapy",
]

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def search_pubmed(query, max_results=1000):
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    resp = requests.get(PUBMED_SEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_abstracts(pmids, batch_size=200):
    abstracts = []

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i:i + batch_size]
        params = {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract",
        }

        resp = requests.get(PUBMED_FETCH_URL, params=params, timeout=60)
        resp.raise_for_status()

        xml = resp.text
        import re
        articles = xml.split("<PubmedArticle>")

        for article in articles[1:]:
            pmid_match = re.search(r"<PMID[^>]*>(\d+)</PMID>", article)
            pmid = pmid_match.group(1) if pmid_match else "unknown"

            abstract_parts = re.findall(r"<AbstractText[^>]*>(.*?)</AbstractText>", article, re.DOTALL)
            if abstract_parts:
                abstract = " ".join(abstract_parts)
                abstract = re.sub(r"<[^>]+>", "", abstract)
                abstract = abstract.strip()

                if len(abstract) > 50:
                    abstracts.append({
                        "pmid": f"pmid_{pmid}",
                        "text": abstract,
                    })

        time.sleep(0.4)  # rate limit: max 3 req/sec

    return abstracts


def main():
    parser = argparse.ArgumentParser(description="Download PubMed abstracts")
    parser.add_argument("--max-per-query", type=int, default=1000,
                        help="Max results per search query (default: 1000)")
    parser.add_argument("--bucket", type=str, default=None,
                        help="S3 bucket to upload to")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading PubMed abstracts\n")

    all_pmids = set()

    for query in tqdm(SEARCH_QUERIES, desc="Searching"):
        pmids = search_pubmed(query, max_results=args.max_per_query)
        all_pmids.update(pmids)
        time.sleep(0.4)

    print(f"\n  Unique PMIDs found: {len(all_pmids)}")

    print("  Fetching abstracts...")
    pmid_list = list(all_pmids)
    abstracts = fetch_abstracts(pmid_list)

    print(f"  Abstracts downloaded: {len(abstracts)}")

    df = pd.DataFrame(abstracts)
    output_file = DATA_DIR / "pubmed_abstracts.parquet"
    df.to_parquet(output_file, index=False)
    print(f"  Saved to: {output_file}")
    print(f"  File size: {output_file.stat().st_size / (1024*1024):.1f} MB")

    if args.bucket:
        import boto3
        s3 = boto3.client("s3")
        s3_key = "data/pubmed/pubmed_abstracts.parquet"
        s3.upload_file(str(output_file), args.bucket, s3_key)
        print(f"  Uploaded to: s3://{args.bucket}/{s3_key}")

    stats = {
        "total_abstracts": len(abstracts),
        "search_queries": SEARCH_QUERIES,
        "avg_length": round(df["text"].str.len().mean(), 1),
        "output_file": str(output_file),
    }
    with open(RESULTS_DIR / "pubmed_download_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Stats: {RESULTS_DIR / 'pubmed_download_stats.json'}")


if __name__ == "__main__":
    main()
