import json
from pathlib import Path
from tools import analyze_clinical_document, analyze_document_file

SAMPLE_DIR = Path("sample_docs")


def main():
    print("Clinical document analysis\n")

    for doc_path in sorted(SAMPLE_DIR.glob("*.txt")):
        print(f"--- {doc_path.name} ---")

        text = doc_path.read_text().strip()
        print(f"  Preview: {text[:100]}...")

        result = json.loads(analyze_document_file(str(doc_path)))

        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue

        print(f"  Tokens: {result['num_tokens']}")
        print(f"  Entities found: {result['entities_found']}")
        print(f"  Summary: {result['entity_summary']}")

        for entity in result["entities"]:
            print(f"    [{entity['label']:>8}] {entity['text']:<30} "
                  f"(confidence: {entity['confidence']})")
        print()

    # inline text test
    print("--- Inline text test ---")

    test_text = (
        "Patient diagnosed with rheumatoid arthritis. Started hydroxychloroquine "
        "200mg BID and methotrexate 15mg weekly. Monitor CBC and liver function "
        "every 3 months. Folic acid 1mg daily to reduce methotrexate side effects."
    )
    print(f"  Text: {test_text[:80]}...")

    result = json.loads(analyze_clinical_document(test_text))
    print(f"  Entities found: {result['entities_found']}")
    for entity in result["entities"]:
        print(f"    [{entity['label']:>8}] {entity['text']:<30} "
              f"(confidence: {entity['confidence']})")


if __name__ == "__main__":
    main()
