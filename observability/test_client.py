import json
import time
import requests
import random
from tqdm import tqdm
from pathlib import Path

SERVER_URL = "http://localhost:8000"
RESULTS_DIR = Path("results")

CLINICAL_TEXTS = [
    "Patient was prescribed metformin 500mg twice daily for type 2 diabetes mellitus.",
    "Blood pressure medication lisinopril 10mg caused persistent dry cough. Switched to losartan 50mg.",
    "Diagnosed with chronic migraine. Started sumatriptan 50mg PRN and topiramate 25mg daily for prophylaxis.",
    "Allergic reaction to amoxicillin documented. Rash and mild angioedema. Prescribed azithromycin as alternative.",
    "Post-operative recovery after total knee arthroplasty progressing well. Physical therapy three times weekly.",
    "Depression screening PHQ-9 score 14. Initiated sertraline 50mg daily. Behavioral health referral placed.",
    "Annual wellness exam. BMI 28.5. Lipid panel ordered. Counseled on diet and exercise.",
    "Thyroid function tests show TSH 7.2. Started levothyroxine 25mcg daily. Recheck in 6 weeks.",
    "Asthma exacerbation triggered by seasonal allergies. Increased fluticasone dose. Added montelukast.",
    "Diabetic foot exam normal. Monofilament testing intact bilaterally. No ulcers or calluses.",
    "Chest pain evaluation. Troponin negative. ECG normal sinus rhythm. Stress test scheduled.",
    "Urinary tract infection confirmed by culture. Prescribed nitrofurantoin 100mg BID for 5 days.",
    "Osteoarthritis of bilateral knees. Trial of celecoxib 200mg daily. Physical therapy referral.",
    "Insomnia management. Sleep hygiene counseling provided. Melatonin 3mg at bedtime recommended.",
    "Hypertension management. Amlodipine 5mg added to current regimen. Target BP below 130/80.",
    "COPD exacerbation. Prednisone taper initiated. Albuterol nebulizer treatments every 4 hours.",
    "Vitamin D deficiency. Level 18 ng/mL. Started ergocalciferol 50000 IU weekly for 8 weeks.",
    "Anxiety disorder. GAD-7 score 12. Discussed SSRI options. Patient prefers counseling first.",
    "Iron deficiency anemia. Ferritin 8. Started ferrous sulfate 325mg daily with vitamin C.",
    "Gout flare right great toe. Colchicine 0.6mg BID for 3 days. Allopurinol to start after resolution.",
]


def test_health():
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        print(f"Health: {resp.json()}")
        return resp.json()["status"] == "healthy"
    except requests.ConnectionError:
        print("ERROR: Server not running. Start with: python inference_server.py")
        return False


def send_prediction(text, request_id=None):
    payload = {"text": text}
    if request_id:
        payload["request_id"] = request_id

    resp = requests.post(f"{SERVER_URL}/predict", json=payload, timeout=30)
    return resp.json()


def run_load_test(num_requests=100, delay_ms=50):
    print(f"\nSending {num_requests} requests...")

    results = []
    for i in tqdm(range(num_requests)):
        text = random.choice(CLINICAL_TEXTS)
        try:
            result = send_prediction(text, request_id=f"test_{i:04d}")
            results.append({
                "request_id": result["request_id"],
                "latency_ms": result["latency_ms"],
                "num_entities": len(result["entities"]),
                "num_tokens": result["num_tokens"],
                "status": "success",
            })
        except Exception as e:
            results.append({
                "request_id": f"test_{i:04d}",
                "status": "error",
                "error": str(e),
            })

        if delay_ms > 0:
            time.sleep(delay_ms / 1000)

    return results


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Inference server test client\n")

    if not test_health():
        return

    # single request
    print("\n--- Single request ---")
    result = send_prediction(CLINICAL_TEXTS[0])
    print(f"Text: {CLINICAL_TEXTS[0][:60]}...")
    print(f"Entities found: {len(result['entities'])}")
    for entity in result["entities"]:
        print(f"  [{entity['label']}] {entity['text']} (confidence: {entity['confidence']})")
    print(f"Latency: {result['latency_ms']:.2f} ms")

    # load test
    print("\n--- Load test (100 requests) ---")
    results = run_load_test(num_requests=100, delay_ms=10)

    successes = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]
    latencies = [r["latency_ms"] for r in successes]

    print(f"\n  Successful: {len(successes)}/{len(results)}")
    print(f"  Errors: {len(errors)}")
    if latencies:
        print(f"  Avg latency: {sum(latencies)/len(latencies):.2f} ms")
        print(f"  P95 latency: {sorted(latencies)[int(len(latencies)*0.95)]:.2f} ms")
        print(f"  SLA violations (>{50}ms): {sum(1 for l in latencies if l > 50)}")

    with open(RESULTS_DIR / "load_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {RESULTS_DIR / 'load_test_results.json'}")

    print(f"\n  View metrics: {SERVER_URL}/metrics")
    print(f"  View health: {SERVER_URL}/health")


if __name__ == "__main__":
    main()
